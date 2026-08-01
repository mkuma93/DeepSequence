#!/usr/bin/env python3
"""
Causal recursive multi-horizon rollout helpers (v1.6 feature contract).

Protocol
--------
Standing after observing day ``t`` (true history with ds <= t), forecast
``y_{t+1}, ..., y_{t+H}``.

- Calendar / holiday distances at ``t+h`` come from the real panel date
  (known future).
- Lags + intermittent state are updated with predicted demand after each step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from deepsequence_hierarchical_attention.intermittent_features import (
    SKUDemandState,
    empty_state,
)
from deepsequence_hierarchical_attention.eval.helpers import cummae_from_rollout, kpi_block


PredictFn = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, Optional[np.ndarray]]]
# predict_fn(X[n, F] or windows[n, L, C], sku[n, 1]) -> (yhat[n], p[n] or None)


@dataclass
class SkuTimeline:
    sku: str
    dates: np.ndarray  # datetime64
    y: np.ndarray  # float32
    holidays: np.ndarray  # [n, n_hol]
    time_index: np.ndarray  # raw days since epoch
    cyclical: np.ndarray  # [n, 6]


def _cyclical_from_dates(dates: pd.DatetimeIndex) -> np.ndarray:
    dow = dates.dayofweek.to_numpy()
    month = dates.month.to_numpy()
    doy = dates.dayofyear.to_numpy()
    cols = [
        np.sin(2 * np.pi * dow / 7),
        np.cos(2 * np.pi * dow / 7),
        np.sin(2 * np.pi * month / 12),
        np.cos(2 * np.pi * month / 12),
        np.sin(2 * np.pi * doy / 365.25),
        np.cos(2 * np.pi * doy / 365.25),
    ]
    return np.stack(cols, axis=1).astype(np.float32)


def build_sku_timelines(
    panel: pd.DataFrame,
    holidays: pd.DataFrame,
    holiday_names: Sequence[str],
) -> Dict[str, SkuTimeline]:
    """panel/holidays must be row-aligned; columns id_var, ds, Quantity."""
    df = panel.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df["id_var"] = df["id_var"].astype(str)
    h = holidays[list(holiday_names)].reset_index(drop=True)
    df = df.reset_index(drop=True)
    out: Dict[str, SkuTimeline] = {}
    for sku, g in df.groupby("id_var", sort=False):
        g = g.sort_values("ds", kind="mergesort")
        order = g.index.to_numpy()
        dates = pd.to_datetime(g["ds"])
        epoch = pd.Timestamp("1970-01-01")
        time_index = (dates - epoch).dt.days.to_numpy(dtype=np.float64)
        out[str(sku)] = SkuTimeline(
            sku=str(sku),
            dates=dates.dt.normalize().to_numpy(),
            y=g["Quantity"].to_numpy(np.float32),
            holidays=h.loc[order, list(holiday_names)].to_numpy(np.float32),
            time_index=time_index.astype(np.float32),
            cyclical=_cyclical_from_dates(pd.DatetimeIndex(dates)),
        )
    return out


def state_after_history(tl: SkuTimeline, end_idx: int, max_lag: int = 7) -> SKUDemandState:
    """Consume true y[0..end_idx] inclusive into a fresh state."""
    st = empty_state(max_lag=max_lag)
    for i in range(end_idx + 1):
        st.update(pd.Timestamp(tl.dates[i]), float(tl.y[i]))
    return st


def assemble_feature_row(
    tl: SkuTimeline,
    idx: int,
    state: SKUDemandState,
    lag_periods: Sequence[int],
    tmin: float,
    span: float,
) -> np.ndarray:
    """28-d v1.6 row: trend(norm) + cyclical + lags + intermittent + holidays."""
    date = pd.Timestamp(tl.dates[idx])
    reg = state.features_at(date, lags=lag_periods)
    time_n = (float(tl.time_index[idx]) - tmin) / span
    lag_vals = [float(reg[f"lag_{p}"]) for p in lag_periods]
    inter = [
        float(reg["days_since_last_sale"]),
        float(reg["last_sale_quantity"]),
        float(reg["lifetime_cumsum"]),
    ]
    return np.concatenate(
        [
            np.array([time_n], dtype=np.float32),
            tl.cyclical[idx],
            np.asarray(lag_vals, dtype=np.float32),
            np.asarray(inter, dtype=np.float32),
            tl.holidays[idx],
        ]
    )


def collect_origins(
    timelines: Dict[str, SkuTimeline],
    sku_map: Dict[str, int],
    horizon: int,
    origin_split_mask: Optional[Dict[str, np.ndarray]] = None,
    max_origins_per_sku: Optional[int] = None,
    seed: int = 42,
) -> List[Tuple[str, int]]:
    """
    Origins are indices ``t`` such that ``t+horizon < n``.

    If ``origin_split_mask[sku]`` is provided (bool per row), only those rows
    may be origins (e.g. test-split days).
    """
    rng = np.random.default_rng(seed)
    origins: List[Tuple[str, int]] = []
    for sku, tl in timelines.items():
        if sku not in sku_map:
            continue
        n = len(tl.y)
        if n <= horizon:
            continue
        cand = np.arange(0, n - horizon)
        if origin_split_mask is not None and sku in origin_split_mask:
            m = origin_split_mask[sku]
            cand = cand[m[cand]]
        if len(cand) == 0:
            continue
        if max_origins_per_sku is not None and len(cand) > max_origins_per_sku:
            cand = rng.choice(cand, size=max_origins_per_sku, replace=False)
            cand = np.sort(cand)
        for t in cand:
            origins.append((sku, int(t)))
    return origins


def rollout_tabular(
    timelines: Dict[str, SkuTimeline],
    origins: Sequence[Tuple[str, int]],
    sku_map: Dict[str, int],
    predict_fn: PredictFn,
    lag_periods: Sequence[int],
    tmin: float,
    span: float,
    horizon: int,
    batch_size: int = 2048,
) -> Dict[str, np.ndarray]:
    """Recursive tabular rollout -> y_true/yhat/p [n_origins, H], skus."""
    n = len(origins)
    H = int(horizon)
    n_hol = timelines[origins[0][0]].holidays.shape[1]
    n_feat = 1 + 6 + len(lag_periods) + 3 + n_hol
    y_true = np.zeros((n, H), np.float32)
    yhat = np.zeros((n, H), np.float32)
    p_out = np.zeros((n, H), np.float32)
    skus = np.array([o[0] for o in origins], dtype=object)
    has_p = True

    states = [
        state_after_history(timelines[sku], t_idx, max_lag=max(lag_periods))
        for sku, t_idx in origins
    ]

    for h in range(H):
        Xb = np.zeros((n, n_feat), np.float32)
        skb = np.zeros((n, 1), np.int32)
        for i, (sku, t_idx) in enumerate(origins):
            tl = timelines[sku]
            pred_idx = t_idx + 1 + h
            y_true[i, h] = tl.y[pred_idx]
            Xb[i] = assemble_feature_row(tl, pred_idx, states[i], lag_periods, tmin, span)
            skb[i, 0] = sku_map[sku]

        yh_parts, p_parts = [], []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            yh, p = predict_fn(Xb[start:end], skb[start:end])
            yh_parts.append(np.asarray(yh, np.float32).reshape(-1))
            if p is None:
                has_p = False
                p_parts.append(np.zeros(end - start, np.float32))
            else:
                p_parts.append(np.asarray(p, np.float32).reshape(-1))
        yhat[:, h] = np.concatenate(yh_parts)
        p_out[:, h] = np.concatenate(p_parts)

        for i, (sku, t_idx) in enumerate(origins):
            pred_idx = t_idx + 1 + h
            states[i].update(pd.Timestamp(timelines[sku].dates[pred_idx]), float(yhat[i, h]))

    return {
        "y_true": y_true,
        "yhat": yhat,
        "p": p_out if has_p else None,
        "skus": skus,
    }


HybridPredictFn = Callable[
    [np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, Optional[np.ndarray]],
]
# predict_fn(X[n,F], sku[n,1], windows[n,L,C]) -> (yhat[n], p[n] or None)


def rollout_hybrid(
    timelines: Dict[str, SkuTimeline],
    origins: Sequence[Tuple[str, int]],
    sku_map: Dict[str, int],
    predict_fn: HybridPredictFn,
    lag_periods: Sequence[int],
    tmin: float,
    span: float,
    horizon: int,
    lookback: int,
    batch_size: int = 1024,
) -> Dict[str, np.ndarray]:
    """Recursive hybrid rollout: current-row tabular features + lookback window.

    Lookback ends the day before the prediction date (same as sequence models).
    Demand path uses true y through the origin, then recursive yhat.
    """
    n = len(origins)
    H = int(horizon)
    L = int(lookback)
    n_hol = timelines[origins[0][0]].holidays.shape[1]
    n_feat = 1 + 6 + len(lag_periods) + 3 + n_hol
    n_ch = 1 + n_feat

    y_true = np.zeros((n, H), np.float32)
    yhat = np.zeros((n, H), np.float32)
    p_out = np.zeros((n, H), np.float32)
    skus = np.array([o[0] for o in origins], dtype=object)
    has_p = True

    demand_paths: List[List[float]] = []
    for sku, t_idx in origins:
        tl = timelines[sku]
        demand_paths.append([float(x) for x in tl.y[: t_idx + 1]])

    for h in range(H):
        Xb = np.zeros((n, n_feat), np.float32)
        windows = np.zeros((n, L, n_ch), np.float32)
        skb = np.zeros((n, 1), np.int32)

        for i, (sku, t_idx) in enumerate(origins):
            tl = timelines[sku]
            pred_idx = t_idx + 1 + h
            y_true[i, h] = tl.y[pred_idx]
            skb[i, 0] = sku_map[sku]
            path = demand_paths[i]

            st = empty_state(max_lag=max(lag_periods))
            for k in range(0, pred_idx):
                qty_k = float(path[k]) if k < len(path) else 0.0
                st.update(pd.Timestamp(tl.dates[k]), qty_k)
            Xb[i] = assemble_feature_row(tl, pred_idx, st, lag_periods, tmin, span)

            st_lb = empty_state(max_lag=max(lag_periods))
            start_j = pred_idx - L
            for k in range(0, max(0, start_j)):
                st_lb.update(
                    pd.Timestamp(tl.dates[k]),
                    float(path[k]) if k < len(path) else 0.0,
                )
            for li, j in enumerate(range(start_j, pred_idx)):
                if j < 0:
                    windows[i, li, :] = 0.0
                    continue
                feat = assemble_feature_row(tl, j, st_lb, lag_periods, tmin, span)
                qty = float(path[j]) if j < len(path) else 0.0
                windows[i, li, 0] = qty
                windows[i, li, 1:] = feat
                st_lb.update(pd.Timestamp(tl.dates[j]), qty)

        yh_parts, p_parts = [], []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            yh, p = predict_fn(
                Xb[start:end], skb[start:end], windows[start:end]
            )
            yh_parts.append(np.asarray(yh, np.float32).reshape(-1))
            if p is None:
                has_p = False
                p_parts.append(np.zeros(end - start, np.float32))
            else:
                p_parts.append(np.asarray(p, np.float32).reshape(-1))
        yhat[:, h] = np.concatenate(yh_parts)
        p_out[:, h] = np.concatenate(p_parts)

        for i, (_sku, _t_idx) in enumerate(origins):
            demand_paths[i].append(float(yhat[i, h]))

    return {
        "y_true": y_true,
        "yhat": yhat,
        "p": p_out if has_p else None,
        "skus": skus,
    }


def rollout_sequence(
    timelines: Dict[str, SkuTimeline],
    origins: Sequence[Tuple[str, int]],
    sku_map: Dict[str, int],
    predict_fn: PredictFn,
    lag_periods: Sequence[int],
    tmin: float,
    span: float,
    horizon: int,
    lookback: int,
    batch_size: int = 1024,
) -> Dict[str, np.ndarray]:
    """
    Recursive sequence rollout.

    Lookback ends the day before the prediction date. Demand uses true y for
    dates <= origin and recursive yhat afterward.
    """
    n = len(origins)
    H = int(horizon)
    L = int(lookback)
    n_hol = timelines[origins[0][0]].holidays.shape[1]
    n_feat = 1 + 6 + len(lag_periods) + 3 + n_hol
    n_ch = 1 + n_feat

    y_true = np.zeros((n, H), np.float32)
    yhat = np.zeros((n, H), np.float32)
    p_out = np.zeros((n, H), np.float32)
    skus = np.array([o[0] for o in origins], dtype=object)
    has_p = True

    demand_paths: List[List[float]] = []
    for sku, t_idx in origins:
        tl = timelines[sku]
        demand_paths.append([float(x) for x in tl.y[: t_idx + 1]])

    for h in range(H):
        windows = np.zeros((n, L, n_ch), np.float32)
        skb = np.zeros((n, 1), np.int32)
        for i, (sku, t_idx) in enumerate(origins):
            tl = timelines[sku]
            pred_idx = t_idx + 1 + h
            y_true[i, h] = tl.y[pred_idx]
            skb[i, 0] = sku_map[sku]

            st_lb = empty_state(max_lag=max(lag_periods))
            path = demand_paths[i]
            start_j = pred_idx - L
            for k in range(0, max(0, start_j)):
                st_lb.update(pd.Timestamp(tl.dates[k]), float(path[k]))

            for li, j in enumerate(range(start_j, pred_idx)):
                if j < 0:
                    windows[i, li, :] = 0.0
                    continue
                feat = assemble_feature_row(tl, j, st_lb, lag_periods, tmin, span)
                qty = float(path[j]) if j < len(path) else 0.0
                windows[i, li, 0] = qty
                windows[i, li, 1:] = feat
                st_lb.update(pd.Timestamp(tl.dates[j]), qty)

        yh_parts, p_parts = [], []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            yh, p = predict_fn(windows[start:end], skb[start:end])
            yh_parts.append(np.asarray(yh, np.float32).reshape(-1))
            if p is None:
                has_p = False
                p_parts.append(np.zeros(end - start, np.float32))
            else:
                p_parts.append(np.asarray(p, np.float32).reshape(-1))
        yhat[:, h] = np.concatenate(yh_parts)
        p_out[:, h] = np.concatenate(p_parts)

        for i, (sku, t_idx) in enumerate(origins):
            pred_idx = t_idx + 1 + h
            yh = float(yhat[i, h])
            demand_paths[i].append(yh)

    return {
        "y_true": y_true,
        "yhat": yhat,
        "p": p_out if has_p else None,
        "skus": skus,
    }


def rollout_direct_tabular(
    timelines: Dict[str, SkuTimeline],
    origins: Sequence[Tuple[str, int]],
    sku_map: Dict[str, int],
    predict_fn: PredictFn,
    lag_periods: Sequence[int],
    tmin: float,
    span: float,
    horizon: int,
    batch_size: int = 4096,
) -> dict:
    """
    Direct multi-horizon predict: after observing ``t``, build features at
    ``t+1`` with true history through ``t``, then predict all H steps at once.

    ``predict_fn(X[n,F], sku[n,1]) -> (yhat[n,H], p[n,H] or None)``.
    """
    H = int(horizon)
    n = len(origins)
    y_true = np.zeros((n, H), np.float32)
    yhat = np.zeros((n, H), np.float32)
    p_out = np.zeros((n, H), np.float32)
    skus = np.array([o[0] for o in origins], dtype=object)
    X = np.zeros((n, 28), np.float32)
    skb = np.zeros((n, 1), np.int32)
    has_p = True

    for i, (sku, t_idx) in enumerate(origins):
        tl = timelines[sku]
        pred0 = t_idx + 1
        for h in range(H):
            y_true[i, h] = tl.y[pred0 + h]
        st = state_after_history(tl, t_idx, max_lag=max(lag_periods))
        X[i] = assemble_feature_row(tl, pred0, st, lag_periods, tmin, span)
        skb[i, 0] = sku_map[sku]

    yh_parts, p_parts = [], []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yh, p = predict_fn(X[start:end], skb[start:end])
        yh = np.asarray(yh, np.float32)
        if yh.ndim == 1:
            yh = yh.reshape(-1, 1)
        yh_parts.append(yh)
        if p is None:
            has_p = False
            p_parts.append(np.zeros_like(yh))
        else:
            pp = np.asarray(p, np.float32)
            if pp.ndim == 1:
                pp = pp.reshape(-1, 1)
            p_parts.append(pp)
    yhat = np.concatenate(yh_parts, axis=0)
    p_mat = np.concatenate(p_parts, axis=0)
    if yhat.shape[1] < H:
        raise ValueError(f"direct predict returned H={yhat.shape[1]}, expected {H}")
    yhat = yhat[:, :H]
    p_mat = p_mat[:, :H]
    return {
        "y_true": y_true,
        "yhat": yhat,
        "p": p_mat if has_p else None,
        "skus": skus,
    }


def horizon_metrics(
    y_true: np.ndarray,
    yhat: np.ndarray,
    p: Optional[np.ndarray],
    skus: np.ndarray,
    volume_map: dict,
    report_horizons: Sequence[int] = (1, 7, 14, 21, 28),
    mase_scale: Optional[float] = None,
) -> dict:
    """Per-horizon KPIs + mean over 1..H. Horizons are 1-indexed.

    Pointwise IWMAE uses column ``h-1`` only. Cumulative planning error
    ``CumMAE(H) = mean |sum_{k=1..H} yhat - sum_{k=1..H} y|`` is reported
    under ``by_horizon_cum`` (and nested ``cummae`` / ``cum_iwmae`` aliases)
    without changing the primary pointwise ranking protocol.

    Callers typically pass horizons that fit within the rollout length H;
    entries with h > H are skipped.
    """
    H = y_true.shape[1]
    out = {"by_horizon": {}, "by_horizon_cum": {}, "mean_1_to_H": {}}
    yt = y_true.reshape(-1)
    yh = yhat.reshape(-1)
    pp = None if p is None else p.reshape(-1)
    out["mean_1_to_H"] = {
        "overall": kpi_block(yt, yh, pp, mase_scale=mase_scale),
        "n_origin_steps": int(yt.size),
    }

    for h in report_horizons:
        if h < 1 or h > H:
            continue
        col = h - 1
        yt_h = y_true[:, col]
        yh_h = yhat[:, col]
        pp_h = None if p is None else p[:, col]
        block = {"overall": kpi_block(yt_h, yh_h, pp_h, mase_scale=mase_scale)}
        bands = np.array([volume_map.get(s, "unk") for s in skus])
        for band in ("low", "mid", "high"):
            m = bands == band
            block[band] = kpi_block(
                yt_h[m],
                yh_h[m],
                None if pp_h is None else pp_h[m],
                mase_scale=mase_scale,
            )
        out["by_horizon"][str(h)] = block

    cum = cummae_from_rollout(
        y_true, yhat, p, report_horizons=report_horizons, mase_scale=mase_scale
    )
    # Nest overall + volume bands for parity with by_horizon.
    for h_key, flat in cum["by_horizon"].items():
        col = int(h_key) - 1
        yt_c = np.cumsum(y_true, axis=1)[:, col]
        yh_c = np.cumsum(np.maximum(yhat, 0.0), axis=1)[:, col]
        pp_h = None if p is None else p[:, col]
        block = {"overall": flat}
        bands = np.array([volume_map.get(s, "unk") for s in skus])
        for band in ("low", "mid", "high"):
            m = bands == band
            b = kpi_block(
                yt_c[m],
                yh_c[m],
                None if pp_h is None else pp_h[m],
                mase_scale=mase_scale,
            )
            b["cummae"] = b["mae_all"]
            b["cummae_rounded"] = b["mae_all_rounded"]
            b["cum_iwmae"] = b["iwmae"]
            b["cum_iwmae_rounded"] = b["iwmae_rounded"]
            block[band] = b
        out["by_horizon_cum"][h_key] = block
    return out
