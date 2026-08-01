#!/usr/bin/env python3
"""Per-series Prophet baseline on locked panels (Car Parts monthly + daily subset).

Prophet is fit independently per SKU (structural single-series baseline).
DeepSequence is a global multi-series model — protocol differences are intentional
and documented in the output JSON / PAPER.md.

Requires: prophet + cmdstanpy (installed in .venv-test).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from deepsequence_hierarchical_attention.eval.helpers import (
    filter_aligned,
    kpi_block,
    resolve_eval_seeds,
    select_eval_skus,
    train_mase_scale,
    train_volume_terciles,
)

# Suppress prophet/cmdstan chatter in workers
os.environ.setdefault("CMDSTAN_VERBOSE", "FALSE")
warnings.filterwarnings("ignore")


def _silence_cmdstan():
    import logging

    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        choices=("carparts", "daily"),
        default="carparts",
        help="carparts=monthly Monash; daily=enterprise panel subset.",
    )
    p.add_argument("--data_dir", default=None)
    p.add_argument("--sku_list", default=None)
    p.add_argument("--max_skus", type=int, default=800)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--horizons",
        default=None,
        help="Comma-separated 1-indexed horizons. Default: carparts 1,2,6; daily 1,28,60.",
    )
    p.add_argument(
        "--max_origins_per_sku",
        type=int,
        default=None,
        help="Cap test origins per SKU (daily). Default: carparts=all test months; daily=8.",
    )
    p.add_argument("--n_jobs", type=int, default=4)
    p.add_argument("--out_json", required=True)
    p.add_argument(
        "--yearly_seasonality",
        type=int,
        default=1,
        help="Prophet yearly_seasonality (1/0).",
    )
    p.add_argument(
        "--weekly_seasonality",
        type=int,
        default=None,
        help="Prophet weekly_seasonality. Default: off for monthly, on for daily.",
    )
    return p.parse_args()


def _default_data_dir(dataset: str) -> Path:
    if dataset == "carparts":
        return ROOT / "public_data/car_parts/panel"
    env = os.environ.get("DEEPSEQUENCE_DATA_DIR")
    if env:
        return Path(env)
    return Path(
        "/Users/mritunjaykumar/Library/CloudStorage/GoogleDrive-mritunjay.kmr1@gmail.com/"
        "My Drive/jubilant/data"
    )


def _default_sku_list(dataset: str) -> str:
    if dataset == "carparts":
        return "ab_runs/recompare/sku_list_carparts_data42.json"
    return "ab_runs/recompare/sku_list_daily_data42.json"


def _fit_predict_one(payload: dict) -> dict:
    """Worker: fit Prophet on one SKU train+val history; forecast test dates."""
    _silence_cmdstan()
    from prophet import Prophet

    hist = pd.DataFrame(
        {"ds": pd.to_datetime(payload["hist_ds"]), "y": np.asarray(payload["hist_y"], float)}
    )
    future_ds = pd.to_datetime(payload["future_ds"])
    yearly = bool(payload["yearly_seasonality"])
    weekly = bool(payload["weekly_seasonality"])
    # Intermittent monthly/daily: disable daily seasonality; keep floor at 0.
    m = Prophet(
        yearly_seasonality=yearly,
        weekly_seasonality=weekly,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=1.0,
    )
    # Short / sparse series: Prophet can fail; return zeros on failure.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(hist)
        fut = pd.DataFrame({"ds": future_ds})
        pred = m.predict(fut)
        yhat = np.maximum(pred["yhat"].to_numpy(np.float64), 0.0)
        ok = True
        err = None
    except Exception as exc:  # noqa: BLE001 — per-series robustness
        yhat = np.zeros(len(future_ds), dtype=np.float64)
        ok = False
        err = str(exc)
    return {
        "sku": payload["sku"],
        "yhat": yhat.tolist(),
        "y_true": list(payload["y_true"]),
        "ok": ok,
        "err": err,
    }


def _carparts_panel(data_dir: Path, chosen, mase_season: int = 12):
    train_df = pd.read_csv(data_dir / "train_split.csv", parse_dates=["ds"])
    val_df = pd.read_csv(data_dir / "val_split.csv", parse_dates=["ds"])
    test_df = pd.read_csv(data_dir / "test_split.csv", parse_dates=["ds"])
    # Holidays unused by classic Prophet here (monthly indicators differ).
    h_tr = pd.read_csv(data_dir / "holiday_features_train.csv")
    h_va = pd.read_csv(data_dir / "holiday_features_val.csv")
    h_te = pd.read_csv(data_dir / "holiday_features_test.csv")
    train_df, _ = filter_aligned(train_df, h_tr, chosen)
    val_df, _ = filter_aligned(val_df, h_va, chosen)
    test_df, _ = filter_aligned(test_df, h_te, chosen)
    volume_map, volume_stats = train_volume_terciles(train_df)
    mase_scale = train_mase_scale(train_df, season=mase_season)
    return train_df, val_df, test_df, volume_map, volume_stats, mase_scale


def _build_carparts_jobs(train_df, val_df, test_df, horizons: list[int]):
    """Fixed origin = last val month; forecast next H test months (same as MH eval)."""
    H = max(horizons)
    jobs = []
    for sku, te in test_df.groupby("id_var", sort=False):
        te = te.sort_values("ds")
        if len(te) < H:
            continue
        hist = pd.concat(
            [
                train_df.loc[train_df["id_var"] == sku, ["ds", "Quantity"]],
                val_df.loc[val_df["id_var"] == sku, ["ds", "Quantity"]],
            ],
            ignore_index=True,
        ).sort_values("ds")
        if len(hist) < 3 or hist["Quantity"].sum() <= 0:
            continue
        fut = te.iloc[:H]
        jobs.append(
            {
                "sku": str(sku),
                "hist_ds": hist["ds"].astype(str).tolist(),
                "hist_y": hist["Quantity"].astype(float).tolist(),
                "future_ds": fut["ds"].astype(str).tolist(),
                "y_true": fut["Quantity"].astype(float).tolist(),
            }
        )
    return jobs


def _build_daily_jobs(
    train_df, val_df, test_df, horizons: list[int], max_origins_per_sku: int
):
    """Multiple test origins per SKU; forecast H days ahead from each origin."""
    H = max(horizons)
    jobs = []
    for sku in sorted(train_df["id_var"].unique()):
        tr = train_df.loc[train_df["id_var"] == sku].sort_values("ds")
        va = val_df.loc[val_df["id_var"] == sku].sort_values("ds")
        te = test_df.loc[test_df["id_var"] == sku].sort_values("ds")
        if len(te) < H:
            continue
        base_hist = pd.concat([tr, va], ignore_index=True).sort_values("ds")
        # Origins: stand before each test day that has H future days.
        n_origins = max(1, len(te) - H + 1)
        if max_origins_per_sku is not None:
            n_origins = min(n_origins, int(max_origins_per_sku))
        # Evenly spaced origins across available window.
        if n_origins == 1:
            origin_idxs = [0]
        else:
            origin_idxs = np.linspace(0, len(te) - H, n_origins, dtype=int).tolist()
        for oi in origin_idxs:
            # History = train+val + test days strictly before origin target day.
            past_te = te.iloc[:oi]
            hist = pd.concat([base_hist, past_te], ignore_index=True).sort_values("ds")
            fut = te.iloc[oi : oi + H]
            if len(hist) < 7:
                continue
            jobs.append(
                {
                    "sku": f"{sku}__o{oi}",
                    "hist_ds": hist["ds"].astype(str).tolist(),
                    "hist_y": hist["Quantity"].astype(float).tolist(),
                    "future_ds": fut["ds"].astype(str).tolist(),
                    "y_true": fut["Quantity"].astype(float).tolist(),
                }
            )
    return jobs


def _aggregate(results, horizons, mase_scale):
    H = max(horizons)
    y_true = np.asarray([r["y_true"] for r in results], np.float32)
    yhat = np.asarray([r["yhat"] for r in results], np.float32)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        yhat = yhat.reshape(-1, 1)
    # Pad / trim to H
    y_true = y_true[:, :H]
    yhat = yhat[:, :H]
    out = {
        "n_series_ok": int(sum(1 for r in results if r["ok"])),
        "n_series_fail": int(sum(1 for r in results if not r["ok"])),
        "mean_1_to_H": kpi_block(
            y_true.reshape(-1), yhat.reshape(-1), None, mase_scale=mase_scale
        ),
        "by_horizon": {},
    }
    for h in horizons:
        col = h - 1
        out["by_horizon"][str(h)] = kpi_block(
            y_true[:, col], yhat[:, col], None, mase_scale=mase_scale
        )
    return out


def main():
    args = parse_args()
    try:
        from prophet import Prophet  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "prophet not installed. In .venv-test: uv pip install prophet"
        ) from exc

    dataset = args.dataset
    data_dir = Path(args.data_dir) if args.data_dir else _default_data_dir(dataset)
    sku_list = args.sku_list or _default_sku_list(dataset)
    data_seed, train_seed = resolve_eval_seeds(args.seed, None, None)

    if args.horizons:
        horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    else:
        horizons = [1, 2, 6] if dataset == "carparts" else [1, 28, 60]

    weekly = args.weekly_seasonality
    if weekly is None:
        weekly = 0 if dataset == "carparts" else 1
    yearly = bool(args.yearly_seasonality)

    train_df = pd.read_csv(data_dir / "train_split.csv", parse_dates=["ds"])
    universe = sorted(train_df["id_var"].unique())
    chosen = select_eval_skus(
        universe,
        max_skus=args.max_skus,
        data_seed=data_seed,
        sku_list_path=sku_list,
    )
    # Locked lists are length-800; honor --max_skus as a stratified prefix so
    # daily Prophet can run a tractable subset without changing the lock order.
    if len(chosen) > int(args.max_skus):
        rng = np.random.default_rng(int(data_seed))
        # Keep lock order for reproducibility: take evenly spaced indices.
        idx = np.linspace(0, len(chosen) - 1, int(args.max_skus), dtype=int)
        chosen = [chosen[i] for i in idx]
        print(
            f"Subset: took {len(chosen)} SKUs from locked list "
            f"(max_skus={args.max_skus}, evenly spaced)"
        )
    print(
        f"Prophet baseline: dataset={dataset} n_skus={len(chosen)} "
        f"horizons={horizons} data_seed={data_seed}"
    )

    mase_season = 12 if dataset == "carparts" else 7
    train_df, val_df, test_df, volume_map, volume_stats, mase_scale = _carparts_panel(
        data_dir, chosen, mase_season=mase_season
    )

    if dataset == "carparts":
        jobs = _build_carparts_jobs(train_df, val_df, test_df, horizons)
        max_origins = None
    else:
        max_origins = (
            8 if args.max_origins_per_sku is None else int(args.max_origins_per_sku)
        )
        jobs = _build_daily_jobs(
            train_df, val_df, test_df, horizons, max_origins_per_sku=max_origins
        )

    print(f"Jobs: {len(jobs)} (n_jobs={args.n_jobs})", flush=True)
    for j in jobs:
        j["yearly_seasonality"] = yearly
        j["weekly_seasonality"] = bool(weekly)

    _silence_cmdstan()
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_path.with_suffix(out_path.suffix + ".partial.json")
    # resume from checkpoint if present
    done_skus: set[str] = set()
    results: list[dict] = []
    if ckpt_path.exists():
        try:
            prev = json.loads(ckpt_path.read_text())
            results = list(prev.get("results", []))
            done_skus = {r["sku"] for r in results}
            print(f"Resume: loaded {len(results)} from {ckpt_path}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"Checkpoint unreadable ({exc}); starting fresh", flush=True)
            results = []
            done_skus = set()

    pending = [j for j in jobs if j["sku"] not in done_skus]
    print(f"Pending jobs: {len(pending)}/{len(jobs)}", flush=True)

    t0 = time.time()

    def _maybe_ckpt():
        ckpt_path.write_text(
            json.dumps({"results": results, "n": len(results)}, indent=0) + "\n"
        )

    if args.n_jobs <= 1:
        for i, j in enumerate(pending, 1):
            results.append(_fit_predict_one(j))
            if i % 25 == 0 or i == len(pending):
                print(f"  done {len(results)}/{len(jobs)}", flush=True)
                _maybe_ckpt()
    else:
        with ThreadPoolExecutor(max_workers=args.n_jobs) as ex:
            futs = {ex.submit(_fit_predict_one, j): j["sku"] for j in pending}
            for i, fut in enumerate(as_completed(futs), 1):
                results.append(fut.result())
                if i % 25 == 0 or i == len(futs):
                    print(f"  done {len(results)}/{len(jobs)}", flush=True)
                    _maybe_ckpt()
    elapsed = time.time() - t0

    metrics = _aggregate(results, horizons, mase_scale)
    out = {
        "config": {
            "model": "prophet",
            "protocol": (
                "per-series Prophet fit on train+val (carparts: fixed origin; "
                "daily: multi-origin recursive calendar forecast)"
            ),
            "dataset": dataset,
            "data_dir": str(data_dir),
            "sku_list": sku_list,
            "n_skus": len(chosen),
            "max_skus": args.max_skus,
            "seed": args.seed,
            "data_seed": data_seed,
            "train_seed": train_seed,
            "horizons": horizons,
            "max_origins_per_sku": max_origins,
            "n_jobs": args.n_jobs,
            "yearly_seasonality": yearly,
            "weekly_seasonality": bool(weekly),
            "daily_seasonality": False,
            "volume_stats": volume_stats,
            "honest_limits": (
                "Prophet is fit per series; DeepSequence/LightGBM/TSB are global "
                "or classical intermittent. No shared SKU pooling for Prophet. "
                "Holiday distances / intermittent state features are not injected "
                "as Prophet regressors in this baseline (calendar seasonality only)."
            ),
        },
        "models": {
            "prophet": {
                "train_predict_seconds": elapsed,
                **metrics,
            }
        },
    }
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    if ckpt_path.exists():
        ckpt_path.unlink()
    print(f"\nWrote {out_path} in {elapsed:.1f}s", flush=True)
    for h in horizons:
        iw = metrics["by_horizon"][str(h)]["iwmae"]
        print(f"  h={h} IWMAE={iw:.4f}", flush=True)


if __name__ == "__main__":
    main()
