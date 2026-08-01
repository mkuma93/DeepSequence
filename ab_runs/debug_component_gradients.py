#!/usr/bin/env python3
"""Gradient / sensitivity diagnostics for trend, seasonal, holiday experts.

Structural sensitivity at *random init* (no trained checkpoint). Interprets
how each expert is wired into the forecast, not learned importance on a panel.

Usage:
  TF_USE_LEGACY_KERAS=1 .venv-test/bin/python ab_runs/debug_component_gradients.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deepsequence_hierarchical_attention.components_lightweight import (  # noqa: E402
    build_hierarchical_model_lightweight,
)

# ---------------------------------------------------------------------------
# Config — small but representative of the live lightweight hierarchy
# ---------------------------------------------------------------------------
SEED = 42
BATCH = 64
N_SKUS = 16
N_TEMPORAL = 1
N_FOURIER = 6  # dow/month/year sin-cos style
N_HOLIDAY = 3  # small holiday set (full daily config has ~15)
N_LAG = 6
HIDDEN = 16
SKU_DIM = 4
EPS_FD = 1e-3
NAMES = ("trend", "seasonal", "holiday", "regressor")


def _set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _fourier_from_day(day_of_year: np.ndarray) -> np.ndarray:
    """Synthetic cyclical features matching feature_config style."""
    dow = (day_of_year.astype(np.float32) % 7.0)
    month = ((day_of_year % 365.25) / 30.4375) % 12.0
    yfrac = (day_of_year % 365.25) / 365.25
    return np.stack(
        [
            np.sin(2 * np.pi * dow / 7.0),
            np.cos(2 * np.pi * dow / 7.0),
            np.sin(2 * np.pi * month / 12.0),
            np.cos(2 * np.pi * month / 12.0),
            np.sin(2 * np.pi * yfrac),
            np.cos(2 * np.pi * yfrac),
        ],
        axis=-1,
    ).astype(np.float32)


def make_batch(
    *,
    holiday_mode: str = "mixed",
    batch: int = BATCH,
    seed: int = SEED,
) -> list[np.ndarray]:
    """Build synthetic inputs shaped like the live model.

    holiday_mode:
      - near: distances clustered near 0 (on/near holiday)
      - far:  distances large (>> changepoint span sense)
      - mixed: half near, half far
    """
    rng = np.random.default_rng(seed)
    # Time in [0, 1] — TrendComponentLightweight defaults time_min/max to that.
    t = np.linspace(0.05, 0.95, batch, dtype=np.float32).reshape(-1, 1)
    # Map t → fake day-of-year for seasonal Fourier
    day = (t.ravel() * 365.25).astype(np.float32)
    fourier = _fourier_from_day(day)

    if holiday_mode == "near":
        # On-holiday / short lead-lag (days). Non-negative: holiday
        # ChangepointReLU is defined on [0, 365].
        holiday = rng.uniform(0.0, 7.0, size=(batch, N_HOLIDAY)).astype(np.float32)
        holiday[:, 0] = rng.uniform(0.0, 2.0, size=batch).astype(np.float32)
    elif holiday_mode == "far":
        holiday = rng.uniform(120.0, 250.0, size=(batch, N_HOLIDAY)).astype(np.float32)
    elif holiday_mode == "hinge_active":
        # Past default first holiday hinge (~time_max / n_cps ≈ 73d at init).
        holiday = rng.uniform(80.0, 200.0, size=(batch, N_HOLIDAY)).astype(np.float32)
    else:
        near = rng.uniform(0.0, 7.0, size=(batch // 2, N_HOLIDAY)).astype(np.float32)
        far = rng.uniform(120.0, 250.0, size=(batch - batch // 2, N_HOLIDAY)).astype(
            np.float32
        )
        holiday = np.concatenate([near, far], axis=0)

    lag = rng.random((batch, N_LAG), dtype=np.float32) * 2.0
    # Intermittent-ish: often near-zero lags
    lag[:, :3] *= (rng.random((batch, 3)) < 0.35).astype(np.float32)
    sku = rng.integers(0, N_SKUS, size=(batch, 1), dtype=np.int32)
    return [t, fourier, holiday, lag, sku]


def build_model(*, use_intermittent: bool = True, trend_monotonic: bool = True):
    return build_hierarchical_model_lightweight(
        n_temporal_features=N_TEMPORAL,
        n_fourier_features=N_FOURIER,
        n_holiday_features=N_HOLIDAY,
        n_lag_features=N_LAG,
        n_skus=N_SKUS,
        n_changepoints=10,
        hidden_dim=HIDDEN,
        sku_embedding_dim=SKU_DIM,
        dropout_rate=0.0,  # deterministic diagnostics
        use_cross_layers=True,
        use_intermittent=use_intermittent,
        use_sku=True,
        trend_monotonic=trend_monotonic,
        horizon=1,
        output_activation="linear",
    )


def _as_tensors(batch: list[np.ndarray]) -> list[tf.Tensor]:
    return [tf.convert_to_tensor(x) for x in batch]


def build_probe(model: tf.keras.Model) -> tf.keras.Model:
    """Expose expert scalars, attention, base, gate, and final forecast."""
    layer_names = {layer.name for layer in model.layers}
    outs = {
        "trend": model.get_layer("trend").output,
        "seasonal": model.get_layer("seasonal").output,
        "holiday": model.get_layer("holiday").output,
        "regressor": model.get_layer("regressor").output,
        "attn": model.get_layer("component_attention_softmax").output,
        "final_forecast": model.output["final_forecast"],
    }
    if "base_forecast" in layer_names:
        outs["base_forecast"] = model.get_layer("base_forecast").output
    elif isinstance(model.output, dict) and "base_forecast" in model.output:
        outs["base_forecast"] = model.output["base_forecast"]
    else:
        # No intermittent gate: final IS the softplus base level.
        outs["base_forecast"] = model.output["final_forecast"]
    if isinstance(model.output, dict) and "non_zero_probability" in model.output:
        outs["non_zero_probability"] = model.output["non_zero_probability"]
    else:
        ones = tf.ones_like(outs["final_forecast"])
        outs["non_zero_probability"] = ones
    return tf.keras.Model(model.inputs, outs, name="component_grad_probe")


def grad_norms_wrt_experts(
    probe: tf.keras.Model, batch: list[np.ndarray]
) -> dict[str, float]:
    """Mean |∂ŷ/∂c| for each expert scalar c (ŷ = final_forecast)."""
    xt = _as_tensors(batch)
    with tf.GradientTape(persistent=True) as tape:
        o = probe(xt, training=False)
        # Watch intermediate expert scalars for ∂ŷ/∂c
        for name in NAMES:
            tape.watch(o[name])
        y_final = tf.reduce_sum(o["final_forecast"])
        y_base = tf.reduce_sum(o["base_forecast"])
    norms_final = {}
    norms_base = {}
    for name in NAMES:
        g_f = tape.gradient(y_final, o[name])
        g_b = tape.gradient(y_base, o[name])
        norms_final[name] = (
            float(tf.reduce_mean(tf.abs(g_f)).numpy()) if g_f is not None else 0.0
        )
        norms_base[name] = (
            float(tf.reduce_mean(tf.abs(g_b)).numpy()) if g_b is not None else 0.0
        )
    del tape
    return {"final": norms_final, "base": norms_base}


def grad_norms_wrt_inputs(
    probe: tf.keras.Model, batch: list[np.ndarray]
) -> dict[str, float]:
    """Mean |∂ŷ/∂input| norms for temporal / fourier / holiday / lag."""
    xt = _as_tensors(batch)
    keys = ("temporal", "fourier", "holiday", "lag", "sku")
    with tf.GradientTape() as tape:
        for x in xt[:-1]:  # skip int sku for input grads
            tape.watch(x)
        o = probe(xt, training=False)
        y = tf.reduce_sum(o["final_forecast"])
    grads = tape.gradient(y, xt[:-1])
    out = {}
    for key, g in zip(keys[:-1], grads):
        if g is None:
            out[key] = 0.0
        else:
            # Per-sample L2 over feature dim, then mean
            flat = tf.reshape(g, [tf.shape(g)[0], -1])
            out[key] = float(tf.reduce_mean(tf.norm(flat, axis=-1)).numpy())
    return out


def finite_diff_sensitivity(
    probe: tf.keras.Model,
    batch: list[np.ndarray],
    eps: float = EPS_FD,
) -> dict[str, float]:
    """Finite-difference Δŷ from perturbing each input group."""
    base = probe(_as_tensors(batch), training=False)
    y0 = np.asarray(base["final_forecast"], dtype=np.float64).ravel()

    def _delta(group_idx: int, scale: float = 1.0) -> float:
        pert = [x.copy() for x in batch]
        pert[group_idx] = pert[group_idx] + np.float32(eps * scale)
        y1 = np.asarray(
            probe(_as_tensors(pert), training=False)["final_forecast"],
            dtype=np.float64,
        ).ravel()
        return float(np.mean(np.abs(y1 - y0)) / abs(eps * scale))

    # Holiday: also try a signed day-shift of +1 day on distance features
    holiday_day = [x.copy() for x in batch]
    holiday_day[2] = holiday_day[2] + np.float32(1.0)
    y_h = np.asarray(
        probe(_as_tensors(holiday_day), training=False)["final_forecast"],
        dtype=np.float64,
    ).ravel()

    # Seasonal phase shift: rotate year angle slightly via day bump on fourier
    day = (batch[0].ravel() * 365.25 + 1.0).astype(np.float32)
    seasonal_shift = [x.copy() for x in batch]
    seasonal_shift[1] = _fourier_from_day(day)
    y_s = np.asarray(
        probe(_as_tensors(seasonal_shift), training=False)["final_forecast"],
        dtype=np.float64,
    ).ravel()

    return {
        "d_y_d_time (FD / eps)": _delta(0),
        "d_y_d_fourier (FD / eps)": _delta(1),
        "d_y_d_holiday (FD / eps)": _delta(2),
        "d_y_d_lag (FD / eps)": _delta(3),
        "mean_|Δŷ| holiday +1 day": float(np.mean(np.abs(y_h - y0))),
        "mean_|Δŷ| seasonal +1 day phase": float(np.mean(np.abs(y_s - y0))),
    }


def attention_and_expert_stats(
    probe: tf.keras.Model, batch: list[np.ndarray]
) -> dict:
    o = probe(_as_tensors(batch), training=False)
    attn = np.asarray(o["attn"], dtype=np.float64)
    stats = {
        "attn_mean": {n: float(attn[:, i].mean()) for i, n in enumerate(NAMES)},
        "attn_std": {n: float(attn[:, i].std()) for i, n in enumerate(NAMES)},
        "expert_mean": {
            n: float(np.asarray(o[n]).mean()) for n in NAMES
        },
        "expert_std": {
            n: float(np.asarray(o[n]).std()) for n in NAMES
        },
        "gate_mean": float(np.asarray(o["non_zero_probability"]).mean()),
        "final_mean": float(np.asarray(o["final_forecast"]).mean()),
        "base_mean": float(np.asarray(o["base_forecast"]).mean()),
    }
    return stats


def compare_near_far(probe: tf.keras.Model) -> dict:
    near = make_batch(holiday_mode="near", seed=SEED + 1)
    far = make_batch(holiday_mode="far", seed=SEED + 1)
    active = make_batch(holiday_mode="hinge_active", seed=SEED + 1)
    # Keep temporal/fourier/lag/sku identical; only holiday distances differ
    for other in (far, active):
        other[0], other[1], other[3], other[4] = (
            near[0],
            near[1],
            near[3],
            near[4],
        )

    g_near = grad_norms_wrt_inputs(probe, near)
    g_far = grad_norms_wrt_inputs(probe, far)
    g_act = grad_norms_wrt_inputs(probe, active)
    o_near = probe(_as_tensors(near), training=False)
    o_far = probe(_as_tensors(far), training=False)
    o_act = probe(_as_tensors(active), training=False)
    h_near = np.asarray(o_near["holiday"]).ravel()
    h_far = np.asarray(o_far["holiday"]).ravel()
    h_act = np.asarray(o_act["holiday"]).ravel()
    y_near = np.asarray(o_near["final_forecast"]).ravel()
    y_far = np.asarray(o_far["final_forecast"]).ravel()
    y_act = np.asarray(o_act["final_forecast"]).ravel()
    return {
        "holiday_input_grad_near_0_7d": g_near["holiday"],
        "holiday_input_grad_hinge_active_80_200d": g_act["holiday"],
        "holiday_input_grad_far_120_250d": g_far["holiday"],
        "holiday_expert_|mean|_near": float(np.mean(np.abs(h_near))),
        "holiday_expert_|mean|_active": float(np.mean(np.abs(h_act))),
        "holiday_expert_|mean|_far": float(np.mean(np.abs(h_far))),
        "mean_|Δŷ| near_vs_far": float(np.mean(np.abs(y_near - y_far))),
        "mean_|Δŷ| near_vs_active": float(np.mean(np.abs(y_near - y_act))),
        "attn_holiday_near": float(np.asarray(o_near["attn"])[:, 2].mean()),
        "attn_holiday_far": float(np.asarray(o_far["attn"])[:, 2].mean()),
        "note": (
            "At init, holiday ChangepointReLU spreads hinges over [0,365]; "
            "first hinge ≈ 73d → distances 0–7d often have dead ReLU grads "
            "until training relocates cps."
        ),
    }


def holiday_distance_sweep(probe: tf.keras.Model) -> list[tuple[float, float, float]]:
    """Sweep a shared holiday distance; report |∂ŷ/∂h| and |Δŷ vs d=0|."""
    base = make_batch(holiday_mode="near", seed=SEED + 2)
    # Fix all holiday channels to the same distance d
    rows = []
    y0 = None
    for d in (0.0, 3.0, 14.0, 40.0, 80.0, 120.0, 200.0, 300.0):
        b = [x.copy() for x in base]
        b[2][:] = np.float32(d)
        g = grad_norms_wrt_inputs(probe, b)["holiday"]
        y = np.asarray(
            probe(_as_tensors(b), training=False)["final_forecast"], dtype=np.float64
        ).ravel()
        if y0 is None:
            y0 = y
        rows.append((d, g, float(np.mean(np.abs(y - y0)))))
    return rows


def gate_attenuation(probe: tf.keras.Model, batch: list[np.ndarray]) -> dict:
    """Show how intermittent gate scales expert influence into final ŷ."""
    o = probe(_as_tensors(batch), training=False)
    gate = np.asarray(o["non_zero_probability"], dtype=np.float64).ravel()
    base = np.asarray(o["base_forecast"], dtype=np.float64).ravel()
    final = np.asarray(o["final_forecast"], dtype=np.float64).ravel()
    # Split by low vs high gate
    lo = gate < np.median(gate)
    hi = ~lo
    return {
        "gate_median": float(np.median(gate)),
        "mean_|final|/|base| low-gate": float(
            np.mean(np.abs(final[lo]) / (np.abs(base[lo]) + 1e-8))
        ),
        "mean_|final|/|base| high-gate": float(
            np.mean(np.abs(final[hi]) / (np.abs(base[hi]) + 1e-8))
        ),
        "corr(gate, |final|)": float(
            np.corrcoef(gate, np.abs(final))[0, 1]
        ),
    }


def _fmt_dict(d: dict, indent: int = 2) -> str:
    pad = " " * indent
    lines = []
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"{pad}{k}:")
            lines.append(_fmt_dict(v, indent + 2))
        elif isinstance(v, float):
            lines.append(f"{pad}{k}: {v:.6g}")
        else:
            lines.append(f"{pad}{k}: {v}")
    return "\n".join(lines)


def main() -> None:
    _set_seed(SEED)
    print("=" * 72)
    print("DeepSequence component gradient diagnostics")
    print("(random-init structural sensitivity — NOT trained importance)")
    print("=" * 72)

    model = build_model(use_intermittent=True, trend_monotonic=True)
    probe = build_probe(model)
    batch = make_batch(holiday_mode="mixed", seed=SEED)

    # Confirm monotone trend path
    trend_layer = model.get_layer("trend")
    print(
        f"\nTrend path: trend_monotonic={getattr(trend_layer, 'trend_monotonic', '?')} "
        f"raw_sign={getattr(trend_layer, 'raw_sign', None)}"
    )
    print(f"Params: {model.count_params():,}")

    stats = attention_and_expert_stats(probe, batch)
    print("\n--- A) Component attention weights (mean ± std) ---")
    for n in NAMES:
        print(
            f"  {n:10s}  attn={stats['attn_mean'][n]:.4f}±{stats['attn_std'][n]:.4f}  "
            f"expert_out μ={stats['expert_mean'][n]:+.4f} σ={stats['expert_std'][n]:.4f}"
        )
    print(
        f"  gate μ={stats['gate_mean']:.4f}  "
        f"base μ={stats['base_mean']:.4f}  final μ={stats['final_mean']:.4f}"
    )

    g_exp = grad_norms_wrt_experts(probe, batch)
    print("\n--- B) Mean |∂ŷ/∂expert_output| ---")
    print("  (base_forecast before/at base head; final after intermittent gate)")
    print(f"  {'expert':10s}  {'|∂base/∂c|':>12s}  {'|∂final/∂c|':>12s}")
    for n in NAMES:
        print(
            f"  {n:10s}  {g_exp['base'][n]:12.6g}  {g_exp['final'][n]:12.6g}"
        )

    g_in = grad_norms_wrt_inputs(probe, batch)
    print("\n--- C) Mean per-sample ||∂ŷ/∂input_group||_2 ---")
    for k, v in g_in.items():
        print(f"  {k:10s}  {v:.6g}")

    fd = finite_diff_sensitivity(probe, batch)
    print("\n--- D) Finite-difference / local sensitivity ---")
    for k, v in fd.items():
        print(f"  {k}: {v:.6g}")

    nf = compare_near_far(probe)
    print("\n--- E) Near-holiday vs far-from-holiday (matched covariates) ---")
    for k, v in nf.items():
        if isinstance(v, str):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v:.6g}")

    sweep = holiday_distance_sweep(probe)
    print("\n--- E2) Holiday distance sweep (all channels = d) ---")
    print(f"  {'d_days':>8s}  {'||∂ŷ/∂h||':>12s}  {'mean_|Δŷ vs d=0|':>18s}")
    for d, g, dy in sweep:
        print(f"  {d:8.1f}  {g:12.6g}  {dy:18.6g}")

    gate = gate_attenuation(probe, batch)
    print("\n--- F) Intermittent gate attenuation ---")
    for k, v in gate.items():
        print(f"  {k}: {v:.6g}")

    # Optional: no-gate ablation for structural comparison
    _set_seed(SEED)
    model_ng = build_model(use_intermittent=False, trend_monotonic=True)
    probe_ng = build_probe(model_ng)
    g_ng = grad_norms_wrt_experts(probe_ng, batch)
    print("\n--- G) Same batch without intermittent gate (|∂final/∂c|) ---")
    for n in NAMES:
        print(f"  {n:10s}  {g_ng['final'][n]:.6g}")

    print("\n" + "=" * 72)
    print("INTERPRETATION (structural, random init)")
    print("=" * 72)
    print(
        """
  Trend:    monotone PWL of normalized time (softplus slopes × ReLU hinges).
            ∂ŷ/∂time dominates input grads → long-run level / calendar progress;
            SKU shift-scale personalizes slope/intercept.

  Seasonal: Fourier / cyclical features → masked attention → scalar.
            Sensitivity to fourier / +1-day phase shift = calendar periodicity.
            High |∂ŷ/∂seasonal_expert| means the mixer listens when seasonal fires.

  Holiday:  per-holiday distance → changepoint hinges → hierarchical attn.
            Attention mass is present (~1/4) even at init (SKU-conditioned mixer).
            Input sensitivity is localized in distance space and, at init, often
            silent for d ≪ first hinge (~73d) until cps move in training.

  Gate:     final ≈ base × P(nonzero). Low gate days shrink all expert
            influence into the published forecast even if base experts fire.
"""
    )


if __name__ == "__main__":
    main()
