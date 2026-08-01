#!/usr/bin/env python3
"""Multi-seed long-horizon + loyalty evaluation (daily + car-parts).

Panel convention
----------------
SKU list is **locked** from seed-42 lists:
  - ab_runs/recompare/sku_list_daily_data42.json
  - ab_runs/recompare/sku_list_carparts_data42.json
``--data_seed 42`` + locked sku_list; ``--seed`` / ``--train_seed`` vary
so variance is optimization / init (TF + LightGBM), not panel resampling.

Stack (fixed): softsign + mono + Level-1 holiday/regressor attention,
mixer on, FiLM off, use_cross_layers=False.

Usage (repo root)::

  TF_USE_LEGACY_KERAS=1 .venv-test/bin/python \\
    ab_runs/reclaim/multiseed/run_multiseed_long_loyalty.py --phase all
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "ab_runs" / "reclaim" / "multiseed"
SEEDS = (42, 43, 44, 45, 46)

DAILY_SKU = "ab_runs/recompare/sku_list_daily_data42.json"
CARPARTS_SKU = "ab_runs/recompare/sku_list_carparts_data42.json"
DAILY_DATA = (
    "/Users/mritunjaykumar/Library/CloudStorage/"
    "GoogleDrive-mritunjay.kmr1@gmail.com/My Drive/jubilant/data"
)
# Reuse seed-42 bake-offs already run with the same stack/panel when present.
DAILY_S42_SOURCE = ROOT / "ab_runs/reclaim/daily_mh_1_60_level1_cross_off_all_models.json"
CARPARTS_S42_SOURCE = ROOT / "ab_runs/reclaim/carparts_mh_1_2_6_level1_cross_off.json"

DAILY_MODELS = "deepsequence,temporal_transformer,lightgbm"
CARPARTS_MODELS = "deepsequence,tsb,lightgbm"

DS_STACK = {
    "output_activation": "softsign",
    "trend_monotonic": True,
    "holiday_monotonic": True,
    "regressor_monotonic": True,
    "context_aware_component_mixer": True,
    "context_film_seasonal_holiday": False,
    "use_cross_layers": False,
}

PY = str(ROOT / ".venv-test" / "bin" / "python")


def _run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TF_USE_LEGACY_KERAS"] = "1"
    env.setdefault("DEEPSEQUENCE_DATA_DIR", DAILY_DATA)
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    print(f"    log → {log_path}", flush=True)
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"CMD: {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log.write(f"\nEXIT:{proc.returncode} elapsed_s={time.time() - t0:.1f}\n")
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): see {log_path}")
    print(f"    ok in {time.time() - t0:.1f}s", flush=True)


def _mean_std(xs: list[float]) -> dict:
    if not xs:
        return {"mean": None, "std": None, "n": 0, "values": []}
    if len(xs) == 1:
        return {"mean": xs[0], "std": 0.0, "n": 1, "values": xs}
    return {
        "mean": float(statistics.mean(xs)),
        "std": float(statistics.stdev(xs)),  # sample std
        "n": len(xs),
        "values": xs,
    }


def _fmt_ms(ms: dict) -> str:
    if ms["mean"] is None:
        return "n/a"
    return f"{ms['mean']:.4f} ± {ms['std']:.4f}"


def _iwmae_from_mh(mh: dict, model_key: str, horizon: str) -> float | None:
    block = (mh.get("models") or {}).get(model_key) or {}
    by_h = block.get("by_horizon") or {}
    cell = by_h.get(str(horizon)) or {}
    # Daily MH nests volume bands under overall; car-parts stores metrics flat.
    overall = cell.get("overall") if isinstance(cell.get("overall"), dict) else cell
    if isinstance(overall, dict):
        if "iwmae_rounded" in overall:
            return float(overall["iwmae_rounded"])
        if "iwmae" in overall:
            return float(overall["iwmae"])
    # Fallback: daily comparison is dict[horizon]→list; car-parts is a flat list.
    comp = mh.get("comparison")
    rows = []
    if isinstance(comp, dict):
        rows = comp.get(str(horizon)) or []
    elif isinstance(comp, list):
        rows = comp
    for row in rows:
        if row.get("model") != model_key:
            continue
        # Car-parts mean comparison may expose h{H}_iwmae.
        for key in (
            "iwmae_rounded",
            "iwmae",
            f"h{horizon}_iwmae",
            f"h{horizon}_iwmae_rounded",
        ):
            if key in row and row[key] is not None:
                return float(row[key])
    return None


def _pi_from_loyalty(
    loyalty: dict, loy_key: str, lt_key: str, margin_key: str, model_name: str
) -> float | None:
    by_loy = loyalty.get("by_loyalty") or {}
    block = by_loy.get(loy_key) or {}
    lt = (block.get("by_lead_time") or {}).get(lt_key) or {}
    regimes = lt.get("pi_margin_regimes") or {}
    ranking = (regimes.get(margin_key) or {}).get("ranking") or []
    for row in ranking:
        if row.get("model") == model_name:
            return float(row["pi_per_day"])
    return None


def _winner(
    loyalty: dict, loy_key: str, lt_key: str, margin_key: str, kind: str = "pi"
) -> str | None:
    by_loy = loyalty.get("by_loyalty") or {}
    block = by_loy.get(loy_key) or {}
    # Prefer portfolio_selector nested under by_loyalty
    sel = (block.get("portfolio_selector") or {}).get("by_lead_time") or {}
    row = sel.get(lt_key) or {}
    field = f"{margin_key}_{kind}_winner"
    if row.get(field):
        return row[field]
    # Fallback: pi_margin_regimes
    lt = (block.get("by_lead_time") or {}).get(lt_key) or {}
    regimes = lt.get("pi_margin_regimes") or {}
    return (regimes.get(margin_key) or {}).get(f"{kind}_winner")


def run_daily_seed(seed: int, *, force: bool = False) -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mh_path = OUT_DIR / f"daily_s{seed}_mh60.json"
    loy_path = OUT_DIR / f"daily_s{seed}_loyalty.json"
    log_path = OUT_DIR / f"daily_s{seed}_mh60.log"

    if mh_path.exists() and not force:
        print(f"skip daily MH seed={seed} (exists)", flush=True)
    elif seed == 42 and DAILY_S42_SOURCE.exists() and not force:
        print(f"reuse seed-42 daily MH from {DAILY_S42_SOURCE.name}", flush=True)
        shutil.copy2(DAILY_S42_SOURCE, mh_path)
    else:
        cmd = [
            PY,
            "-m",
            "deepsequence_hierarchical_attention.eval.multihorizon_compare",
            "--data_dir",
            DAILY_DATA,
            "--horizon",
            "60",
            "--report_horizons",
            "1,7,14,28,60",
            "--models",
            DAILY_MODELS,
            "--sku_list",
            DAILY_SKU,
            "--data_seed",
            "42",
            "--seed",
            str(seed),
            "--train_seed",
            str(seed),
            "--use_cross_layers",
            "0",
            "--out_json",
            str(mh_path),
        ]
        _run(cmd, log_path)

    if loy_path.exists() and not force:
        print(f"skip daily loyalty seed={seed} (exists)", flush=True)
    else:
        cmd = [
            PY,
            "ab_runs/simulate_decision_economics_mh_all.py",
            "--mh-json",
            str(mh_path.relative_to(ROOT)),
            "--out-json",
            str(loy_path.relative_to(ROOT)),
            "--holding-cost",
            "0.10",
            "--margins",
            "0.08,0.25,0.55",
            "--loyalty-costs",
            "0,0.25",
            "--c-model-mode",
            "tier",
            "--lead-times",
            "7,14,28,60",
            "--include-h1",
            "--models",
            DAILY_MODELS,
        ]
        _run(cmd, OUT_DIR / f"daily_s{seed}_loyalty.log")
    return mh_path, loy_path


def run_carparts_seed(seed: int, *, force: bool = False) -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mh_path = OUT_DIR / f"carparts_s{seed}_mh6.json"
    loy_path = OUT_DIR / f"carparts_s{seed}_loyalty.json"
    log_path = OUT_DIR / f"carparts_s{seed}_mh6.log"

    if mh_path.exists() and not force:
        print(f"skip carparts MH seed={seed} (exists)", flush=True)
    elif seed == 42 and CARPARTS_S42_SOURCE.exists() and not force:
        print(f"reuse seed-42 carparts MH from {CARPARTS_S42_SOURCE.name}", flush=True)
        shutil.copy2(CARPARTS_S42_SOURCE, mh_path)
    else:
        cmd = [
            PY,
            "-m",
            "deepsequence_hierarchical_attention.eval.public_carparts_mh_all",
            "--horizon",
            "6",
            "--models",
            CARPARTS_MODELS,
            "--sku_list",
            CARPARTS_SKU,
            "--data_seed",
            "42",
            "--seed",
            str(seed),
            "--train_seed",
            str(seed),
            "--use_cross_layers",
            "0",
            "--out_json",
            str(mh_path),
        ]
        _run(cmd, log_path)

    if loy_path.exists() and not force:
        print(f"skip carparts loyalty seed={seed} (exists)", flush=True)
    else:
        cmd = [
            PY,
            "ab_runs/simulate_decision_economics_mh_all.py",
            "--mh-json",
            str(mh_path.relative_to(ROOT)),
            "--out-json",
            str(loy_path.relative_to(ROOT)),
            "--holding-cost",
            "0.10",
            "--margins",
            "0.08,0.25,0.55",
            "--loyalty-costs",
            "0,0.25",
            "--c-model-mode",
            "tier",
            "--lead-times",
            "1:1:lt_1_month,2:2:lt_2_months,6:6:lt_6_months",
            "--models",
            CARPARTS_MODELS,
        ]
        _run(cmd, OUT_DIR / f"carparts_s{seed}_loyalty.log")
    return mh_path, loy_path


def aggregate_daily(seeds: tuple[int, ...]) -> dict:
    model_keys = ["deepsequence", "temporal_transformer", "lightgbm"]
    model_names = {
        "deepsequence": "plain DS",
        "temporal_transformer": "TST lite",
        "lightgbm": "LightGBM",
    }
    horizons = ["1", "7", "14", "28", "60"]
    lt_map = {"1": "lt_1", "7": "lt_7", "14": "lt_14", "28": "lt_28", "60": "lt_60"}

    iwmae: dict[str, dict[str, list[float]]] = {h: {m: [] for m in model_keys} for h in horizons}
    pi_mid_loy025: dict[str, dict[str, list[float]]] = {
        h: {m: [] for m in model_keys} for h in ("7", "14", "28", "60")
    }
    wins_ds_vs_tst = {h: 0 for h in ("28", "60")}
    wins_ds_vs_tst_n = {h: 0 for h in ("28", "60")}
    lgbm_low_win = {
        "loyalty_0": {h: 0 for h in ("7", "14", "28", "60")},
        "loyalty_0p25": {h: 0 for h in ("7", "14", "28", "60")},
    }
    lgbm_low_n = {
        "loyalty_0": {h: 0 for h in ("7", "14", "28", "60")},
        "loyalty_0p25": {h: 0 for h in ("7", "14", "28", "60")},
    }
    mid_pi_winner_loy025 = {h: [] for h in ("7", "14", "28", "60")}
    per_seed = []

    for seed in seeds:
        mh = json.loads((OUT_DIR / f"daily_s{seed}_mh60.json").read_text())
        loy = json.loads((OUT_DIR / f"daily_s{seed}_loyalty.json").read_text())
        seed_row = {"seed": seed, "iwmae": {}, "pi_mid_loyalty_0p25": {}, "winners": {}}
        for h in horizons:
            seed_row["iwmae"][h] = {}
            for m in model_keys:
                v = _iwmae_from_mh(mh, m, h)
                if v is not None:
                    iwmae[h][m].append(v)
                    seed_row["iwmae"][h][m] = v
        for h in ("7", "14", "28", "60"):
            lt = lt_map[h]
            seed_row["pi_mid_loyalty_0p25"][h] = {}
            for m in model_keys:
                pv = _pi_from_loyalty(
                    loy, "loyalty_0p25", lt, "mid_margin", model_names[m]
                )
                if pv is not None:
                    pi_mid_loy025[h][m].append(pv)
                    seed_row["pi_mid_loyalty_0p25"][h][m] = pv
            w_mid = _winner(loy, "loyalty_0p25", lt, "mid_margin", "pi")
            mid_pi_winner_loy025[h].append(w_mid)
            seed_row["winners"][h] = {
                "mid_pi_loyalty_0p25": w_mid,
                "low_pi_loyalty_0": _winner(loy, "loyalty_0", lt, "low_margin", "pi"),
                "low_pi_loyalty_0p25": _winner(
                    loy, "loyalty_0p25", lt, "low_margin", "pi"
                ),
            }
            # DS vs TST by mid-margin π at long horizons
            if h in ("28", "60"):
                ds = seed_row["pi_mid_loyalty_0p25"][h].get("deepsequence")
                tst = seed_row["pi_mid_loyalty_0p25"][h].get("temporal_transformer")
                if ds is not None and tst is not None:
                    wins_ds_vs_tst_n[h] += 1
                    if ds > tst:
                        wins_ds_vs_tst[h] += 1
            for loy_key in ("loyalty_0", "loyalty_0p25"):
                w = _winner(loy, loy_key, lt, "low_margin", "pi")
                if w is not None:
                    lgbm_low_n[loy_key][h] += 1
                    if w == "LightGBM":
                        lgbm_low_win[loy_key][h] += 1
        per_seed.append(seed_row)

    summary = {
        "framing": (
            "Multi-seed long-horizon + loyalty (daily). Panel locked to "
            "sku_list_daily_data42.json; seeds vary train/init only "
            "(--data_seed 42 --sku_list locked; --seed/--train_seed vary)."
        ),
        "seeds": list(seeds),
        "panel_convention": {
            "sku_list": DAILY_SKU,
            "data_seed": 42,
            "train_seed": "varies with --seed",
            "note": (
                "Same SKU panel across seeds; variance is optimization/init "
                "(TF set_random_seed(train_seed) + LightGBM random_state=seed)."
            ),
        },
        "ds_stack": DS_STACK,
        "models": list(model_keys),
        "policy": {
            "C_hold": 0.10,
            "margins": [0.08, 0.25, 0.55],
            "C_loyalty": [0.0, 0.25],
            "c_model_mode": "tier",
        },
        "iwmae_mean_std": {
            h: {m: _mean_std(iwmae[h][m]) for m in model_keys} for h in horizons
        },
        "pi_mid_margin_loyalty_0p25_mean_std": {
            h: {m: _mean_std(pi_mid_loy025[h][m]) for m in model_keys}
            for h in ("7", "14", "28", "60")
        },
        "win_rates": {
            "ds_beats_tst_mid_pi_loyalty_0p25": {
                h: {
                    "wins": wins_ds_vs_tst[h],
                    "n": wins_ds_vs_tst_n[h],
                    "rate": (
                        wins_ds_vs_tst[h] / wins_ds_vs_tst_n[h]
                        if wins_ds_vs_tst_n[h]
                        else None
                    ),
                }
                for h in ("28", "60")
            },
            "lgbm_low_margin_pi_winner": {
                loy_key: {
                    h: {
                        "wins": lgbm_low_win[loy_key][h],
                        "n": lgbm_low_n[loy_key][h],
                        "rate": (
                            lgbm_low_win[loy_key][h] / lgbm_low_n[loy_key][h]
                            if lgbm_low_n[loy_key][h]
                            else None
                        ),
                    }
                    for h in ("7", "14", "28", "60")
                }
                for loy_key in ("loyalty_0", "loyalty_0p25")
            },
            "mid_pi_winner_loyalty_0p25_by_seed": mid_pi_winner_loy025,
        },
        "per_seed": per_seed,
        "tables_pretty": {
            "iwmae": {
                h: {m: _fmt_ms(_mean_std(iwmae[h][m])) for m in model_keys}
                for h in horizons
            },
            "pi_mid_loyalty_0p25": {
                h: {m: _fmt_ms(_mean_std(pi_mid_loy025[h][m])) for m in model_keys}
                for h in ("7", "14", "28", "60")
            },
        },
    }

    # Stability verdict
    long_ds_better_iwmae = all(
        (summary["iwmae_mean_std"][h]["deepsequence"]["mean"] or math.inf)
        < (summary["iwmae_mean_std"][h]["temporal_transformer"]["mean"] or -math.inf)
        for h in ("28", "60")
    )
    ds_win_long = all(
        (summary["win_rates"]["ds_beats_tst_mid_pi_loyalty_0p25"][h]["rate"] or 0) >= 0.6
        for h in ("28", "60")
    )
    lgbm_drop = (
        (summary["win_rates"]["lgbm_low_margin_pi_winner"]["loyalty_0"]["60"]["rate"] or 0)
        > (summary["win_rates"]["lgbm_low_margin_pi_winner"]["loyalty_0p25"]["60"]["rate"] or 1)
    )
    summary["stability_verdict"] = {
        "long_horizon_ds_better_iwmae_mean_28_60": long_ds_better_iwmae,
        "ds_beats_tst_mid_pi_loy025_rate_ge_0p6_at_28_60": ds_win_long,
        "loyalty_reduces_lgbm_low_margin_winrate_at_h60": lgbm_drop,
        "stable_across_seeds": bool(long_ds_better_iwmae and ds_win_long),
        "note": (
            "Stable if DS mean IWMAE beats TST at h=28/60 and DS mid-π "
            "(C_loyalty=0.25) beats TST in ≥60% of seeds at those horizons."
        ),
    }
    return summary


def aggregate_carparts(seeds: tuple[int, ...]) -> dict:
    model_keys = ["deepsequence", "tsb", "lightgbm"]
    model_names = {
        "deepsequence": "plain DS",
        "tsb": "TSB",
        "lightgbm": "LightGBM",
    }
    horizons = ["1", "2", "6"]
    lt_map = {"1": "lt_1_month", "2": "lt_2_months", "6": "lt_6_months"}

    iwmae: dict[str, dict[str, list[float]]] = {h: {m: [] for m in model_keys} for h in horizons}
    pi_mid_loy025: dict[str, dict[str, list[float]]] = {
        h: {m: [] for m in model_keys} for h in horizons
    }
    wins_ds_vs_tsb = {h: 0 for h in ("6",)}
    wins_ds_vs_tsb_n = {h: 0 for h in ("6",)}
    lgbm_low_win = {
        "loyalty_0": {h: 0 for h in horizons},
        "loyalty_0p25": {h: 0 for h in horizons},
    }
    lgbm_low_n = {
        "loyalty_0": {h: 0 for h in horizons},
        "loyalty_0p25": {h: 0 for h in horizons},
    }
    mid_pi_winner_loy025 = {h: [] for h in horizons}
    per_seed = []

    for seed in seeds:
        mh = json.loads((OUT_DIR / f"carparts_s{seed}_mh6.json").read_text())
        loy = json.loads((OUT_DIR / f"carparts_s{seed}_loyalty.json").read_text())
        seed_row = {"seed": seed, "iwmae": {}, "pi_mid_loyalty_0p25": {}, "winners": {}}
        for h in horizons:
            seed_row["iwmae"][h] = {}
            for m in model_keys:
                v = _iwmae_from_mh(mh, m, h)
                if v is not None:
                    iwmae[h][m].append(v)
                    seed_row["iwmae"][h][m] = v
            lt = lt_map[h]
            seed_row["pi_mid_loyalty_0p25"][h] = {}
            for m in model_keys:
                pv = _pi_from_loyalty(
                    loy, "loyalty_0p25", lt, "mid_margin", model_names[m]
                )
                if pv is not None:
                    pi_mid_loy025[h][m].append(pv)
                    seed_row["pi_mid_loyalty_0p25"][h][m] = pv
            w_mid = _winner(loy, "loyalty_0p25", lt, "mid_margin", "pi")
            mid_pi_winner_loy025[h].append(w_mid)
            seed_row["winners"][h] = {
                "mid_pi_loyalty_0p25": w_mid,
                "low_pi_loyalty_0": _winner(loy, "loyalty_0", lt, "low_margin", "pi"),
                "low_pi_loyalty_0p25": _winner(
                    loy, "loyalty_0p25", lt, "low_margin", "pi"
                ),
            }
            if h == "6":
                ds = seed_row["pi_mid_loyalty_0p25"][h].get("deepsequence")
                tsb = seed_row["pi_mid_loyalty_0p25"][h].get("tsb")
                if ds is not None and tsb is not None:
                    wins_ds_vs_tsb_n[h] += 1
                    if ds > tsb:
                        wins_ds_vs_tsb[h] += 1
            for loy_key in ("loyalty_0", "loyalty_0p25"):
                w = _winner(loy, loy_key, lt, "low_margin", "pi")
                if w is not None:
                    lgbm_low_n[loy_key][h] += 1
                    if w == "LightGBM":
                        lgbm_low_win[loy_key][h] += 1
        per_seed.append(seed_row)

    summary = {
        "framing": (
            "Multi-seed long-horizon + loyalty (car-parts monthly). Panel locked "
            "to sku_list_carparts_data42.json; seeds vary train/init only."
        ),
        "seeds": list(seeds),
        "panel_convention": {
            "sku_list": CARPARTS_SKU,
            "data_seed": 42,
            "train_seed": "varies with --seed",
            "note": (
                "Same SKU panel across seeds; TSB is classical (seed-invariant); "
                "DS/LGBM vary with train seed."
            ),
        },
        "ds_stack": DS_STACK,
        "models": list(model_keys),
        "policy": {
            "C_hold": 0.10,
            "margins": [0.08, 0.25, 0.55],
            "C_loyalty": [0.0, 0.25],
            "c_model_mode": "tier",
        },
        "iwmae_mean_std": {
            h: {m: _mean_std(iwmae[h][m]) for m in model_keys} for h in horizons
        },
        "pi_mid_margin_loyalty_0p25_mean_std": {
            h: {m: _mean_std(pi_mid_loy025[h][m]) for m in model_keys} for h in horizons
        },
        "win_rates": {
            "ds_beats_tsb_mid_pi_loyalty_0p25": {
                "6": {
                    "wins": wins_ds_vs_tsb["6"],
                    "n": wins_ds_vs_tsb_n["6"],
                    "rate": (
                        wins_ds_vs_tsb["6"] / wins_ds_vs_tsb_n["6"]
                        if wins_ds_vs_tsb_n["6"]
                        else None
                    ),
                }
            },
            "lgbm_low_margin_pi_winner": {
                loy_key: {
                    h: {
                        "wins": lgbm_low_win[loy_key][h],
                        "n": lgbm_low_n[loy_key][h],
                        "rate": (
                            lgbm_low_win[loy_key][h] / lgbm_low_n[loy_key][h]
                            if lgbm_low_n[loy_key][h]
                            else None
                        ),
                    }
                    for h in horizons
                }
                for loy_key in ("loyalty_0", "loyalty_0p25")
            },
            "mid_pi_winner_loyalty_0p25_by_seed": mid_pi_winner_loy025,
        },
        "per_seed": per_seed,
        "tables_pretty": {
            "iwmae": {
                h: {m: _fmt_ms(_mean_std(iwmae[h][m])) for m in model_keys}
                for h in horizons
            },
            "pi_mid_loyalty_0p25": {
                h: {m: _fmt_ms(_mean_std(pi_mid_loy025[h][m])) for m in model_keys}
                for h in horizons
            },
        },
    }
    ds_better_iwmae_h6 = (
        (summary["iwmae_mean_std"]["6"]["deepsequence"]["mean"] or math.inf)
        < (summary["iwmae_mean_std"]["6"]["tsb"]["mean"] or -math.inf)
    )
    ds_win_h6 = (summary["win_rates"]["ds_beats_tsb_mid_pi_loyalty_0p25"]["6"]["rate"] or 0) >= 0.6
    lgbm_drop = (
        (summary["win_rates"]["lgbm_low_margin_pi_winner"]["loyalty_0"]["6"]["rate"] or 0)
        > (
            summary["win_rates"]["lgbm_low_margin_pi_winner"]["loyalty_0p25"]["6"]["rate"]
            or 1
        )
    )
    summary["stability_verdict"] = {
        "h6_ds_better_iwmae_mean_vs_tsb": ds_better_iwmae_h6,
        "ds_beats_tsb_mid_pi_loy025_rate_ge_0p6_at_h6": ds_win_h6,
        "loyalty_reduces_lgbm_low_margin_winrate_at_h6": lgbm_drop,
        "stable_across_seeds": bool(ds_better_iwmae_h6 and ds_win_h6),
        "note": (
            "Stable if DS mean IWMAE beats TSB at h=6 and DS mid-π "
            "(C_loyalty=0.25) beats TSB in ≥60% of seeds."
        ),
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--phase",
        choices=("daily", "carparts", "aggregate", "all"),
        default="all",
    )
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--force", action="store_true", help="Re-run even if outputs exist.")
    args = ap.parse_args()
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.phase in ("daily", "all"):
        for seed in seeds:
            run_daily_seed(seed, force=args.force)
        daily = aggregate_daily(seeds)
        out = OUT_DIR / "daily_multiseed_long_loyalty_summary.json"
        out.write_text(json.dumps(daily, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out}", flush=True)

    if args.phase in ("carparts", "all"):
        for seed in seeds:
            run_carparts_seed(seed, force=args.force)
        car = aggregate_carparts(seeds)
        out = OUT_DIR / "carparts_multiseed_long_loyalty_summary.json"
        out.write_text(json.dumps(car, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out}", flush=True)

    if args.phase == "aggregate":
        daily = aggregate_daily(seeds)
        (OUT_DIR / "daily_multiseed_long_loyalty_summary.json").write_text(
            json.dumps(daily, indent=2) + "\n", encoding="utf-8"
        )
        car = aggregate_carparts(seeds)
        (OUT_DIR / "carparts_multiseed_long_loyalty_summary.json").write_text(
            json.dumps(car, indent=2) + "\n", encoding="utf-8"
        )
        print("Aggregated daily + carparts summaries.", flush=True)


if __name__ == "__main__":
    main()
