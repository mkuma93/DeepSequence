#!/usr/bin/env python3
"""All-model decision economics for lead times 7 / 14 / 21 / 28 days.

π proxy (higher better)::

    C_lost = margin * unit_price + C_loyalty
    inv_loss = C_lost * U + C_hold * H
    π = revenue_proxy − inv_loss − C_model

``C_loyalty`` is a scenario switch / loyalty leakage penalty per unit underage
(not estimated). U/H come from rescoring the MH bake-off JSON; C_model is an
explicit ops proxy (train wall-time normalized, with documented fixed-tier
backup).

Hybrid models are out of scope. C_model is intentionally coarse — not a
cloud bill.

Usage (from repo root)::

    python ab_runs/simulate_decision_economics_mh_all.py \\
      --mh-json ab_runs/reclaim/daily_mh_7_14_21_28_all_models.json \\
      --c-model-mode tier \\
      --loyalty-costs 0,0.25,0.5 \\
      --out-json ab_runs/reclaim/daily_decision_economics_mh_7_14_21_28_loyalty.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MH_SOURCE = "ab_runs/reclaim/daily_mh_7_14_21_28_all_models.json"

# Lead time → forecast horizon for the replenishment that arrives then.
# Default = daily weekly ladder; override with --lead-times.
LEAD_TIMES = (
    {
        "key": "lt_1_week",
        "label": "Lead time 1 week",
        "lead_time_days": 7,
        "horizon": "7",
    },
    {
        "key": "lt_2_weeks",
        "label": "Lead time 2 weeks",
        "lead_time_days": 14,
        "horizon": "14",
    },
    {
        "key": "lt_3_weeks",
        "label": "Lead time 3 weeks",
        "lead_time_days": 21,
        "horizon": "21",
    },
    {
        "key": "lt_4_weeks",
        "label": "Lead time 4 weeks",
        "lead_time_days": 28,
        "horizon": "28",
    },
)

# Display name, payload key, coarse risk/profile tag.
MH_MODEL_NAMES = (
    ("LightGBM", "lightgbm", "tabular"),
    ("plain DS", "deepsequence", "lower-stockout"),
    ("TST lite", "temporal_transformer", "lower-overstock"),
    ("TFT lite", "tft_lite", "sequence"),
    ("DeepAR lite", "deepar_lite", "sequence"),
)

# Optional classical intermittent (monthly car-parts); omitted unless present.
CLASSICAL_MODEL_NAMES = (
    ("Croston", "croston", "classical"),
    ("SBA", "sba", "classical"),
    ("TSB", "tsb", "classical"),
)

_MODEL_KEY_TO_SPEC = {
    key: (name, key, profile)
    for name, key, profile in (*MH_MODEL_NAMES, *CLASSICAL_MODEL_NAMES)
}


def _parse_lead_times(raw: str | None) -> list[dict]:
    """Parse ``7,14,28`` or ``7:7,14:14`` or ``1:1:lt_1_month`` specs.

    Forms per entry:
      - ``H``                 → lead_time=H, horizon=H
      - ``lead:horizon``      → explicit lead / horizon
      - ``lead:horizon:key``  → also set selector key (label derived)
    """
    if raw is None or not str(raw).strip():
        return list(LEAD_TIMES)
    out: list[dict] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        bits = part.split(":")
        if len(bits) == 1:
            h = int(bits[0])
            lead, horizon, key = h, str(h), f"lt_{h}"
        elif len(bits) == 2:
            lead, horizon = int(bits[0]), str(int(bits[1]))
            key = f"lt_{lead}"
        elif len(bits) == 3:
            lead, horizon, key = int(bits[0]), str(int(bits[1])), bits[2]
        else:
            raise SystemExit(f"bad --lead-times entry: {part!r}")
        # Friendly labels for common daily / monthly analogues.
        label_map = {
            1: "Lead time 1 period",
            2: "Lead time 2 periods",
            6: "Lead time 6 periods",
            7: "Lead time 1 week",
            14: "Lead time 2 weeks",
            21: "Lead time 3 weeks",
            28: "Lead time 4 weeks",
            60: "Lead time ~2 months (60d)",
        }
        label = label_map.get(lead, f"Lead time {lead}")
        if key.startswith("lt_") and "month" in key:
            label = key.replace("lt_", "Lead time ").replace("_", " ")
        out.append(
            {
                "key": key,
                "label": label,
                "lead_time_days": lead,
                "horizon": horizon,
            }
        )
    if not out:
        raise SystemExit("--lead-times produced an empty list")
    return out


def _resolve_model_specs(raw: str | None, models_payload: dict) -> list[tuple]:
    """Resolve which models to score (display, key, profile)."""
    if raw is None or not str(raw).strip():
        # Prefer canonical daily set; append classical only when present.
        specs = [s for s in MH_MODEL_NAMES if s[1] in models_payload]
        for s in CLASSICAL_MODEL_NAMES:
            if s[1] in models_payload:
                specs.append(s)
        return specs
    specs = []
    for key in str(raw).split(","):
        key = key.strip()
        if not key:
            continue
        if key not in models_payload:
            print(f"Skip model {key}: not in MH JSON")
            continue
        if key in _MODEL_KEY_TO_SPEC:
            specs.append(_MODEL_KEY_TO_SPEC[key])
        else:
            specs.append((key, key, "other"))
    return specs

LOYALTY_SCENARIO_NOTES = {
    0.0: "legacy: one-period contribution only (no switch / loyalty leakage)",
    0.25: "modest: one-time switch penalty ≈ 0.25× unit price per underage unit",
    0.5: "high: strong repeat / brand risk ≈ 0.5× unit price per underage unit",
    1.0: "very high: stockout ≈ full unit-price loyalty hit on top of margin",
}


def _load_inv():
    path = ROOT / "deepsequence_hierarchical_attention" / "inventory_metrics.py"
    spec = importlib.util.spec_from_file_location("ds_inv_econ_mh_all", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["ds_inv_econ_mh_all"] = mod
    spec.loader.exec_module(mod)
    return mod


def _resolve_c_model(
    inv,
    *,
    display_name: str,
    model_key: str,
    train_seconds: float | None,
    train_cost_map: dict[str, float],
    mode: str,
    base_per_day: float,
) -> dict:
    tier_cost = inv.model_ops_cost_from_tiers(
        display_name, base_per_day=base_per_day
    )
    # Also try payload key for tiers.
    if display_name not in inv.DEFAULT_MODEL_OPS_TIER:
        tier_cost = inv.model_ops_cost_from_tiers(
            model_key, base_per_day=base_per_day
        )
    time_cost = train_cost_map.get(display_name)
    if time_cost is None:
        time_cost = train_cost_map.get(model_key)
    if mode == "tier":
        chosen, source = tier_cost, "fixed_tier"
    elif mode == "train_time" and time_cost is not None:
        chosen, source = time_cost, "train_seconds_normalized"
    elif time_cost is not None:
        chosen, source = time_cost, "train_seconds_normalized"
    else:
        chosen, source = tier_cost, "fixed_tier_fallback"
    return {
        "C_model_per_day": float(chosen),
        "source": source,
        "train_seconds": None if train_seconds is None else float(train_seconds),
        "tier_C_model_per_day": float(tier_cost),
        "train_time_C_model_per_day": (
            None if time_cost is None else float(time_cost)
        ),
    }


def _score_lead_time(
    inv,
    *,
    lt: dict,
    model_inv: dict,
    meta: dict,
    c_model_info: dict,
    mean_demand: float | None,
    regimes: dict,
    loyalty: float,
    mh_json: str,
) -> dict:
    # Pair for crossover: prefer TST vs DS when both present.
    pair = ("TST lite", "plain DS")
    if pair[0] not in model_inv or pair[1] not in model_inv:
        names = list(model_inv.keys())
        pair = (names[0], names[1])

    report = inv.decision_economics_report(
        model_inv, pair=pair, margin_regimes=regimes
    )

    pi_regimes = {}
    for rkey, regime in report["margin_regimes"].items():
        rows = []
        for name, inv_block in model_inv.items():
            cm = c_model_info[name]["C_model_per_day"]
            pi = inv.profit_with_model_ops(
                inv_block,
                margin=regime["margin"],
                holding_cost_per_unit=regime["holding_cost_per_unit"],
                model_ops_cost_per_day=cm,
                unit_price=regime.get("unit_price", 1.0),
                loyalty_cost_per_unit=loyalty,
                mean_demand_per_day=mean_demand,
            )
            rows.append({"model": name, **pi})
        rows.sort(key=lambda r: -r["pi_per_day"])
        pi_regimes[rkey] = {
            "label": regime.get("label"),
            "margin": regime["margin"],
            "holding_cost_per_unit": regime["holding_cost_per_unit"],
            "loyalty_cost_per_unit": loyalty,
            "C_lost_per_unit": regime.get("C_lost_per_unit"),
            "cost_ratio_r": regime["cost_ratio_r"],
            "inv_only_winner": regime["winner"],
            "pi_winner": rows[0]["model"] if rows else None,
            "ranking": rows,
        }

    iwmae_rows = []
    for name, block in model_inv.items():
        comp = report["components"].get(name) or {}
        iwmae_rows.append(
            {
                "model": name,
                "iwmae_rounded": block.get("iwmae_rounded"),
                "U": comp.get("U"),
                "H": comp.get("H"),
                "bias": block.get("bias"),
                "C_model_per_day": c_model_info[name]["C_model_per_day"],
                "C_model_source": c_model_info[name]["source"],
            }
        )
    iwmae_rows.sort(
        key=lambda r: (
            r["iwmae_rounded"] is None,
            r["iwmae_rounded"] if r["iwmae_rounded"] is not None else 1e9,
        )
    )

    return {
        "lead_time_days": lt["lead_time_days"],
        "label": lt["label"],
        "horizon": lt["horizon"],
        "loyalty_cost_per_unit": loyalty,
        "policy_note": (
            "Replenishment ordered today arrives after the lead time; "
            f"evaluate forecast errors at horizon h={lt['horizon']}. "
            f"C_loyalty={loyalty} (scenario analysis)."
        ),
        "mean_demand_per_day": mean_demand,
        "models": meta,
        "c_model": c_model_info,
        "components": report["components"],
        "crossover": report["crossover"],
        "inv_only_margin_regimes": {
            k: {
                "winner": v["winner"],
                "cost_ratio_r": v["cost_ratio_r"],
                "C_lost_per_unit": v.get("C_lost_per_unit"),
                "loyalty_cost_per_unit": loyalty,
                "profit_loss_by_model": {
                    n: {
                        "total_profit_loss_per_day": pl[
                            "total_profit_loss_per_day"
                        ],
                        "contribution_loss_per_day": pl[
                            "contribution_loss_per_day"
                        ],
                        "loyalty_loss_per_day": pl["loyalty_loss_per_day"],
                        "holding_loss_per_day": pl["holding_loss_per_day"],
                    }
                    for n, pl in v["profit_loss_by_model"].items()
                },
            }
            for k, v in report["margin_regimes"].items()
        },
        "pi_margin_regimes": pi_regimes,
        "iwmae_ranking": iwmae_rows,
        "curves": report["curves"],
        "source": mh_json,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mh-json", default=MH_SOURCE)
    ap.add_argument(
        "--out-json",
        default="ab_runs/reclaim/daily_decision_economics_mh_7_14_21_28_loyalty.json",
    )
    ap.add_argument("--holding-cost", type=float, default=None)
    ap.add_argument("--margins", default="0.08,0.25,0.55")
    ap.add_argument(
        "--loyalty-cost",
        type=float,
        default=None,
        help="Single loyalty / switch cost per underage unit (overrides list).",
    )
    ap.add_argument(
        "--loyalty-costs",
        default=None,
        help=(
            "Comma-separated loyalty scenarios (default: package "
            "DEFAULT_LOYALTY_SCENARIOS = 0,0.25,0.5). Scenario analysis only."
        ),
    )
    ap.add_argument(
        "--c-model-mode",
        choices=("auto", "train_time", "tier"),
        default="auto",
        help=(
            "auto: prefer train_seconds normalized when present, else fixed "
            "tiers. train_time / tier force one method."
        ),
    )
    ap.add_argument(
        "--c-model-base",
        type=float,
        default=None,
        help="Per-day ops base for tier-1 / fastest model (default package).",
    )
    ap.add_argument(
        "--include-h1",
        action="store_true",
        help="Also score free h=1 from the same MH JSON if present.",
    )
    ap.add_argument(
        "--lead-times",
        default=None,
        help=(
            "Comma-separated lead times / horizons. Forms: H | lead:horizon | "
            "lead:horizon:key. Default: daily 7/14/21/28. Examples: "
            "7,14,28,60 or 1:1:lt_1_month,2:2:lt_2_months,6:6:lt_6_months."
        ),
    )
    ap.add_argument(
        "--models",
        default=None,
        help=(
            "Comma-separated MH JSON model keys to score (default: LightGBM/"
            "DS/TST/TFT/DeepAR plus classical Croston/SBA/TSB when present)."
        ),
    )
    args = ap.parse_args()
    inv = _load_inv()

    holding = (
        float(args.holding_cost)
        if args.holding_cost is not None
        else float(inv.DEFAULT_POLICY_HOLDING_COST_PER_UNIT)
    )
    margins = tuple(float(x) for x in args.margins.split(","))
    unit_price = float(inv.DEFAULT_UNIT_PRICE)
    if args.loyalty_cost is not None:
        loyalty_costs = (float(args.loyalty_cost),)
    elif args.loyalty_costs is not None:
        loyalty_costs = tuple(
            float(x) for x in args.loyalty_costs.split(",") if x.strip()
        )
    else:
        loyalty_costs = tuple(float(x) for x in inv.DEFAULT_LOYALTY_SCENARIOS)
    if not loyalty_costs:
        raise SystemExit("need at least one loyalty cost scenario")

    base_per_day = (
        float(args.c_model_base)
        if args.c_model_base is not None
        else float(inv.DEFAULT_MODEL_OPS_BASE_PER_DAY)
    )

    mh_path = ROOT / args.mh_json
    mh = json.loads(mh_path.read_text(encoding="utf-8"))
    models_payload = mh.get("models") or {}
    model_specs = _resolve_model_specs(args.models, models_payload)
    if len(model_specs) < 2:
        raise SystemExit(
            f"need ≥2 models in MH JSON after --models filter; "
            f"found {[k for _, k, _ in model_specs]}"
        )

    train_seconds = {}
    for name, key, _ in model_specs:
        block = models_payload.get(key) or {}
        ts = block.get("train_seconds")
        if ts is not None:
            train_seconds[name] = float(ts)
            train_seconds[key] = float(ts)
    train_cost_map = inv.model_ops_cost_from_train_seconds(
        {
            k: v
            for k, v in train_seconds.items()
            if k in {n for n, _, _ in model_specs}
        },
        base_per_day=base_per_day,
    )
    # Also index by payload key for lookups.
    for name, key, _ in model_specs:
        if name in train_cost_map:
            train_cost_map[key] = train_cost_map[name]

    lead_specs = _parse_lead_times(args.lead_times)
    if args.include_h1 and not any(lt["horizon"] == "1" for lt in lead_specs):
        lead_specs = [
            {
                "key": "lt_1_day",
                "label": "Lead time 1 day (free h=1)",
                "lead_time_days": 1,
                "horizon": "1",
            },
            *lead_specs,
        ]

    # Rescore U/H once per lead time (loyalty only changes C_lost, not U/H).
    lead_model_cache = {}
    iwmae_table = {}
    for lt in lead_specs:
        model_inv = {}
        meta = {}
        c_model_info = {}
        mean_demand = None
        for name, key, profile in model_specs:
            if key not in models_payload:
                continue
            scored_mh = inv.rescore_multihorizon_model_payload(models_payload[key])
            block = (scored_mh.get("by_horizon") or {}).get(lt["horizon"]) or {}
            if block.get("inventory_mean_under") is None:
                continue
            raw_block = (models_payload[key].get("by_horizon") or {}).get(
                lt["horizon"], {}
            )
            raw = (
                raw_block.get("overall")
                if isinstance(raw_block, dict) and "overall" in raw_block
                else raw_block
            ) or {}
            if raw.get("mean_actual") is not None:
                block = {**block, "mean_actual": raw["mean_actual"]}
                if mean_demand is None:
                    mean_demand = float(raw["mean_actual"])
            if raw.get("iwmae_rounded") is not None:
                block = {**block, "iwmae_rounded": raw["iwmae_rounded"]}
            elif raw.get("iwmae") is not None and block.get("iwmae_rounded") is None:
                block = {**block, "iwmae_rounded": raw["iwmae"]}
            if raw.get("bias") is not None:
                block = {**block, "bias": raw["bias"]}
            model_inv[name] = block
            meta[name] = {
                "model_key": key,
                "profile": profile,
                "source": args.mh_json,
                "horizon": lt["horizon"],
                "n_params": models_payload[key].get("n_params"),
            }
            c_model_info[name] = _resolve_c_model(
                inv,
                display_name=name,
                model_key=key,
                train_seconds=(models_payload[key] or {}).get("train_seconds"),
                train_cost_map=train_cost_map,
                mode=args.c_model_mode,
                base_per_day=base_per_day,
            )

        if len(model_inv) < 2:
            print(f"Skip {lt['key']}: need ≥2 models at h={lt['horizon']}")
            continue

        lead_model_cache[lt["key"]] = {
            "lt": lt,
            "model_inv": model_inv,
            "meta": meta,
            "c_model_info": c_model_info,
            "mean_demand": mean_demand,
        }
        # IWMAE / U / H shared across loyalty scenarios.
        iwmae_rows = []
        for name, block in model_inv.items():
            comp = inv.decision_cost_components(block)
            iwmae_rows.append(
                {
                    "model": name,
                    "iwmae_rounded": block.get("iwmae_rounded"),
                    "U": comp.get("U"),
                    "H": comp.get("H"),
                    "bias": block.get("bias"),
                    "C_model_per_day": c_model_info[name]["C_model_per_day"],
                    "C_model_source": c_model_info[name]["source"],
                }
            )
        iwmae_rows.sort(
            key=lambda r: (
                r["iwmae_rounded"] is None,
                r["iwmae_rounded"] if r["iwmae_rounded"] is not None else 1e9,
            )
        )
        iwmae_table[lt["key"]] = iwmae_rows

    by_loyalty = {}
    selector_by_loyalty = {}

    for loyalty in loyalty_costs:
        loy_key = inv.loyalty_tag(loyalty)
        regimes = inv.margin_regimes_from_policy(
            holding_cost_per_unit=holding,
            margins=margins,
            unit_price=unit_price,
            loyalty_cost_per_unit=loyalty,
        )
        by_lead_time = {}
        for lt_key, cache in lead_model_cache.items():
            block = _score_lead_time(
                inv,
                lt=cache["lt"],
                model_inv=cache["model_inv"],
                meta=cache["meta"],
                c_model_info=cache["c_model_info"],
                mean_demand=cache["mean_demand"],
                regimes=regimes,
                loyalty=loyalty,
                mh_json=args.mh_json,
            )
            by_lead_time[lt_key] = block

            print(
                f"\n======== {cache['lt']['label']}  "
                f"(h={cache['lt']['horizon']}, C_loyalty={loyalty}) ========"
            )
            print("IWMAE / U / H / C_model:")
            for row in block["iwmae_ranking"]:
                print(
                    f"  {row['model']:28s} iwmae={row['iwmae_rounded']}  "
                    f"U={row['U']:.4f}  H={row['H']:.4f}  "
                    f"C_model={row['C_model_per_day']:.4f} ({row['C_model_source']})"
                )
            for rkey, preg in block["pi_margin_regimes"].items():
                print(
                    f"  [{rkey}] r_eff={preg['cost_ratio_r']:.2f}  "
                    f"C_lost={preg['C_lost_per_unit']:.3f}  "
                    f"inv_winner={preg['inv_only_winner']}  "
                    f"π_winner={preg['pi_winner']}"
                )
                for row in preg["ranking"][:5]:
                    print(
                        f"      {row['model']:26s} π={row['pi_per_day']:.4f}  "
                        f"inv_loss={row['total_profit_loss_per_day']:.4f}  "
                        f"loy={row['loyalty_loss_per_day']:.4f}  "
                        f"C_model={row['model_ops_cost_per_day']:.4f}"
                    )

        selector = {"by_lead_time": {}}
        for lt in lead_specs:
            block = by_lead_time.get(lt["key"])
            if not block:
                continue
            selector["by_lead_time"][lt["key"]] = {
                "r_star": (block.get("crossover") or {}).get("r_star"),
                "loyalty_cost_per_unit": loyalty,
                "low_margin_pi_winner": block["pi_margin_regimes"]["low_margin"][
                    "pi_winner"
                ],
                "mid_margin_pi_winner": block["pi_margin_regimes"]["mid_margin"][
                    "pi_winner"
                ],
                "high_margin_pi_winner": block["pi_margin_regimes"]["high_margin"][
                    "pi_winner"
                ],
                "low_margin_inv_winner": block["inv_only_margin_regimes"][
                    "low_margin"
                ]["winner"],
                "mid_margin_inv_winner": block["inv_only_margin_regimes"][
                    "mid_margin"
                ]["winner"],
                "high_margin_inv_winner": block["inv_only_margin_regimes"][
                    "high_margin"
                ]["winner"],
            }

        note = LOYALTY_SCENARIO_NOTES.get(
            float(loyalty),
            f"custom C_loyalty={loyalty} (scenario analysis, not estimated)",
        )
        by_loyalty[loy_key] = {
            "loyalty_cost_per_unit": loyalty,
            "scenario_note": note,
            "policy": {
                "holding_cost_per_unit": holding,
                "margins": list(margins),
                "unit_price": unit_price,
                "loyalty_cost_per_unit": loyalty,
                "regimes": {
                    k: {
                        "margin": v["margin"],
                        "holding_cost_per_unit": v["holding_cost_per_unit"],
                        "loyalty_cost_per_unit": loyalty,
                        "C_lost_per_unit": v["C_lost_per_unit"],
                        "cost_ratio_r": inv.cost_ratio_from_margin(
                            v["margin"],
                            holding,
                            unit_price=unit_price,
                            loyalty_cost_per_unit=loyalty,
                        ),
                    }
                    for k, v in regimes.items()
                },
            },
            "by_lead_time": by_lead_time,
            "portfolio_selector": selector,
        }
        selector_by_loyalty[loy_key] = selector

    # Compact winner matrix: lead × margin × loyalty → π winner.
    winner_matrix = []
    for loyalty in loyalty_costs:
        loy_key = inv.loyalty_tag(loyalty)
        sel = selector_by_loyalty[loy_key]["by_lead_time"]
        for lt_key, row in sel.items():
            winner_matrix.append(
                {
                    "lead_time": lt_key,
                    "loyalty_cost_per_unit": loyalty,
                    "low_margin_pi_winner": row["low_margin_pi_winner"],
                    "mid_margin_pi_winner": row["mid_margin_pi_winner"],
                    "high_margin_pi_winner": row["high_margin_pi_winner"],
                }
            )

    payload = {
        "framing": (
            "All-model decision economics by replenishment lead time "
            "with loyalty / switching-cost scenarios. "
            "π = revenue_proxy − inv_loss − C_model where "
            "inv_loss = (margin*price + C_loyalty)*U + C_hold*H. "
            "C_hold fixed from policy; margin and C_loyalty change C_lost "
            "(and r_eff). Hybrid out of selector. C_model / C_loyalty are "
            "transparent proxies / scenarios — not estimated cloud bills or "
            "CLV fits."
        ),
        "caveats": [
            "C_loyalty is SCENARIO ANALYSIS only — not fit from churn data.",
            "C_model is NOT a measured production / cloud bill.",
            "Primary run uses --c-model-mode as requested; fixed tiers "
            f"(LGBM=1, DS=1.5, TST=1.5, TFT=2, DeepAR=2; classical default 1.5) × "
            f"base_per_day={base_per_day}.",
            "Revenue proxy uses shared mean_actual at the horizon when present; "
            "ranking equals minimizing inv_loss + C_model when demand is shared.",
            "U/H are forecast-error proxies (underage / no-sale holding), not "
            "a full inventory simulation with pipeline stock.",
            "Monthly lead-time analogues use forecast horizons 1/2/6 as "
            "replenishment lag proxies (not calendar-day lead times).",
        ],
        "source": args.mh_json,
        "sku_list": (mh.get("config") or {}).get(
            "sku_list", "ab_runs/recompare/sku_list_daily_data42.json"
        ),
        "ds_stack": (mh.get("config") or {}).get("ds_stack"),
        "policy": {
            "holding_cost_per_unit": holding,
            "margins": list(margins),
            "unit_price": unit_price,
            "loyalty_costs": list(loyalty_costs),
            "c_model_mode": args.c_model_mode,
            "c_model_base_per_day": base_per_day,
            "c_model_tiers": dict(inv.DEFAULT_MODEL_OPS_TIER),
            "note": (
                "C_hold from inventory/finance policy (shared). "
                "C_lost = margin*price + C_loyalty; r_eff = C_lost/C_hold. "
                "C_loyalty scenarios are labelled assumptions. "
                "C_model is an explicit ops proxy."
            ),
            "recommended_default_loyalty": {
                "loyalty_cost_per_unit": 0.25,
                "rationale": (
                    "Modest one-time switch penalty: enough that high-U "
                    "under-forecasters cannot win low-margin rankings solely "
                    "by cheap holding, without assuming extreme CLV loss. "
                    "Always report loyalty_0 (legacy) alongside."
                ),
            },
        },
        "lead_times": [
            {
                "key": lt["key"],
                "label": lt["label"],
                "lead_time_days": lt["lead_time_days"],
                "horizon": lt["horizon"],
            }
            for lt in lead_specs
            if lt["key"] in lead_model_cache
        ],
        "iwmae_by_lead_time": iwmae_table,
        "by_loyalty": by_loyalty,
        "portfolio_selector_by_loyalty": selector_by_loyalty,
        "pi_winner_matrix": winner_matrix,
    }

    # Flat convenience alias for single-scenario runs (legacy shape).
    if len(loyalty_costs) == 1:
        only = by_loyalty[inv.loyalty_tag(loyalty_costs[0])]
        payload["by_lead_time"] = only["by_lead_time"]
        payload["portfolio_selector"] = only["portfolio_selector"]

    out_json = ROOT / args.out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_json}")

    print("\n=== π winner matrix (lead × loyalty × margin) ===")
    for row in winner_matrix:
        print(
            f"  {row['lead_time']:12s}  C_loy={row['loyalty_cost_per_unit']:<4}  "
            f"low→{row['low_margin_pi_winner']}  "
            f"mid→{row['mid_margin_pi_winner']}  "
            f"high→{row['high_margin_pi_winner']}"
        )


if __name__ == "__main__":
    main()
