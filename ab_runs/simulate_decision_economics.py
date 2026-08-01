#!/usr/bin/env python3
"""Decision-economics simulation with replenishment lead times.

Framing
-------
TST / DS are different *risk profiles*, not a universal IWMAE winner:
  - TST: lower-overstock posture (better when holding cost hurts more; low r)
  - DS:  lower-stockout posture (better when lost sales hurt more; high r)

    r = C_lost / C_hold
    cost(r) = r * U + H

C_hold is a single policy input (not varied by margin regime). Margin regimes
only change C_lost = margin * unit_price, so r rises with margin under fixed
carry. Hybrid models are out of scope for this selector.

Lead time maps to the forecast horizon used for the replenishment decision
(daily panel)::

  LT = 1 day   → horizon h=1
  LT = 1 week  → horizon h=7
  LT = 2 weeks → horizon h=14

Usage (from repo root)::

    python ab_runs/simulate_decision_economics.py
    python ab_runs/simulate_decision_economics.py --include-h1-reclaim
    python ab_runs/simulate_decision_economics.py --holding-cost 0.10 --margins 0.08,0.25,0.55
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MH_SOURCE = "ab_runs/recompare_retuned/daily_mh14_locked_all.json"

# Lead time → daily horizon used for the order that arrives then.
LEAD_TIMES = (
    {
        "key": "lt_1_day",
        "label": "Lead time 1 day",
        "lead_time_days": 1,
        "horizon": "1",
    },
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
)

MH_MODEL_NAMES = (
    ("TST lite", "temporal_transformer", "lower-overstock"),
    ("plain DS", "deepsequence", "lower-stockout"),
    ("TFT lite", "tft_lite", "sequence"),
    ("DeepAR lite", "deepar_lite", "sequence"),
)

H1_RECLAIM_MODELS = (
    {
        "name": "TST lite",
        "model_key": "temporal_transformer",
        "source": "ab_runs/reclaim/daily_h1_hybrid_temporal.json",
        "profile": "lower-overstock",
    },
    {
        "name": "plain DS",
        "model_key": "deepsequence",
        "source": "ab_runs/reclaim/daily_h1_hybrid_temporal.json",
        "profile": "lower-stockout",
    },
)


def _load_inv():
    path = ROOT / "deepsequence_hierarchical_attention" / "inventory_metrics.py"
    spec = importlib.util.spec_from_file_location("ds_inv_econ", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["ds_inv_econ"] = mod
    spec.loader.exec_module(mod)
    return mod


def _compact_regimes(margin_regimes: dict) -> dict:
    out = {}
    for k, v in margin_regimes.items():
        out[k] = {
            "label": v.get("label"),
            "margin": v.get("margin"),
            "holding_cost_per_unit": v.get("holding_cost_per_unit"),
            "cost_ratio_r": v["cost_ratio_r"],
            "winner": v["winner"],
            "profit_loss_by_model": {
                n: {
                    "total_profit_loss_per_day": pl["total_profit_loss_per_day"],
                    "contribution_loss_per_day": pl["contribution_loss_per_day"],
                    "holding_loss_per_day": pl["holding_loss_per_day"],
                }
                for n, pl in v["profit_loss_by_model"].items()
            },
        }
    return out


def _plot_single(ax, report: dict, title: str) -> None:
    styles = {
        "TST lite": ("#1f77b4", "-"),
        "plain DS": ("#d62728", "-"),
        "TFT lite": ("#ff7f0e", "--"),
        "DeepAR lite": ("#9467bd", ":"),
    }
    for name, curve in report["curves"].items():
        color, ls = styles.get(name, ("#555555", "-"))
        ax.plot(curve["r"], curve["cost"], color=color, ls=ls, lw=2.0, label=name)

    r_star = report["crossover"].get("r_star")
    if r_star is not None:
        ax.axvline(
            r_star,
            color="#444444",
            ls=":",
            lw=1.4,
            label=f"r*={r_star:.2f}",
        )

    for key, regime in report["margin_regimes"].items():
        r = regime["cost_ratio_r"]
        ax.axvline(r, color="#bbbbbb", ls="--", lw=0.7, alpha=0.8)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel(r"$r = C_{\mathrm{lost}}/C_{\mathrm{hold}}$")
    ax.set_ylabel(r"$r\cdot U + H$")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=7, frameon=False)


def _plot_lead_times(by_lt: dict, out_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skip plot")
        return

    keys = [lt["key"] for lt in LEAD_TIMES if lt["key"] in by_lt]
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, lt in zip(axes, LEAD_TIMES):
        if lt["key"] not in by_lt:
            continue
        block = by_lt[lt["key"]]
        _plot_single(
            ax,
            block["report"],
            f"{lt['label']} (h={lt['horizon']})\n"
            f"r*={block['report']['crossover'].get('r_star')}",
        )
    fig.suptitle(
        "Decision economics by replenishment lead time\n"
        "(same policy; horizon = lead time on daily panel)",
        fontsize=11,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"Wrote {out_png}")


def _print_lead_time(lt: dict, report: dict) -> None:
    print(
        f"\n======== {lt['label']}  (horizon h={lt['horizon']}) ========"
    )
    print("Risk profiles (U=lost-sales/day, H=holding/day):")
    for name, c in report["components"].items():
        print(f"  {name:28s} U={c['U']:.4f}  H={c['H']:.4f}")
    r_star = report["crossover"].get("r_star")
    if r_star is not None:
        print(f"Crossover r* (TST vs DS): {r_star:.3f}")
        print("  r < r* → TST (hold-sensitive);  r > r* → DS (lost-sales-sensitive)")
    else:
        print("Crossover r*: none (no sign change / parallel profiles)")
    for key, regime in report["margin_regimes"].items():
        rows = sorted(
            regime["profit_loss_by_model"].items(),
            key=lambda kv: kv[1]["total_profit_loss_per_day"],
        )
        best = rows[0]
        print(
            f"  [{key}] r={regime['cost_ratio_r']:.2f}  winner={regime['winner']}  "
            f"best_loss={best[1]['total_profit_loss_per_day']:.4f}"
        )
        for name, pl in rows[:3]:
            print(
                f"      {name:26s} total={pl['total_profit_loss_per_day']:.4f}  "
                f"contrib={pl['contribution_loss_per_day']:.4f}  "
                f"hold={pl['holding_loss_per_day']:.4f}"
            )


def _report_from_model_inv(inv, model_inv: dict, *, margin_regimes: dict) -> dict:
    return inv.decision_economics_report(
        model_inv, pair=("TST lite", "plain DS"), margin_regimes=margin_regimes
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mh-json",
        default=MH_SOURCE,
        help="Locked daily multi-horizon bake-off JSON (has h=1,7,14).",
    )
    ap.add_argument(
        "--out-json",
        default="ab_runs/reclaim/daily_decision_economics_by_lead_time.json",
    )
    ap.add_argument(
        "--out-png",
        default="paper_figures/fig_decision_economics_by_lead_time.png",
    )
    ap.add_argument(
        "--holding-cost",
        type=float,
        default=None,
        help=(
            "Inventory holding cost per unit from policy (C_hold). "
            "Default: package DEFAULT_POLICY_HOLDING_COST_PER_UNIT. "
            "Margins only change C_lost; C_hold stays fixed."
        ),
    )
    ap.add_argument(
        "--margins",
        default="0.08,0.25,0.55",
        help="Comma-separated low,mid,high contribution margins (unit price=1).",
    )
    ap.add_argument(
        "--include-h1-reclaim",
        action="store_true",
        help="Also attach H=1 reclaim panel (TST vs DS) as reference.",
    )
    args = ap.parse_args()
    inv = _load_inv()

    holding = (
        float(args.holding_cost)
        if args.holding_cost is not None
        else float(inv.DEFAULT_POLICY_HOLDING_COST_PER_UNIT)
    )
    margins = tuple(float(x) for x in args.margins.split(","))
    regimes = inv.margin_regimes_from_policy(
        holding_cost_per_unit=holding, margins=margins
    )

    mh = json.loads((ROOT / args.mh_json).read_text(encoding="utf-8"))
    by_lead_time = {}

    for lt in LEAD_TIMES:
        model_inv = {}
        meta = {}
        for name, key, profile in MH_MODEL_NAMES:
            if key not in mh.get("models", {}):
                continue
            scored_mh = inv.rescore_multihorizon_model_payload(mh["models"][key])
            block = (scored_mh.get("by_horizon") or {}).get(lt["horizon"]) or {}
            if block.get("inventory_mean_under") is None:
                continue
            model_inv[name] = block
            meta[name] = {
                "model_key": key,
                "profile": profile,
                "source": args.mh_json,
                "horizon": lt["horizon"],
            }
        if "TST lite" not in model_inv or "plain DS" not in model_inv:
            print(f"Skip {lt['key']}: missing TST/DS at h={lt['horizon']}")
            continue
        report = inv.decision_economics_report(
            model_inv, pair=("TST lite", "plain DS"), margin_regimes=regimes
        )
        by_lead_time[lt["key"]] = {
            "lead_time_days": lt["lead_time_days"],
            "label": lt["label"],
            "horizon": lt["horizon"],
            "policy_note": (
                "Replenishment ordered today arrives after the lead time; "
                f"evaluate forecast errors at horizon h={lt['horizon']}."
            ),
            "models": meta,
            "components": report["components"],
            "crossover": report["crossover"],
            "margin_regimes": _compact_regimes(report["margin_regimes"]),
            "curves": report["curves"],
            "report": report,  # full for plotting; stripped below for JSON size
        }
        _print_lead_time(lt, report)

    # Portfolio selector summary across lead times
    selector = {
        "rule": (
            "Pick model by business r=C_lost/C_hold vs lead-time-specific r*. "
            "High-margin / high-penalty → DS when r>r*; "
            "low-margin / high-carry → TST when r<r*."
        ),
        "by_lead_time": {},
    }
    for lt in LEAD_TIMES:
        block = by_lead_time.get(lt["key"])
        if not block:
            continue
        selector["by_lead_time"][lt["key"]] = {
            "r_star": block["crossover"].get("r_star"),
            "low_margin_winner": block["margin_regimes"]["low_margin"]["winner"],
            "mid_margin_winner": block["margin_regimes"]["mid_margin"]["winner"],
            "high_margin_winner": block["margin_regimes"]["high_margin"]["winner"],
        }

    payload = {
        "framing": (
            "Decision economics by replenishment lead time. "
            "Models are risk profiles; lead time selects the horizon. "
            "C_hold is fixed from policy; margin regimes only change C_lost "
            "(and thus r). Hybrid is not in the selector."
        ),
        "source": args.mh_json,
        "sku_list": "ab_runs/recompare/sku_list_daily_data42.json",
        "policy": {
            "holding_cost_per_unit": holding,
            "margins": list(margins),
            "unit_price": float(inv.DEFAULT_UNIT_PRICE),
            "note": (
                "C_hold from inventory/finance policy (shared across regimes). "
                "Only margin varies → r = margin*price / C_hold."
            ),
            "regimes": {
                k: {
                    "margin": v["margin"],
                    "holding_cost_per_unit": v["holding_cost_per_unit"],
                    "cost_ratio_r": inv.cost_ratio_from_margin(
                        v["margin"] * float(inv.DEFAULT_UNIT_PRICE), holding
                    ),
                }
                for k, v in regimes.items()
            },
        },
        "lead_times": [
            {
                "key": lt["key"],
                "label": lt["label"],
                "lead_time_days": lt["lead_time_days"],
                "horizon": lt["horizon"],
            }
            for lt in LEAD_TIMES
        ],
        "by_lead_time": {
            k: {kk: vv for kk, vv in v.items() if kk != "report"}
            for k, v in by_lead_time.items()
        },
        "portfolio_selector": selector,
    }

    if args.include_h1_reclaim:
        model_inv = {}
        for spec in H1_RECLAIM_MODELS:
            overall = json.loads(
                (ROOT / spec["source"]).read_text(encoding="utf-8")
            )["models"][spec["model_key"]]["overall"]
            scored = inv.inventory_cost_from_kpi_summary(overall)
            scored["iwmae_rounded"] = overall.get("iwmae_rounded")
            model_inv[spec["name"]] = scored
        h1_report = _report_from_model_inv(inv, model_inv, margin_regimes=regimes)
        payload["h1_reclaim_reference"] = {
            "note": (
                "Separate H=1 reclaim panel (TST vs plain DS only); not MH-locked."
            ),
            "crossover": h1_report["crossover"],
            "components": h1_report["components"],
            "margin_regimes": _compact_regimes(h1_report["margin_regimes"]),
        }

    out_json = ROOT / args.out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_json}")

    print("\n=== Selector snapshot ===")
    for lt_key, row in selector["by_lead_time"].items():
        print(
            f"  {lt_key}: r*={row['r_star']:.3f}  "
            f"low→{row['low_margin_winner']}  "
            f"mid→{row['mid_margin_winner']}  "
            f"high→{row['high_margin_winner']}"
        )

    _plot_lead_times(by_lead_time, ROOT / args.out_png)


if __name__ == "__main__":
    main()
