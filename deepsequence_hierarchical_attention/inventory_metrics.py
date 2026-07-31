"""Inventory holding + sales-revenue planning metrics for intermittent demand.

Operational story:
  - **No-sale days:** no revenue to lose, but stock that sits has an
    **inventory holding cost** (``inventory_holding_cost_zero``). Forecast-driven
    restock that is not sold is the same quantity (``unnecessary_restock_zero``).
  - **Sale days:** short stock → **sales / revenue loss**
    (``sales_revenue_loss_units``, unit price = 1).

Combined ops cost (per calendar day, holding weight ``h`` relative to unit revenue)::

    combined = sales_revenue_loss_per_day + h * inventory_holding_cost_zero_per_day

Bake-offs still report IWMAE. Prefer:
  - ``PRIMARY_SALES_REVENUE_LOSS_METRIC`` for lost-sales only
  - ``PRIMARY_COMBINED_OPS_METRIC`` (``combined_ops_cost_h0p1``) when both
    revenue loss and holding matter with mild holding (h=0.1).

Newsvendor ``cu/co`` remains as sensitivity. TensorFlow-free for rescoring/tests.
"""

from __future__ import annotations

import numpy as np

# Default underage/overage cost ratios (co normalized to 1).
DEFAULT_CU_CO_RATIOS: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0, 10.0, 20.0)
# Holding cost per unit of stock kept on a zero day, relative to unit revenue.
DEFAULT_HOLDING_WEIGHTS: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0)

PRIMARY_INVENTORY_METRIC = "inventory_nv_cost_rounded_cu2"
SERVICE_CRITICAL_INVENTORY_METRIC = "inventory_nv_cost_rounded_cu10"
PRIMARY_SALES_REVENUE_LOSS_METRIC = "sales_revenue_loss_units"
PRIMARY_SALES_REVENUE_LOSS_METRIC_ROUNDED = "sales_revenue_loss_units_rounded"
PRIMARY_COMBINED_OPS_METRIC = "combined_ops_cost_h0p1"
PRIMARY_HOLDING_METRIC = "inventory_holding_cost_zero"


def _round_forecast(yhat) -> np.ndarray:
    """Match ``forecast_postprocess.round_forecast`` (clip at 0, nearest int)."""
    y = np.asarray(yhat, dtype=np.float64)
    return np.rint(np.maximum(y, 0.0))


def _cu_tag(ratio: float) -> str:
    """Stable field suffix for a cu/co ratio (1 → cu1, 2.5 → cu2p5)."""
    r = float(ratio)
    if abs(r - round(r)) < 1e-12:
        return f"cu{int(round(r))}"
    return "cu" + str(r).replace(".", "p")


def _h_tag(weight: float) -> str:
    """Stable field suffix for holding weight (0.1 → h0p1, 1 → h1)."""
    w = float(weight)
    if abs(w - round(w)) < 1e-12:
        return f"h{int(round(w))}"
    return "h" + str(w).replace(".", "p")


def inventory_cost_metrics(
    y,
    yhat,
    *,
    cu_co_ratios: tuple[float, ...] = DEFAULT_CU_CO_RATIOS,
    holding_weights: tuple[float, ...] = DEFAULT_HOLDING_WEIGHTS,
    include_rounded: bool = True,
) -> dict:
    """Newsvendor-style inventory / planning costs on forecast error.

    Per row (with overage cost ``co=1``, underage ``cu = ratio * co``)::

        under = max(y - yhat, 0)   # stockout / lost sales proxy
        over  = max(yhat - y, 0)   # holding / excess inventory proxy
        cost  = cu * under + co * over

    Also reports:
      - ``inventory_holding_cost_zero``: mean stock kept on no-sale days
        (= mean ŷ | y=0); carrying cost, not revenue loss
      - ``sales_revenue_loss_units`` / ``_rate`` / ``_per_day``: unmet demand
        on sale days (unit price = 1)
      - ``combined_ops_cost_h*``: revenue_loss_per_day + h × holding_per_day
        where holding_per_day = zero_rate × inventory_holding_cost_zero
      - Newsvendor ``inventory_nv_cost_cu*`` as sensitivity

    When ``include_rounded``, the same suite is repeated on count-rounded
    ``yhat``. ``cu/co = 1`` recovers MAE / MAE-rounded.
    """
    y = np.asarray(y, np.float64).reshape(-1)
    yhat = np.maximum(np.asarray(yhat, np.float64).reshape(-1), 0.0)
    n = len(y)
    out: dict = {}
    if n == 0:
        for key in (
            "inventory_mean_under",
            "inventory_mean_over",
            "inventory_holding_proxy_zero",
            "inventory_stockout_proxy_nz",
            "inventory_fill_rate_nz",
            "inventory_qty_fill_nz",
            "sales_revenue_loss_units",
            "sales_revenue_loss_rate",
            "sales_revenue_loss_per_day",
            "unnecessary_restock_zero",
            "inventory_holding_cost_zero",
            "inventory_holding_cost_zero_per_day",
        ):
            out[key] = None
        for ratio in cu_co_ratios:
            out[f"inventory_nv_cost_{_cu_tag(ratio)}"] = None
            if include_rounded:
                out[f"inventory_nv_cost_rounded_{_cu_tag(ratio)}"] = None
        for h in holding_weights:
            out[f"combined_ops_cost_{_h_tag(h)}"] = None
            if include_rounded:
                out[f"combined_ops_cost_{_h_tag(h)}_rounded"] = None
        if include_rounded:
            for key in (
                "sales_revenue_loss_units_rounded",
                "sales_revenue_loss_rate_rounded",
                "sales_revenue_loss_per_day_rounded",
                "unnecessary_restock_zero_rounded",
                "inventory_holding_cost_zero_rounded",
                "inventory_holding_cost_zero_per_day_rounded",
            ):
                out[key] = None
        return out

    nz = y > 0
    z = ~nz
    under = np.maximum(y - yhat, 0.0)
    over = np.maximum(yhat - y, 0.0)
    out["inventory_mean_under"] = float(under.mean())
    out["inventory_mean_over"] = float(over.mean())
    out["inventory_holding_proxy_zero"] = (
        float(over[z].mean()) if z.any() else None
    )
    out["inventory_stockout_proxy_nz"] = (
        float(under[nz].mean()) if nz.any() else None
    )
    out.update(
        _sales_and_holding_fields(
            y, yhat, under, nz, z, holding_weights=holding_weights, suffix=""
        )
    )
    if nz.any():
        out["inventory_fill_rate_nz"] = float(np.mean(yhat[nz] >= y[nz]))
        y_nz_sum = float(y[nz].sum())
        out["inventory_qty_fill_nz"] = (
            float(np.minimum(yhat[nz], y[nz]).sum() / y_nz_sum)
            if y_nz_sum > 0
            else None
        )
    else:
        out["inventory_fill_rate_nz"] = None
        out["inventory_qty_fill_nz"] = None

    for ratio in cu_co_ratios:
        cu = float(ratio)
        co = 1.0
        out[f"inventory_nv_cost_{_cu_tag(ratio)}"] = float(
            np.mean(cu * under + co * over)
        )

    if include_rounded:
        yhat_r = _round_forecast(yhat)
        under_r = np.maximum(y - yhat_r, 0.0)
        over_r = np.maximum(yhat_r - y, 0.0)
        out["inventory_mean_under_rounded"] = float(under_r.mean())
        out["inventory_mean_over_rounded"] = float(over_r.mean())
        out["inventory_holding_proxy_zero_rounded"] = (
            float(over_r[z].mean()) if z.any() else None
        )
        out["inventory_stockout_proxy_nz_rounded"] = (
            float(under_r[nz].mean()) if nz.any() else None
        )
        out.update(
            _sales_and_holding_fields(
                y,
                yhat_r,
                under_r,
                nz,
                z,
                holding_weights=holding_weights,
                suffix="_rounded",
            )
        )
        if nz.any():
            out["inventory_fill_rate_nz_rounded"] = float(
                np.mean(yhat_r[nz] >= y[nz])
            )
            y_nz_sum = float(y[nz].sum())
            out["inventory_qty_fill_nz_rounded"] = (
                float(np.minimum(yhat_r[nz], y[nz]).sum() / y_nz_sum)
                if y_nz_sum > 0
                else None
            )
        else:
            out["inventory_fill_rate_nz_rounded"] = None
            out["inventory_qty_fill_nz_rounded"] = None
        for ratio in cu_co_ratios:
            cu = float(ratio)
            out[f"inventory_nv_cost_rounded_{_cu_tag(ratio)}"] = float(
                np.mean(cu * under_r + over_r)
            )
    return out


def _sales_and_holding_fields(
    y: np.ndarray,
    yhat: np.ndarray,
    under: np.ndarray,
    nz: np.ndarray,
    z: np.ndarray,
    *,
    holding_weights: tuple[float, ...],
    suffix: str,
) -> dict:
    """Revenue loss (sale days) + inventory holding cost (no-sale days).

    - Sale: ``sales_revenue_loss_*`` — unmet demand (unit revenue = 1).
    - Zero: ``inventory_holding_cost_zero`` — mean ŷ (stock kept; carrying cost).
      Alias ``unnecessary_restock_zero`` kept for the same quantity.
    - Combined: ``combined_ops_cost_h*`` = rev_per_day + h × hold_per_day.
    """
    out: dict = {}
    if nz.any():
        under_nz = float(under[nz].mean())
        y_nz_mean = float(y[nz].mean())
        out[f"sales_revenue_loss_units{suffix}"] = under_nz
        out[f"sales_revenue_loss_rate{suffix}"] = (
            float(under_nz / y_nz_mean) if y_nz_mean > 0 else None
        )
    else:
        out[f"sales_revenue_loss_units{suffix}"] = None
        out[f"sales_revenue_loss_rate{suffix}"] = None
    rev_per_day = float(under.mean())
    out[f"sales_revenue_loss_per_day{suffix}"] = rev_per_day

    hold_zero = float(yhat[z].mean()) if z.any() else 0.0
    zero_rate = float(z.mean()) if len(z) else 0.0
    hold_per_day = zero_rate * hold_zero
    # Same quantity: forecast stock sitting on a no-sale day.
    out[f"unnecessary_restock_zero{suffix}"] = (
        hold_zero if z.any() else None
    )
    out[f"inventory_holding_cost_zero{suffix}"] = (
        hold_zero if z.any() else None
    )
    out[f"inventory_holding_cost_zero_per_day{suffix}"] = float(hold_per_day)

    for h in holding_weights:
        tag = _h_tag(h)
        key = f"combined_ops_cost_{tag}{suffix}"
        out[key] = float(rev_per_day + float(h) * hold_per_day)
    return out


def inventory_cost_from_kpi_summary(
    overall: dict,
    *,
    cu_co_ratios: tuple[float, ...] = DEFAULT_CU_CO_RATIOS,
    holding_weights: tuple[float, ...] = DEFAULT_HOLDING_WEIGHTS,
) -> dict:
    """Recover mean under/over (continuous yhat) from logged ``kpi_block`` fields.

    Locked bake-off JSONs store aggregates only (no per-row ``y`` / ``yhat``).
    Mean under-forecast and over-forecast are algebraically recoverable from
    ``mae_nonzero``, ``bias_nonzero``, ``mean_final``, ``mean_actual``, and
    zero rate — so newsvendor costs on *continuous* forecasts are exact.

    Limitation: rounded-forecast inventory costs cannot be recovered without
    per-row preds (rounding is nonlinear). Those fields are omitted here;
    re-run eval (or save preds) for ``inventory_nv_cost_rounded_*``.
    """
    n = overall.get("n_rows")
    n_nz = overall.get("n_nonzero")
    mae_nz = overall.get("mae_nonzero")
    bias_nz = overall.get("bias_nonzero")
    mean_final = overall.get("mean_final")
    mean_actual = overall.get("mean_actual")
    under_rate = overall.get("underforecast_rate_nonzero")

    required = (n, n_nz, mae_nz, bias_nz, mean_final, mean_actual)
    if any(v is None for v in required) or int(n) <= 0:
        return {
            "inventory_rescore_source": "unavailable",
            "inventory_rescore_note": (
                "Need n_rows, n_nonzero, mae_nonzero, bias_nonzero, "
                "mean_final, mean_actual."
            ),
        }

    n = int(n)
    n_nz = int(n_nz)
    n_z = n - n_nz
    if n_nz <= 0 or n_z <= 0:
        return {
            "inventory_rescore_source": "unavailable",
            "inventory_rescore_note": "Need both zero and nonzero rows.",
        }

    pi = n_nz / n
    mae_nz = float(mae_nz)
    bias_nz = float(bias_nz)
    mean_final = float(mean_final)
    mean_actual = float(mean_actual)

    # On nz: under_mean + over_mean = mae; under_mean - over_mean = -bias_nz
    under_nz = 0.5 * (mae_nz - bias_nz)
    over_nz = 0.5 * (mae_nz + bias_nz)
    under_nz = max(under_nz, 0.0)
    over_nz = max(over_nz, 0.0)

    mean_y_nz = mean_actual / pi
    mean_yhat_nz = bias_nz + mean_y_nz
    mean_yhat_z = (mean_final - pi * mean_yhat_nz) / (1.0 - pi)
    mean_yhat_z = max(mean_yhat_z, 0.0)

    mean_under = pi * under_nz
    mean_over = pi * over_nz + (1.0 - pi) * mean_yhat_z

    hold_per_day = (1.0 - pi) * mean_yhat_z
    out: dict = {
        "inventory_rescore_source": "kpi_summary_continuous",
        "inventory_rescore_note": (
            "Exact mean under/over for continuous yhat from aggregates; "
            "rounded nv / revenue-loss_rounded require per-row predictions."
        ),
        "inventory_mean_under": float(mean_under),
        "inventory_mean_over": float(mean_over),
        "inventory_holding_proxy_zero": float(mean_yhat_z),
        "inventory_stockout_proxy_nz": float(under_nz),
        "inventory_fill_rate_nz": (
            float(1.0 - under_rate) if under_rate is not None else None
        ),
        "inventory_qty_fill_nz": (
            float(1.0 - under_nz / mean_y_nz) if mean_y_nz > 0 else None
        ),
        # Sale days: revenue loss. Zero days: holding cost (no revenue loss).
        "sales_revenue_loss_units": float(under_nz),
        "sales_revenue_loss_rate": (
            float(under_nz / mean_y_nz) if mean_y_nz > 0 else None
        ),
        "sales_revenue_loss_per_day": float(mean_under),
        "unnecessary_restock_zero": float(mean_yhat_z),
        "inventory_holding_cost_zero": float(mean_yhat_z),
        "inventory_holding_cost_zero_per_day": float(hold_per_day),
    }
    for h in holding_weights:
        out[f"combined_ops_cost_{_h_tag(h)}"] = float(
            mean_under + float(h) * hold_per_day
        )
    for ratio in cu_co_ratios:
        cu = float(ratio)
        out[f"inventory_nv_cost_{_cu_tag(ratio)}"] = float(
            cu * mean_under + mean_over
        )
    return out


def _kpi_overall_from_block(block) -> dict:
    """Accept either ``{overall: {...}}`` or a flat kpi dict."""
    if not isinstance(block, dict):
        return {}
    if "overall" in block and isinstance(block["overall"], dict):
        # Prefer nested overall when it looks like a kpi block.
        nested = block["overall"]
        if "n_rows" in nested or "mae_nonzero" in nested:
            return nested
    if "n_rows" in block or "mae_nonzero" in block:
        return block
    return {}


def rescore_multihorizon_model_payload(
    model_payload: dict,
    *,
    cu_co_ratios: tuple[float, ...] = DEFAULT_CU_CO_RATIOS,
    holding_weights: tuple[float, ...] = DEFAULT_HOLDING_WEIGHTS,
) -> dict:
    """Attach sales/holding/combined ops metrics to an MH model block.

    Expects ``by_horizon[*]`` (flat kpi or ``.overall``) and optional
    ``mean_1_to_H`` with fields needed by ``inventory_cost_from_kpi_summary``.
    Returns a compact ranking-friendly dict keyed by horizon label.
    """
    out: dict = {"by_horizon": {}, "mean_1_to_H": None}
    by_h = model_payload.get("by_horizon") or {}
    for h_key, block in by_h.items():
        overall = _kpi_overall_from_block(block)
        inv = inventory_cost_from_kpi_summary(
            overall,
            cu_co_ratios=cu_co_ratios,
            holding_weights=holding_weights,
        )
        out["by_horizon"][str(h_key)] = {
            "iwmae_rounded": overall.get("iwmae_rounded"),
            "bias": overall.get("bias"),
            **inv,
        }
    mean_block = _kpi_overall_from_block(model_payload.get("mean_1_to_H") or {})
    if mean_block:
        inv = inventory_cost_from_kpi_summary(
            mean_block,
            cu_co_ratios=cu_co_ratios,
            holding_weights=holding_weights,
        )
        out["mean_1_to_H"] = {
            "iwmae_rounded": mean_block.get("iwmae_rounded"),
            "bias": mean_block.get("bias"),
            **inv,
        }
    return out


def rank_multihorizon_ops(
    models: dict,
    *,
    horizon: str = "mean",
    sort_key: str = PRIMARY_COMBINED_OPS_METRIC,
    cu_co_ratios: tuple[float, ...] = DEFAULT_CU_CO_RATIOS,
    holding_weights: tuple[float, ...] = DEFAULT_HOLDING_WEIGHTS,
) -> list:
    """Rank MH models at one horizon (``1``/``7``/``14``/``mean``) by ``sort_key``."""
    rows = []
    for name, payload in models.items():
        scored = rescore_multihorizon_model_payload(
            payload,
            cu_co_ratios=cu_co_ratios,
            holding_weights=holding_weights,
        )
        if horizon == "mean":
            block = scored.get("mean_1_to_H") or {}
        else:
            block = (scored.get("by_horizon") or {}).get(str(horizon)) or {}
        if not block or block.get(sort_key) is None:
            continue
        rows.append({"model": name, "horizon": horizon, **block})
    return sorted(
        rows,
        key=lambda r: (
            r.get(sort_key) is None,
            r.get(sort_key) if r.get(sort_key) is not None else 1e9,
        ),
    )


# ---------------------------------------------------------------------------
# Decision economics: cost ratio selector (not pure forecast accuracy)
# ---------------------------------------------------------------------------

# Holding cost is a *policy* input (carry rate × unit cost, capital charge, etc.).
# It is NOT inferred from margin. Margin only sets C_lost = margin * unit_price.
DEFAULT_POLICY_HOLDING_COST_PER_UNIT = 0.10
DEFAULT_UNIT_PRICE = 1.0
DEFAULT_MARGINS: tuple[float, ...] = (0.08, 0.25, 0.55)


def margin_regimes_from_policy(
    *,
    holding_cost_per_unit: float = DEFAULT_POLICY_HOLDING_COST_PER_UNIT,
    margins: tuple[float, ...] = DEFAULT_MARGINS,
    unit_price: float = DEFAULT_UNIT_PRICE,
) -> dict:
    """Build low/mid/high margin regimes with a shared inventory holding cost.

    ``holding_cost_per_unit`` comes from inventory/finance policy (same across
    regimes unless the business truly has different carry by segment).
    Only ``margin`` changes → only C_lost and r = C_lost/C_hold change.
    """
    labels = ("low_margin", "mid_margin", "high_margin")
    if len(margins) != 3:
        raise ValueError("expected three margins: low, mid, high")
    h = float(holding_cost_per_unit)
    out = {}
    for key, m in zip(labels, margins):
        out[key] = {
            "label": key.replace("_", " ").title(),
            "unit_price": float(unit_price),
            "margin": float(m),
            "holding_cost_per_unit": h,
            "note": (
                f"C_hold={h} from policy (fixed); "
                f"C_lost=margin*price={float(m)*float(unit_price):.4f}; "
                f"r=C_lost/C_hold={float(m)*float(unit_price)/h:.3f}"
            ),
        }
    return out


# Default regimes: one policy C_hold, three catalog margins.
DEFAULT_MARGIN_REGIMES: dict = margin_regimes_from_policy()


def cost_ratio_from_margin(margin: float, holding_cost_per_unit: float) -> float:
    """r = C_lost / C_hold with C_lost = margin * unit_price (price=1)."""
    h = float(holding_cost_per_unit)
    if h <= 0:
        raise ValueError("holding_cost_per_unit must be > 0")
    return float(margin) / h


def decision_cost_components(inv_summary: dict) -> dict:
    """Extract mean under (lost sales qty) and holding qty from KPI rescore.

    Policy (same for every model):
      - Lost-sales units: ``inventory_mean_under`` (calendar-day mean underage)
      - Holding units: ``inventory_holding_cost_zero_per_day`` (stock sitting on
        no-sale days, averaged per calendar day)

    Total cost at ratio r = C_lost/C_hold with C_hold normalized to 1::

        cost(r) = r * U + H
    """
    u = inv_summary.get("inventory_mean_under")
    h = inv_summary.get("inventory_holding_cost_zero_per_day")
    if u is None or h is None:
        return {"U": None, "H": None}
    return {"U": float(u), "H": float(h)}


def decision_cost(u: float, h: float, r: float, *, c_hold: float = 1.0) -> float:
    """Simulated total ops cost: C_lost*U + C_hold*H with r=C_lost/C_hold."""
    return float(c_hold) * (float(r) * float(u) + float(h))


def profit_loss(
    inv_summary: dict,
    *,
    margin: float,
    holding_cost_per_unit: float,
    unit_price: float = 1.0,
) -> dict:
    """Profit / contribution loss under a margin + holding regime.

    - Revenue (contribution) loss = margin * unit_price * unmet_per_day
    - Holding loss = holding_cost_per_unit * hold_per_day
    - Total = sum (lower is better)
    """
    u = float(inv_summary["inventory_mean_under"])
    h = float(inv_summary["inventory_holding_cost_zero_per_day"])
    rev_loss = float(margin) * float(unit_price) * u
    hold_loss = float(holding_cost_per_unit) * h
    r = cost_ratio_from_margin(margin * unit_price, holding_cost_per_unit)
    return {
        "margin": float(margin),
        "holding_cost_per_unit": float(holding_cost_per_unit),
        "unit_price": float(unit_price),
        "cost_ratio_r": float(r),
        "contribution_loss_per_day": float(rev_loss),
        "holding_loss_per_day": float(hold_loss),
        "total_profit_loss_per_day": float(rev_loss + hold_loss),
        "U": u,
        "H": h,
    }


def cost_curve(
    u: float,
    h: float,
    r_grid: np.ndarray | list | tuple,
    *,
    c_hold: float = 1.0,
) -> dict:
    """Evaluate cost(r) = c_hold * (r*U + H) on a grid."""
    rs = np.asarray(r_grid, dtype=np.float64).reshape(-1)
    costs = np.array([decision_cost(u, h, float(r), c_hold=c_hold) for r in rs])
    return {"r": rs.tolist(), "cost": costs.tolist()}


def pairwise_crossover_r(u_a: float, h_a: float, u_b: float, h_b: float) -> float | None:
    """r* where cost_a(r) = cost_b(r). Left of r*: lower H wins; right: lower U.

    cost = r*U + H  ⇒  r*(U_a - U_b) = H_b - H_a
    r* = (H_b - H_a) / (U_a - U_b) when denominators allow.
    """
    du = float(u_a) - float(u_b)
    dh = float(h_b) - float(h_a)
    if abs(du) < 1e-15:
        return None
    r_star = dh / du
    if r_star <= 0:
        return None
    return float(r_star)


def select_model_by_r(
    models: dict[str, dict],
    r: float,
    *,
    c_hold: float = 1.0,
) -> dict:
    """Pick lowest-cost model at cost ratio ``r``.

    ``models`` maps name → ``{"U", "H"}`` (from ``decision_cost_components``).
    """
    scored = []
    for name, comp in models.items():
        u, h = comp["U"], comp["H"]
        if u is None or h is None:
            continue
        scored.append(
            {
                "model": name,
                "cost": decision_cost(u, h, r, c_hold=c_hold),
                "U": float(u),
                "H": float(h),
                "r": float(r),
            }
        )
    scored.sort(key=lambda x: x["cost"])
    return {
        "r": float(r),
        "winner": scored[0]["model"] if scored else None,
        "ranking": scored,
    }


def decision_economics_report(
    model_inv: dict[str, dict],
    *,
    r_grid: np.ndarray | list | tuple | None = None,
    margin_regimes: dict | None = None,
    pair: tuple[str, str] = ("TST lite", "plain DS"),
) -> dict:
    """Full decision-economics pack: curves, crossover, margin regime table.

    ``model_inv`` maps display name → output of ``inventory_cost_from_kpi_summary``.
    """
    if r_grid is None:
        r_grid = np.concatenate(
            [
                np.linspace(0.25, 2.0, 8),
                np.linspace(2.5, 12.0, 20),
                np.linspace(14.0, 40.0, 14),
            ]
        )
    regimes = margin_regimes or DEFAULT_MARGIN_REGIMES

    components = {}
    curves = {}
    for name, inv in model_inv.items():
        comp = decision_cost_components(inv)
        components[name] = {
            **comp,
            "sales_revenue_loss_units": inv.get("sales_revenue_loss_units"),
            "inventory_holding_cost_zero": inv.get("inventory_holding_cost_zero"),
            "iwmae_rounded": inv.get("iwmae_rounded"),
        }
        if comp["U"] is not None and comp["H"] is not None:
            curves[name] = cost_curve(comp["U"], comp["H"], r_grid)

    a, b = pair
    r_star = None
    if a in components and b in components:
        ca, cb = components[a], components[b]
        if ca["U"] is not None and cb["U"] is not None:
            r_star = pairwise_crossover_r(ca["U"], ca["H"], cb["U"], cb["H"])

    selector = {
        "rule": (
            f"For r < r*: prefer lower-holding model ({a} if it has lower H). "
            f"For r > r*: prefer lower-stockout model ({b} if it has lower U)."
        ),
        "pair": list(pair),
        "r_star": r_star,
        "interpretation": {
            "left_of_r_star": "Holding hurts relatively more → cooler / lower-overstock posture",
            "right_of_r_star": "Lost sales hurt more → lower-stockout posture",
        },
    }

    regime_table = {}
    for key, spec in regimes.items():
        r = cost_ratio_from_margin(spec["margin"], spec["holding_cost_per_unit"])
        pick = select_model_by_r(components, r)
        per_model = {}
        for name, inv in model_inv.items():
            per_model[name] = profit_loss(
                inv,
                margin=spec["margin"],
                holding_cost_per_unit=spec["holding_cost_per_unit"],
                unit_price=spec.get("unit_price", 1.0),
            )
        regime_table[key] = {
            **spec,
            "cost_ratio_r": r,
            "winner": pick.get("winner"),
            "ranking": pick.get("ranking"),
            "profit_loss_by_model": per_model,
        }

    return {
        "framing": (
            "Decision economics, not universal IWMAE winner: models are risk "
            "profiles. r = C_lost/C_hold. cost(r)=r*U+H with U=mean underage, "
            "H=mean holding on no-sale days (per calendar day)."
        ),
        "components": components,
        "curves": curves,
        "crossover": selector,
        "margin_regimes": regime_table,
    }


# ---------------------------------------------------------------------------
# Model operating cost proxy (transparent, intentionally coarse)
# ---------------------------------------------------------------------------

# Fixed relative ops tiers used when wall-time is missing or as a documented
# sensitivity check. Units are dimensionless multipliers, not dollars.
DEFAULT_MODEL_OPS_TIER: dict[str, float] = {
    "lightgbm": 1.0,
    "LightGBM": 1.0,
    "deepsequence": 1.5,
    "plain DS": 1.5,
    "DeepSequence": 1.5,
    "temporal_transformer": 1.5,
    "TST lite": 1.5,
    "tft_lite": 2.0,
    "TFT lite": 2.0,
    "deepar_lite": 2.0,
    "DeepAR lite": 2.0,
}

# Scale so a tier-1 model costs this many "profit units" per calendar day.
# Chosen small vs typical inventory loss magnitudes so C_model is a tie-breaker,
# not a dominant driver. Documented as a proxy — not a measured cloud bill.
DEFAULT_MODEL_OPS_BASE_PER_DAY = 0.01


def model_ops_cost_from_tiers(
    model_key_or_name: str,
    *,
    tiers: dict[str, float] | None = None,
    base_per_day: float = DEFAULT_MODEL_OPS_BASE_PER_DAY,
) -> float:
    """Map a model name/key to a fixed relative per-day ops penalty."""
    table = tiers or DEFAULT_MODEL_OPS_TIER
    mult = float(table.get(model_key_or_name, 1.5))
    return float(base_per_day) * mult


def model_ops_cost_from_train_seconds(
    train_seconds_by_model: dict[str, float],
    *,
    base_per_day: float = DEFAULT_MODEL_OPS_BASE_PER_DAY,
    reference_seconds: float | None = None,
) -> dict[str, float]:
    """Normalize wall-clock train time to a small per-day ops penalty.

    Fastest model (or ``reference_seconds``) gets multiplier 1.0; others scale
    linearly with train_seconds / ref. Missing / non-positive times → tier
    fallback is the caller's responsibility.
    """
    positive = {
        k: float(v)
        for k, v in train_seconds_by_model.items()
        if v is not None and float(v) > 0
    }
    if not positive:
        return {}
    ref = (
        float(reference_seconds)
        if reference_seconds is not None and float(reference_seconds) > 0
        else min(positive.values())
    )
    return {
        k: float(base_per_day) * (sec / ref) for k, sec in positive.items()
    }


def profit_with_model_ops(
    inv_summary: dict,
    *,
    margin: float,
    holding_cost_per_unit: float,
    model_ops_cost_per_day: float,
    unit_price: float = 1.0,
    mean_demand_per_day: float | None = None,
) -> dict:
    """π proxy = revenue − inventory cost − model ops cost (higher is better).

    Inventory side matches ``profit_loss`` (contribution loss from underage +
    holding on no-sale days). Revenue uses ``mean_demand_per_day`` when given
    (shared across models at a horizon), else ``mean_actual`` on the summary,
    else falls back to underage-only ranking via negative total loss.
    """
    pl = profit_loss(
        inv_summary,
        margin=margin,
        holding_cost_per_unit=holding_cost_per_unit,
        unit_price=unit_price,
    )
    demand = mean_demand_per_day
    if demand is None:
        demand = inv_summary.get("mean_actual")
    c_model = float(model_ops_cost_per_day)
    if demand is not None:
        revenue = float(margin) * float(unit_price) * float(demand)
        # Fulfilled ≈ demand − U; revenue_fulfilled = margin*price*(demand−U)
        # π = margin*price*(demand−U) − C_hold*H − C_model
        #   = margin*price*demand − (margin*price*U + C_hold*H) − C_model
        pi = revenue - pl["total_profit_loss_per_day"] - c_model
        revenue_out = float(revenue)
    else:
        # Same ranking as maximizing −(inv_loss + C_model)
        pi = -pl["total_profit_loss_per_day"] - c_model
        revenue_out = None
    return {
        **pl,
        "model_ops_cost_per_day": c_model,
        "revenue_proxy_per_day": revenue_out,
        "pi_per_day": float(pi),
        "note": (
            "π = revenue_proxy − inv_loss − C_model. C_model is a transparent "
            "proxy (train-time or fixed tier), not a measured production bill."
        ),
    }
