"""
Causal intermittent / regressor demand-history features.

All features at date t use only that SKU's history with ds < t (no same-day
or future leakage). Suitable for train/val/test batch transforms and for
online inference via per-SKU state updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple, Union
import json

import numpy as np
import pandas as pd

# Default cold-start: never sold before prediction date
DEFAULT_DAYS_SINCE_SENTINEL = -1.0
DEFAULT_LAGS = (1, 2, 7)

INTERMITTENT_FEATURE_NAMES = (
    "days_since_last_sale",
    "last_sale_quantity",
    "lifetime_cumsum",
)


@dataclass
class SKUDemandState:
    """Causal demand memory for one SKU (as-of last applied observation)."""

    as_of_date: Optional[pd.Timestamp] = None
    last_sale_date: Optional[pd.Timestamp] = None
    last_sale_quantity: float = 0.0
    lifetime_cumsum: float = 0.0
    # Newest demand last; length kept <= max_lag
    recent_demand: List[float] = field(default_factory=list)
    max_lag: int = 7
    days_since_sentinel: float = DEFAULT_DAYS_SINCE_SENTINEL

    def copy(self) -> "SKUDemandState":
        return SKUDemandState(
            as_of_date=self.as_of_date,
            last_sale_date=self.last_sale_date,
            last_sale_quantity=float(self.last_sale_quantity),
            lifetime_cumsum=float(self.lifetime_cumsum),
            recent_demand=list(self.recent_demand),
            max_lag=int(self.max_lag),
            days_since_sentinel=float(self.days_since_sentinel),
        )

    def features_at(self, date: Union[str, pd.Timestamp], lags: Iterable[int] = DEFAULT_LAGS) -> Dict[str, float]:
        """Build regressor features for prediction date ``date`` (no update)."""
        date = pd.Timestamp(date).normalize()
        if self.last_sale_date is None:
            days_since = float(self.days_since_sentinel)
            last_qty = 0.0
        else:
            days_since = float((date - pd.Timestamp(self.last_sale_date).normalize()).days)
            last_qty = float(self.last_sale_quantity)

        out = {
            "days_since_last_sale": days_since,
            "last_sale_quantity": last_qty,
            "lifetime_cumsum": float(self.lifetime_cumsum),
        }
        for lag in lags:
            lag = int(lag)
            if lag <= 0:
                raise ValueError(f"lag must be positive, got {lag}")
            if len(self.recent_demand) >= lag:
                out[f"lag_{lag}"] = float(self.recent_demand[-lag])
            else:
                out[f"lag_{lag}"] = 0.0
        return out

    def update(self, date: Union[str, pd.Timestamp], quantity: float) -> "SKUDemandState":
        """
        Apply an observation at ``date`` (actual or recursive prediction).

        Raises if ``date`` is not strictly after ``as_of_date`` (when set).
        """
        date = pd.Timestamp(date).normalize()
        quantity = float(quantity)
        if self.as_of_date is not None and date <= pd.Timestamp(self.as_of_date).normalize():
            raise ValueError(
                f"Refusing non-causal update: date={date.date()} as_of={self.as_of_date.date()}"
            )

        self.as_of_date = date
        self.lifetime_cumsum += quantity
        self.recent_demand.append(quantity)
        if len(self.recent_demand) > self.max_lag:
            self.recent_demand = self.recent_demand[-self.max_lag :]

        if quantity > 0.0:
            self.last_sale_date = date
            self.last_sale_quantity = quantity
        return self

    def to_dict(self) -> dict:
        d = asdict(self)
        d["as_of_date"] = None if self.as_of_date is None else str(pd.Timestamp(self.as_of_date).date())
        d["last_sale_date"] = (
            None if self.last_sale_date is None else str(pd.Timestamp(self.last_sale_date).date())
        )
        return d

    @classmethod
    def from_dict(cls, d: Mapping) -> "SKUDemandState":
        return cls(
            as_of_date=None if d.get("as_of_date") in (None, "") else pd.Timestamp(d["as_of_date"]),
            last_sale_date=(
                None if d.get("last_sale_date") in (None, "") else pd.Timestamp(d["last_sale_date"])
            ),
            last_sale_quantity=float(d.get("last_sale_quantity", 0.0)),
            lifetime_cumsum=float(d.get("lifetime_cumsum", 0.0)),
            recent_demand=[float(x) for x in d.get("recent_demand", [])],
            max_lag=int(d.get("max_lag", 7)),
            days_since_sentinel=float(d.get("days_since_sentinel", DEFAULT_DAYS_SINCE_SENTINEL)),
        )


StateMap = Dict[str, SKUDemandState]


def empty_state(
    max_lag: int = 7,
    days_since_sentinel: float = DEFAULT_DAYS_SINCE_SENTINEL,
) -> SKUDemandState:
    return SKUDemandState(max_lag=max_lag, days_since_sentinel=days_since_sentinel)


def copy_states(states: Optional[Mapping[str, SKUDemandState]]) -> StateMap:
    if not states:
        return {}
    return {str(k): v.copy() for k, v in states.items()}


def save_states(states: Mapping[str, SKUDemandState], path: str) -> None:
    payload = {str(k): v.to_dict() for k, v in states.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def load_states(path: str) -> StateMap:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {str(k): SKUDemandState.from_dict(v) for k, v in payload.items()}


def features_from_state(
    state: SKUDemandState,
    date: Union[str, pd.Timestamp],
    lags: Iterable[int] = DEFAULT_LAGS,
) -> Dict[str, float]:
    return state.features_at(date, lags=lags)


def update_state(
    state: SKUDemandState,
    date: Union[str, pd.Timestamp],
    quantity: float,
) -> SKUDemandState:
    return state.update(date, quantity)


def transform_panel(
    df: pd.DataFrame,
    *,
    id_col: str = "id_var",
    date_col: str = "ds",
    quantity_col: str = "Quantity",
    lags: Iterable[int] = DEFAULT_LAGS,
    prior_states: Optional[Mapping[str, SKUDemandState]] = None,
    days_since_sentinel: float = DEFAULT_DAYS_SINCE_SENTINEL,
    return_states: bool = True,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, StateMap]]:
    """
    Causally transform a demand panel into lag + intermittent regressor features.

    For each row at date t, features use only that SKU's history with ds < t.
    Optional ``prior_states`` warm-starts SKUs (e.g. end of train → val/test).

    Returns feature frame aligned to ``df`` row order after sorting by
    (id, date). Caller should pass an already row-aligned holiday frame if
    concatenating elsewhere — this function re-sorts; use
    ``transform_panel_preserve_order`` when index alignment matters, or sort
    holidays the same way.
    """
    lags = tuple(int(x) for x in lags)
    max_lag = max(lags) if lags else 7

    work = df[[id_col, date_col, quantity_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col])
    work[quantity_col] = work[quantity_col].astype(float)
    work[id_col] = work[id_col].astype(str)
    work = work.sort_values([id_col, date_col], kind="mergesort")
    original_index = work.index.to_numpy()

    states = copy_states(prior_states)
    n = len(work)
    col_names = [f"lag_{lag}" for lag in lags] + list(INTERMITTENT_FEATURE_NAMES)
    cols = {name: np.empty(n, dtype=np.float64) for name in col_names}

    ids = work[id_col].to_numpy()
    dates = work[date_col].to_numpy()
    qtys = work[quantity_col].to_numpy(dtype=np.float64)

    i = 0
    while i < n:
        sku = ids[i]
        j = i + 1
        while j < n and ids[j] == sku:
            j += 1

        state = states.get(sku)
        if state is None:
            state = empty_state(max_lag=max_lag, days_since_sentinel=days_since_sentinel)
        else:
            # Ensure max_lag covers configured lags
            state.max_lag = max(state.max_lag, max_lag)
            state.days_since_sentinel = days_since_sentinel

        for k in range(i, j):
            date_k = pd.Timestamp(dates[k]).normalize()
            feats = state.features_at(date_k, lags=lags)
            for name in INTERMITTENT_FEATURE_NAMES:
                cols[name][k] = feats[name]
            for lag in lags:
                cols[f"lag_{lag}"][k] = feats[f"lag_{lag}"]
            state.update(date_k, float(qtys[k]))

        states[sku] = state
        i = j

    feat_df = pd.DataFrame(cols, index=original_index)
    # Restore caller row order
    feat_df = feat_df.loc[df.index]
    if return_states:
        return feat_df, states
    return feat_df


def build_states_from_history(
    history_df: pd.DataFrame,
    *,
    id_col: str = "id_var",
    date_col: str = "ds",
    quantity_col: str = "Quantity",
    lags: Iterable[int] = DEFAULT_LAGS,
    days_since_sentinel: float = DEFAULT_DAYS_SINCE_SENTINEL,
) -> StateMap:
    """Scan history and return end-of-history states (for inference warm-start)."""
    _, states = transform_panel(
        history_df,
        id_col=id_col,
        date_col=date_col,
        quantity_col=quantity_col,
        lags=lags,
        days_since_sentinel=days_since_sentinel,
        return_states=True,
    )
    return states


class CausalInferenceFeatureServer:
    """
    Serve causal regressor features at inference from per-SKU state.

      states = build_states_from_history(history_df)
      server = CausalInferenceFeatureServer(states)
      feats = server.features_for(sku, date_t)  # no leakage
      server.observe(sku, date_t, y_actual)     # after observation
    """

    def __init__(
        self,
        states: Optional[Mapping[str, SKUDemandState]] = None,
        lags: Iterable[int] = DEFAULT_LAGS,
        days_since_sentinel: float = DEFAULT_DAYS_SINCE_SENTINEL,
    ):
        self.lags = tuple(int(x) for x in lags)
        self.days_since_sentinel = float(days_since_sentinel)
        self.states: StateMap = copy_states(states)

    def _get(self, sku: str) -> SKUDemandState:
        sku = str(sku)
        if sku not in self.states:
            self.states[sku] = empty_state(
                max_lag=max(self.lags) if self.lags else 7,
                days_since_sentinel=self.days_since_sentinel,
            )
        return self.states[sku]

    def features_for(self, sku: str, date: Union[str, pd.Timestamp]) -> Dict[str, float]:
        return self._get(sku).features_at(date, lags=self.lags)

    def observe(
        self,
        sku: str,
        date: Union[str, pd.Timestamp],
        quantity: float,
    ) -> SKUDemandState:
        return self._get(sku).update(date, quantity)

    def save(self, path: str) -> None:
        save_states(self.states, path)

    @classmethod
    def load(
        cls,
        path: str,
        lags: Iterable[int] = DEFAULT_LAGS,
        days_since_sentinel: float = DEFAULT_DAYS_SINCE_SENTINEL,
    ) -> "CausalInferenceFeatureServer":
        return cls(
            states=load_states(path),
            lags=lags,
            days_since_sentinel=days_since_sentinel,
        )
