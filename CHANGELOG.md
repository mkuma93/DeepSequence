# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added

- **Inventory / newsvendor planning metrics** in
  `deepsequence_hierarchical_attention/inventory_metrics.py` (also re-exported /
  attached by `deepsequence_hierarchical_attention.eval.helpers.kpi_block`). Reports asymmetric cost with
  configurable `cu/co` (defaults 1, 2, 3), holding proxy on zero days, stockout
  proxy on sale days, and fill-rate / quantity-fill. Primary planning key for
  future bake-offs: `inventory_nv_cost_rounded_cu2`. Re-score locked aggregate
  JSONs with `python ab_runs/rescore_inventory.py` (continuous yhat only;
  rounded costs need per-row preds).

### Removed (internal cleanup)

Dead code removal in `deepsequence_hierarchical_attention/components_lightweight.py`.
None of the removed symbols were exported from the package (`__all__`) or used by
tests, examples, or the documented public API, so this is a low-risk cleanup rather
than a change to the supported surface.

- **`create_lightweight_model_simple(...)`** — legacy Gather-based builder. Superseded
  by the production `build_hierarchical_model_lightweight(...)`, which is the only
  supported model-construction path.
- **`ComponentWithSKUWrapper`** — Keras layer used only by
  `create_lightweight_model_simple`; removed together with its sole consumer.
- **`combination_mode` parameter** of `build_hierarchical_model_lightweight(...)` — was
  accepted but never referenced in the function body (no-op). Callers that passed it
  explicitly by keyword will now receive a `TypeError`; no runtime behavior changes
  otherwise.

The retained public path is `build_hierarchical_model_lightweight(...)` (plus the
tested helper `create_model_from_features(...)`).

### Changed

- Bake-off scripts split **`--data_seed`** (SKU panel) from **`--train_seed`**
  (TF/numpy training RNG), with optional **`--sku_list` / `--save_sku_list`** to freeze
  the exact series set so every model shares the same train/val/test. Legacy
  `--seed` still sets both when the split flags are omitted.
- Removed builder-attached uncertainty variables and their compile-time loss wrappers.
  `AdaptiveWeightedModel` is now the sole adaptive loss-weighting path.
- Registered graph helper layers for Keras serialization and moved nested/lazy layer
  creation into `build()`.
- Replaced builder diagnostics and silent exception handling with module logging and
  an explicit serializable orthogonality-penalty layer.
- Split component attention, SKU setup, flag resolution, forecast combination, and
  regularization into focused helpers. Former tuning constants are exposed as named
  builder keyword arguments with behavior-preserving defaults.
- Unified Keras serializable package names under
  `deepsequence_hierarchical_attention` and extracted component / intermittent-head
  builders so the remaining graph construction is helper-driven.

### Fixed (architecture)

- **Learnable Fourier periods are calendar-correct and frequency-aware.** The daily
  defaults used whole numbers (30, 91, 365) even though months run 28-31 days and years
  365/366; they now use the mean Gregorian year (30.4375 / 91.3125 / 365.25). Periods are
  measured in *time steps*, so the daily tuple was also wrong for any other grain.
  `FOURIER_PERIODS_BY_FREQUENCY` / `fourier_periods_for_frequency()` supply per-frequency
  defaults (monthly `3, 6, 12`; weekly `4.35, 13.04, 26.09, 52.18`; quarterly `2, 4`),
  selected via the new `fourier_frequency` builder argument.
- **`max_period` no longer pins the yearly cycle at initialization.** It was hard-coded to
  365, which clipped a 365.25-day frequency on the first step and allowed meaningless
  multi-decade cycles at monthly grain. It now derives from the longest initial period
  (`FOURIER_MAX_PERIOD_SLACK`), overridable via `fourier_max_period`.
- **Short period lists no longer crash the learnable Fourier layer.** The "remaining
  frequencies will use log spacing" warning was never implemented, so fewer periods than
  `n_learnable_frequencies` produced an initializer/shape mismatch; `pad_fourier_periods()`
  now fills the gap.
- **Multi-horizon occurrence gate no longer bypasses the intermittent handler.** With
  `horizon > 1` the handler's zero-probability output was unused, so Keras pruned
  `IntermittentHandlerLightweight` out of the graph entirely and the gate was an
  unconstrained `Dense(horizon, sigmoid)`. The per-horizon gate is now
  `sigmoid(offsets + logit(P_nonzero))` via `HorizonGateFromBaseProbability`, with
  offsets zero-initialized so training starts from the handler's estimate.
- **Component attention no longer spends probability mass on disabled components.**
  `TemperatureSoftmax` accepts an `active_mask`; previously a disabled component (for
  example holiday on monthly data) absorbed roughly a quarter of the mixture even
  though its output is forced to zero, and the entropy penalty rewarded keeping it
  there because uniform weights maximize entropy.
- **`MaskedEntropyAttention` keeps a fixed entropy scale (`softplus(0.01)≈0.698`),
  not a trainable one and not full strength.** The historical trainable
  `softplus` multiplier collapsed toward zero; removing the scale entirely
  (effective 1.0) over-sparsified feature attention on daily data (mean_p / bias
  spike). The scale is now that frozen init value, preserving the weights the
  builder was tuned under.
- **`SeasonalComponentLightweight` honors `output_activation`.** It accepted and
  serialized the argument but hard-coded a linear output projection, unlike the trend,
  holiday, and regressor components.
- `AdaptiveWeightedModel` dropped its no-op `build()` override, which made tf-keras
  refuse `get_weights()` and broke `EarlyStopping(restore_best_weights=True)`.
- `python -m deepsequence_hierarchical_attention.eval.public_carparts` imports LightGBM lazily so the other models run
  without it installed.
