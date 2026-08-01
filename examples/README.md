# Examples

This folder holds **Jupyter notebooks only**. Library code and CLI entry points live in the installable package:

| Need | Location |
|------|----------|
| Feature config loader | `deepsequence_hierarchical_attention.data.feature_config_loader` |
| Holiday calendar | `deepsequence_hierarchical_attention.holidays.calendar` |
| Eval helpers / MH rollout | `deepsequence_hierarchical_attention.eval.*` |
| Adaptive train wrapper | `deepsequence_hierarchical_attention.training.adaptive_loss` |
| Weekly panel prepare | `python -m deepsequence_hierarchical_attention.data.prepare_weekly_panel` |
| Bake-offs | `python -m deepsequence_hierarchical_attention.eval.<name>` |

## Notebooks

- [`v16_deepsequence_example.ipynb`](v16_deepsequence_example.ipynb) — synthetic end-to-end demo (feature contract v1.6)
- [`reproduce_paper_findings.ipynb`](reproduce_paper_findings.ipynb) — paper tables/figures from locked artifacts (when present)

```bash
jupyter notebook examples/v16_deepsequence_example.ipynb
```
