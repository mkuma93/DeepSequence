# DeepSequence: Hierarchical Attention for Multi-Series Intermittent Demand Forecasting

**Author:** Mritunjay Kumar  
**Correspondence:** see repository  
**Status:** Methods preprint (arXiv-oriented); shaped for eventual submission to the *International Journal of Forecasting*  
**Date:** August 2026

---

> **Figures not showing in Cursor preview?** Relative image embeds often fail for Google Drive–backed workspaces (`My Drive` path with spaces). Use one of:
> 1. Open [`paper_figures/VIEW.html`](paper_figures/VIEW.html) in a browser (File → Open), or run `open paper_figures/VIEW.html` from the repo root.
> 2. Click the **Open PNG** links under each figure (or browse [`paper_figures/`](paper_figures/)).
> 3. View on GitHub: [PAPER.md](https://github.com/mkuma93/DeepSequence/blob/main/PAPER.md) · [figure gallery](https://github.com/mkuma93/DeepSequence/tree/main/paper_figures) · dedicated list in [`PAPER_figures.md`](PAPER_figures.md).


## Abstract

Intermittent demand—long runs of zeros punctuated by sparse sales—remains difficult for both classical sparse-series methods and modern global neural forecasters. Prophet-style additive structure (trend, seasonality, holidays, regressors) is widely trusted for single-series forecasting, yet does not, out of the box, yield a shared multi-series model for intermittent retail and distribution panels. We propose **DeepSequence**, a lightweight architecture that carries that structural vocabulary to the panel setting: four Prophet-like expert trunks with optional SKU personalization, hierarchical attention inside and across experts, a context-aware component mixer conditioned on lag and intermittent regime features, an occurrence–magnitude gate \(\hat{y}=p\cdot b\), and softplus-monotone maps on trend, holiday distances, and regressor channels.

Under a locked evaluation protocol on an enterprise daily intermittent panel (800 series; five training seeds), DeepSequence is **not** universally best on short-horizon intermittent weighted mean absolute error (IWMAE)—a temporal transformer often leads at horizons \(h\le 14\)—but shows a stable long-horizon advantage at \(h=28\) and \(h=60\) (DeepSequence beats the transformer on IWMAE in \(5/5\) seeds). Decision-economics scenarios that penalize loyalty / switching cost reverse LightGBM’s “under-forecast looks cheap” ranking: short lead times favor the transformer; long lead times favor DeepSequence (\(5/5\) mid-margin \(\pi\) wins at \(h=28/60\)). On public Monash Car Parts (monthly; domain mismatch), Teunter–Syntetos–Babai (TSB) remains strong at short horizons; DeepSequence leads IWMAE at \(h=6\) across five seeds, while mid-margin \(\pi\) still favors TSB. We argue for a **portfolio** view—classical intermittent methods or transformers for short horizons, structural multi-series models for long-horizon intermittent panels—rather than a universal accuracy claim.

**Keywords:** intermittent demand; multi-series forecasting; Prophet; hierarchical attention; decision economics; inventory forecasting

---

## 1. Introduction

Retail and wholesale distribution panels are often **intermittent**: most days (or months) record zero demand, with occasional positive sales of variable size. Forecast errors on quiet periods and on sale events have different operational consequences—holding cost versus lost sales and, in many retail settings, customer switching or loyalty erosion. Classical intermittent methods such as Croston’s method, Syntetos–Boylan approximation (SBA), and Teunter–Syntetos–Babai (TSB) remain strong defaults on short, sparse series. Gradient-boosted trees are competitive tabular baselines but, under absolute-error training, tend to under-forecast sale days. Global sequence models (DeepAR, temporal transformers, Temporal Fusion Transformers) add history attention and covariate handling, yet rarely encode an explicit **structural** decomposition that practitioners already trust from Prophet.

**Prophet** (Taylor and Letham, 2018) popularized additive trend, seasonality, and holiday effects for *single-series* forecasting with interpretable components. The open problem we target is not “beat every baseline on one-step IWMAE,” but:

> How can Prophet-style structural forecasting be carried into a **shared multi-series neural model** for intermittent panels—preserving component semantics, adding intermittency handling, and personalizing across SKUs?

**DeepSequence** answers with a hierarchical expert backbone that mirrors Prophet’s blocks, then adds panel-scale parameter sharing, intermittency factorization, and two levels of attention that perform **component and feature selection**—not day-level temporal self-attention over a lookback window.

**Contributions.**

1. **Hierarchical attention for Prophet-like experts.** *Level-1 (intra-expert):* seasonal masked-entropy attention over Fourier frequencies; holiday and regressor selection attention over softplus-monotone channel maps; trend uses a softplus-monotone changepoint basis with **no** within-expert attention. *Level-2 (inter-expert):* an entropy-regularized component mixer over \(\{\mathrm{trend},\mathrm{seasonal},\mathrm{holiday},\mathrm{regressor}\}\).

2. **Context-aware component mixing.** Level-2 weights are conditioned on lag and intermittent **regime** features (optionally concatenated with a SKU embedding)—not SKU identity alone—so the same calendar can reweight experts after a recent sale versus a long zero run.

3. **Occurrence–magnitude gate** \(\hat{y}=p\cdot b\) with a softplus magnitude head, separating “will demand occur?” from “how large given structural drivers?”

4. **Monotone softplus structural maps** on trend time, holiday distances, and regressor channels—neuralized Prophet-like shape constraints inside a panel model.

5. **Empirical portfolio evidence** under a locked architecture and multi-seed evaluation: long-horizon IWMAE advantage on daily \(h=28/60\) (\(5/5\) versus a temporal transformer) and monthly Car Parts \(h=6\); short horizons often favor the transformer (daily) or TSB (monthly); loyalty-aware decision economics favor DeepSequence at long daily lead times. We do **not** claim universal IWMAE leadership.

Secondary design choices—softsign-bounded expert outputs and DCN-style cross-layers off by default—are supported by ablations but are not the headline claim.

---

## 2. Related work

### 2.1 Classical intermittent demand forecasting

Croston (1972) separated intermittent demand into inter-demand intervals and demand sizes, updating each with exponential smoothing. Syntetos and Boylan (2001, 2005) analyzed bias in Croston’s estimator and proposed the Syntetos–Boylan approximation (SBA). Teunter, Syntetos, and Babai (2011) introduced TSB, which updates the probability of demand occurrence and is better suited to obsolescence and intermittent series with changing occurrence rates. These methods remain required baselines on spare-parts and other short intermittent panels (Syntetos et al., 2015; Boylan and Syntetos, 2021). Reviews of intermittent demand emphasize that accuracy metrics and inventory costs can disagree, and that zero-heavy series reward near-zero predictors under all-day MAE (Wallström and Segerstedt, 2010; Prestwich et al., 2014).

### 2.2 Metrics and decision-aware evaluation

Standard MAE and RMSE are poorly aligned with intermittent inventory risk because high zero rates dominate the loss. Intermittent-aware and scaled metrics (e.g., mean absolute scaled error variants, period-weighted errors) and inventory-oriented evaluation have been discussed extensively in the intermittent-demand literature (Syntetos and Boylan, 2005; Hyndman and Koehler, 2006; Kolassa, 2016). In this paper we use **IWMAE** (intermittent weighted MAE) as the primary accuracy metric and a transparent **decision-proxy** \(\pi\) that combines underage and holding cost proxies with an optional loyalty / switching penalty. The \(\pi\) construction is scenario analysis, not a fitted churn model or a full inventory simulator (Section 5.4).

### 2.3 Structural and Prophet-style models

Prophet (Taylor and Letham, 2018) decomposes a univariate series into piecewise trend, Fourier seasonality, and holiday effects with interpretable additive structure and Bayesian / Stan-backed estimation. Related structural time-series approaches include Bayesian structural time series (Scott and Varian, 2014) and classical unobserved-components models. DeepSequence keeps Prophet’s *block vocabulary* (trend, seasonal, holiday, regressor) but trains a **shared** neural trunk across many intermittent series, with attention *inside* and *across* blocks rather than per-series local fits.

### 2.4 Global and deep forecasting

Global forecasting—pooling strength across related series—has become standard in retail and competition settings (Januschowski et al., 2020; Montero-Manso and Hyndman, 2021). DeepAR (Salinas et al., 2020) trains an autoregressive RNN likelihood across many series. N-BEATS (Oreshkin et al., 2020) and N-HiTS (Challu et al., 2023) use deep residual stacks of fully connected blocks for univariate multi-horizon forecasting. Temporal Fusion Transformers (Lim et al., 2021) combine variable selection, gated residual networks, and multi-head attention for interpretable multi-horizon forecasting with static and known-future covariates. PatchTST and related temporal transformers (Nie et al., 2023) show that channel-independent patch attention is competitive for long-term forecasting. Comparative work on sparse retail demand often finds that shallow global models compete with heavy transformers (Makridakis et al., 2022). DeepSequence is complementary: lighter than full temporal self-attention stacks, denser in **structural** inductive bias.

### 2.5 Neural approaches to intermittent and sparse demand

Neural intermittent forecasting has explored zero-inflated and hurdle-style heads, separate occurrence and size models, and inventory-aware losses (Kourentzes, 2013; Turkmen et al., 2021); book-length treatment of intermittent methods and applications is given by Boylan and Syntetos (2021). Soft gating of the form \(\hat{y}=p\cdot b\) is closely related to classical occurrence–size factorization and to zero-inflated continuous heads. Our contribution is not intermittency gating alone, but gating combined with Prophet-like experts, hierarchical selection attention, and regime-conditioned mixing in a multi-series setting.

### 2.6 Tree ensembles and tabular baselines

LightGBM (Ke et al., 2017) and related gradient-boosted trees remain strong tabular baselines when rich covariates are available. Under L1 / MAE objectives on intermittent targets, trees often predict near zero on quiet days and under-forecast sale magnitudes—an effect that can look advantageous under holding-heavy cost proxies unless lost-sales or loyalty costs are counted (Section 6.2).

### 2.7 Positioning

| Family | Example | Fit for intermittent multi-series panels |
|--------|---------|------------------------------------------|
| Single-series structural | Prophet | Interpretable; not shared-panel by default |
| Classical intermittent | Croston, SBA, TSB | Strong short / sparse; weak rich covariates |
| Trees | LightGBM | Fast; L1 bias toward zeros / under-forecast |
| Temporal DL | DeepAR, TST, TFT | Strong history models; less Prophet-like structure |
| **This work** | **DeepSequence** | Multi-series Prophet-like experts + hierarchical attention + gate |

---

## 3. Method

### 3.1 Problem setup

For series \(i\) and time \(t\), observe demand \(y_{i,t}\ge 0\) and occurrence \(z_{i,t}=\mathbf{1}[y_{i,t}>0]\). Features \(x_{i,t}\) are **causal**: no same-day leakage from \(y_{i,t}\) into lags or intermittent state. The model predicts \(\hat{y}_{i,t}\) for inventory decisions; discrete-unit reporting uses \(\mathrm{round}(\hat{y}_{i,t})\) where noted.

### 3.2 Causal feature contract

**Daily enterprise panel.** Trend time index; day/month/year Fourier terms; lags \(1,2,7\); days since last sale, last sale quantity, lifetime cumulative sales; holiday **distances** only (no same-day holiday indicator leakage beyond known calendar).

**Monthly Car Parts.** Month index; quarter and calendar-month Fourier; lags \(1,2,12\) (monthly frequency preset / `feature_config_monthly.yaml`); months-since-last-sale state; month-has-holiday indicators (not day-distance).

Lags and intermittent features use only history with timestamp strictly less than \(t\).

### 3.3 Hierarchical experts

Inputs are routed to four experts:

| Expert | Role (Prophet analogue) | Level-1 mechanism |
|--------|-------------------------|-------------------|
| **Trend** | Piecewise / changepoint trend | Softplus-monotone piecewise-linear map in time; **no** attention (single temporal basis) |
| **Seasonal** | Fourier seasonality | Masked-entropy attention over frequency channels |
| **Holiday** | Holiday effects | Softplus-monotone map per distance channel → selection attention |
| **Regressor** | History regressors | Softplus-monotone map per lag / state channel → selection attention |

Each expert emits a scalar contribution. By default, a **softsign** output activation bounds signed expert impact. Optional SKU FiLM personalizes expert scalars; calendar FiLM on seasonal and holiday experts defaults off. Figures 1–5 walk the mechanism stack in implementation order.

### 3.4 Changepoint selection (Trend)

Trend uses `ChangepointReLU`: learnable deltas \(\delta\in\mathbb{R}^{K}\) (\(K{=}10\) default; locked runs often use \(K{=}15\)) are mapped to **ordered** locations via \(\delta^{+}=\mathrm{softplus}(\delta)\), \(\mathrm{cp}=\mathrm{cumsum}(\delta^{+})\), then rescaled into \([t_{\min},t_{\max}]\). Hinge features are \(\phi_k(t)=\mathrm{ReLU}(t-\mathrm{cp}_k)\). Locations are continuous parameters—not a discrete subset selection. Under the locked monotone path there is **no** Level-1 attention over changepoints (the legacy unconstrained path that attended over hinges is disabled).

![Figure 1. Changepoint selection.](paper_figures/fig_m1_changepoint_selection.png)

[Open PNG](paper_figures/fig_m1_changepoint_selection.png) · [GitHub](https://github.com/mkuma93/DeepSequence/blob/main/paper_figures/fig_m1_changepoint_selection.png)

*Figure 1. Ordered changepoint parameterization: softplus deltas → cumsum → scale → ReLU hinges (matches `ChangepointReLU`).*

### 3.5 Monotone softplus–PWL maps

For trend time, holiday absolute distances, and each regressor channel, hinge slopes use

\[
m = \mathrm{softplus}(s)\times \tanh(\sigma),
\]

so magnitude is nonnegative and direction is a learned sign (not a hyperparameter). Trend shares one sign across hinges; holiday and regressor learn a sign per channel. SKU FiLM uses a softplus scale so per-SKU affine personalization preserves monotonicity in the structured input for a fixed SKU.

![Figure 2. Monotone softplus-PWL maps.](paper_figures/fig_m2_monotone_softplus.png)

[Open PNG](paper_figures/fig_m2_monotone_softplus.png) · [GitHub](https://github.com/mkuma93/DeepSequence/blob/main/paper_figures/fig_m2_monotone_softplus.png)

*Figure 2. Softplus×tanh slope constraint on trend / holiday / regressor hinge maps.*

### 3.6 Level-1 selection attention

**Level-1 — intra-expert.**

- *Seasonal:* masked-entropy attention over Fourier (or learnable-frequency) channels, with entropy regularization toward sparse frequency use.
- *Holiday / regressor:* each channel is first mapped by the softplus-PWL monotone map, then aggregated by temperature-softmax **selection attention** (`MaskedEntropyAttention`) over channels. Ablating Level-1 replaces learned weights with uniform \(1/n\).
- *Trend:* deliberately without Level-1 attention—one monotone temporal basis, avoiding competing trend heads.

![Figure 3. Level-1 selection attention.](paper_figures/fig_m3_level1_attention.png)

[Open PNG](paper_figures/fig_m3_level1_attention.png) · [GitHub](https://github.com/mkuma93/DeepSequence/blob/main/paper_figures/fig_m3_level1_attention.png)

*Figure 3. Intra-expert selection attention over monotone holiday and lag channels; seasonal freq attention; trend has no Level-1.*

### 3.7 Context-aware component mixer (Level-2)

The Level-2 query is

\[
q_{i,t} = \bigl[\, e_i \,;\; \mathrm{Dense}(c_{i,t}) \,\bigr],
\]

where \(e_i\) is an optional SKU embedding and \(c_{i,t}\) are regressor-block regime signals (lags, days/months since last sale, and related intermittent state)—**not** calendar, Fourier, or holiday distances, which remain inside their experts. Temperature-softmax weights over stacked expert scalars (entropy + orthogonality regularization) yield the mixed base. This is *component* reweighting, not temporal self-attention over a lookback window. Ablating the mixer (SKU-only or stack-only Level-2) is a protocol comparison; locked runs keep the context mixer on.

![Figure 4. Context-aware component mixer.](paper_figures/fig_m4_context_mixer.png)

[Open PNG](paper_figures/fig_m4_context_mixer.png) · [GitHub](https://github.com/mkuma93/DeepSequence/blob/main/paper_figures/fig_m4_context_mixer.png)

*Figure 4. Level-2 mixer: query from SKU embedding ⊕ lag/intermittent context; softmax over expert scalars.*

### 3.8 Occurrence–magnitude gate and full stack

\[
b_{i,t}=\mathrm{softplus}\bigl(\mathrm{mix}(\mathrm{experts}(x_{i,t}), q_{i,t})\bigr),\quad
p_{i,t}=\sigma\bigl(g(x_{i,t}, e_i)\bigr),\quad
\hat{y}_{i,t}=p_{i,t}\cdot b_{i,t}.
\]

Interpretation: \(p\) is the predicted probability that demand occurs; \(b\) is the predicted magnitude given structural drivers; the product is a soft Bernoulli–magnitude expectation. Optional per-SKU zero-rate priors can bias gate logits from historical zero rates (secondary). Cross-network layers default off.

![Figure 5. End-to-end DeepSequence architecture.](paper_figures/fig_m5_architecture.png)

[Open PNG](paper_figures/fig_m5_architecture.png) · [GitHub](https://github.com/mkuma93/DeepSequence/blob/main/paper_figures/fig_m5_architecture.png)

*Figure 5. End-to-end stack: trend time index + fixed Fourier (learnable \(\omega\) optional) + holiday distances + lags/intermittent state → Trend / Seasonal / Holiday / Regressor experts (monotone maps on trend/holiday/regressor only) → Level-1 → mixer query \(q=[e_i;\mathrm{Dense}(\mathrm{lag/state})]\) → gate \(\hat{y}=p\cdot b\). Shared SKU embedding \(e_i\) (purple) fans out to FiLM, mixer, and gate.*

### 3.9 Training objective

With empirical zero rate \(\pi_0\approx P(y=0)\), the default three-term loss is

\[
\mathcal{L}
= \alpha\,\mathrm{BCE}_{w_+}(z, p)
+ w_g\,\mathrm{MAE}_{\mathrm{inv}}(y, \hat{y})
+ w_m\,\mathrm{MAE}_{\mathrm{nz}}(y, b),
\]

where \(\mathrm{MAE}_{\mathrm{inv}}\) is inverse-class-weighted all-day MAE (timing) and \(\mathrm{MAE}_{\mathrm{nz}}\) is sale-day magnitude MAE against the magnitude head. Typical weights: \(\alpha=0.2\), \(w_g=w_m=1\).

### 3.10 Multi-horizon evaluation

Primary tables use **one-step models with recursive rollout** to the horizons of interest (daily maximum horizon \(H=60\), report \(h\in\{1,7,14,28,60\}\); monthly \(h\in\{1,2,6\}\)). A direct multi-horizon head exists in the software for planning; it is **not** the primary claim of this preprint. Earlier drafts that emphasized direct multi-horizon wins at short horizons under a previous protocol are relegated to Appendix D and are not restated as primary evidence.

---

## 4. Experimental design

### 4.1 Datasets

| Panel | Grain | Series | Approx. train zeros | Role |
|-------|-------|-------:|--------------------:|------|
| Proprietary enterprise | Daily | 800 (SKU list locked) | ≈90% | Primary daily evidence |
| Monash Car Parts (Godahewa et al., 2021) | Monthly | 800 (same lock convention) | ≈74% | Public domain-mismatch sanity check |

The enterprise panel cannot be released. Code, feature contracts, a synthetic demo, and public Car Parts adapters are in the accompanying repository (Appendix A).

### 4.2 Locked protocol

| Item | Specification |
|------|----------------|
| Architecture stack | Softsign expert outputs; monotone Level-1 maps; context-aware mixer; calendar FiLM off; cross-network layers **off** |
| Features | Identical causal feature matrix for trees and neural models |
| Sequence lookback | 14 (daily) / 12 (monthly) for DeepAR, temporal transformer, and TFT baselines |
| Seeds | SKU panels locked once. Seed-42 full-baseline tables in Section 6.1–6.3; multi-seed means \(\pm\) standard deviation over training seeds \(\{42,\ldots,46\}\) |
| Metrics | IWMAE (primary accuracy); occurrence F1; underforecast on sales; bias; decision \(\pi\) with loyalty scenarios |
| Baselines | LightGBM; DeepAR-lite; temporal transformer (TST); TFT-lite; Croston / SBA / TSB on Car Parts; **Prophet (per-series)** |

**Why not all-day MAE alone.** High zero rates reward near-zero predictors. IWMAE and sale-day underforecast better reflect intermittent inventory risk.

### 4.3 Prophet baseline

If DeepSequence is framed as a multi-series extension of Prophet-style decomposition, experiments must include Prophet itself:

| Item | Specification |
|------|----------------|
| Protocol | One Prophet model **per series** (no SKU pooling) |
| Fit window | Train + validation history; forecast test origins |
| Car Parts | 800 locked SKUs; fixed origin → \(h\in\{1,2,6\}\) |
| Daily | 150-SKU evenly spaced subset of the locked list; at most four origins per SKU; \(h\in\{1,28,60\}\) |
| Features | Calendar seasonality only (yearly; weekly on daily). Holiday distances and intermittent lags are **not** injected as Prophet regressors |
| Honest limit | Prophet = local structural baseline; DeepSequence / LightGBM / TSB = global or classical intermittent |

A full 800-SKU daily Prophet bake-off remains future work (Section 8). Daily Prophet numbers are therefore **not** comparable to the locked 800-SKU global tables.

### 4.4 Novelty ablations

On the locked daily panel, training seed 42, vary one factor at a time from the full stack (Level-1 selection attention on, monotone maps on, context mixer on, gate on, cross-layers off):

| Arm | Change |
|-----|--------|
| Full | All novelty ingredients on; cross-layers off |
| −context mixer | Level-2 without regime context |
| −Level-1 selection attn | Uniform \(1/n\) aggregation over monotone channels |
| −mono | Unconstrained (non-monotone) expert maps |
| −gate | Magnitude-only head (no occurrence factor) |
| +cross | Cross-network layers on |

Ablations are **single-seed** and do not duplicate the multi-seed loyalty orchestrator.

### 4.5 Decision economics (scenario analysis)

\[
\pi \approx \mathrm{revenue\_proxy}
- \underbrace{(m\cdot \mathrm{price}+C_{\mathrm{loyalty}})\,U + C_{\mathrm{hold}}\,H}_{\mathrm{inventory\ loss\ proxy}}
- C_{\mathrm{model}},
\]

with holding cost \(C_{\mathrm{hold}}=0.1\), margins \(m\in\{0.08,0.25,0.55\}\), and loyalty / switching costs \(C_{\mathrm{loyalty}}\in\{0,0.25,0.5\}\). Default reporting uses \(C_{\mathrm{loyalty}}=0.25\) alongside the legacy \(C_{\mathrm{loyalty}}=0\) case. Here \(U\) and \(H\) are forecast-error underage and holding *proxies*, not outputs of a full inventory pipeline; \(C_{\mathrm{loyalty}}\) is **not** estimated from churn data.

---

## 5. Results

### 5.1 Daily multi-horizon IWMAE

Recursive rollout after origin \(t\); known-future calendar and holidays; demand predictions fed into lags and intermittent state. Seed-42 full bake-off: \(n=5538\) origins, 800 SKUs.

**Table 1.** Daily recursive IWMAE, seed 42 (locked stack). Bold = best in row.

| Horizon | DeepSequence | TST | TFT | DeepAR | LightGBM | Best |
|--------:|-------------:|----:|----:|-------:|---------:|:-----|
| \(h=1\) | 4.035 | **3.860** | 4.010 | 4.154 | 4.451 | TST |
| \(h=7\) | 4.381 | **4.323** | 4.566 | 5.169 | 4.688 | TST |
| \(h=14\) | 4.211 | **4.177** | 4.589 | 5.006 | 4.615 | TST |
| \(h=28\) | **6.417** | 6.877 | 6.891 | 7.212 | 6.866 | **DS** |
| \(h=60\) | **3.891** | 4.495 | 4.308 | 4.696 | 4.375 | **DS** |
| mean \(1..60\) | **4.374** | 4.666 | 4.736 | 5.103 | 4.790 | **DS** |

**Table 2.** Daily multi-seed IWMAE (mean \(\pm\) std over training seeds \(42\)–\(46\); locked SKU panel; DeepSequence / TST / LightGBM).

| Horizon | DeepSequence | TST | LightGBM |
|--------:|-------------:|----:|---------:|
| \(h=1\) | \(4.023\pm0.257\) | \(\mathbf{3.912\pm0.273}\) | \(4.483\pm0.324\) |
| \(h=7\) | \(4.299\pm0.500\) | \(\mathbf{4.139\pm0.523}\) | \(4.549\pm0.531\) |
| \(h=14\) | \(5.494\pm0.935\) | \(\mathbf{5.454\pm0.890}\) | \(5.828\pm0.883\) |
| \(h=28\) | \(\mathbf{4.823\pm0.970}\) | \(5.290\pm0.991\) | \(5.194\pm1.031\) |
| \(h=60\) | \(\mathbf{4.345\pm0.619}\) | \(5.055\pm0.417\) | \(4.710\pm0.550\) |

**Reading.** Short horizons favor the **temporal transformer**; DeepSequence leads at **long horizons** (\(h=28/60\)) on both the seed-42 full bake-off and the five-seed mean. DeepSequence beats TST IWMAE at \(h=28\) and \(h=60\) in **\(5/5\)** seeds. This is the opposite of a claim that DeepSequence wins one-step IWMAE everywhere. Figure 6 plots the multi-seed IWMAE curves from Table 2.

![Figure 6. Daily multi-seed IWMAE vs horizon.](paper_figures/fig1_daily_iwmae_horizon.png)

[Open PNG](paper_figures/fig1_daily_iwmae_horizon.png) · [GitHub](https://github.com/mkuma93/DeepSequence/blob/main/paper_figures/fig1_daily_iwmae_horizon.png)

*Figure 6. Daily recursive IWMAE (mean ± std over training seeds \(42\)–\(46\)) for DeepSequence, TST, and LightGBM at \(h\in\{1,7,14,28,60\}\). Short horizons favor TST; DeepSequence leads at \(h=28/60\).*

### 5.2 Decision economics with loyalty (daily)

Without loyalty (\(C_{\mathrm{loyalty}}=0\)), **LightGBM** often wins **low-margin** \(\pi\): under-forecasting reduces holding \(H\) and looks cheap when lost sales are under-weighted. With the recommended scenario \(C_{\mathrm{loyalty}}=0.25\), that ranking flips.

**Table 3.** Seed-42 \(\pi\) winners by lead time (proxy = forecast horizon) and loyalty (low / mid / high margin).

| Lead time | \(C_{\mathrm{loyalty}}=0\) | \(C_{\mathrm{loyalty}}=0.25\) |
|-----------|--------------------------|-------------------------------|
| 7 days | LGBM / TST / TST | **TST / TST / TST** |
| 14 days | LGBM / DS / TST | DS / TST / TST |
| 28 days | LGBM / DS / DS | **DS / DS / DS** |
| 60 days | LGBM / DS / DS | **DS / DS / DS** |

**Table 4.** Multi-seed mid-margin \(\pi\) at \(C_{\mathrm{loyalty}}=0.25\) (seeds \(42\)–\(46\); higher is better).

| Horizon | DeepSequence | TST | LightGBM | Mid-\(\pi\) winner |
|--------:|-------------:|----:|---------:|:-------------------|
| \(h=7\) | \(-0.243\pm0.022\) | \(\mathbf{-0.218\pm0.024}\) | \(-0.262\pm0.023\) | TST \(4/5\) |
| \(h=14\) | \(-0.291\pm0.046\) | \(-0.291\pm0.040\) | \(-0.313\pm0.034\) | TST \(3/5\), DS \(2/5\) |
| \(h=28\) | \(\mathbf{-0.273\pm0.045}\) | \(-0.353\pm0.068\) | \(-0.308\pm0.051\) | **DS \(5/5\)** |
| \(h=60\) | \(\mathbf{-0.255\pm0.042}\) | \(-0.373\pm0.046\) | \(-0.301\pm0.031\) | **DS \(5/5\)** |

Loyalty collapses LightGBM’s low-margin win rate (\(h=7/14\): \(5/5\to0/5\); \(h=28\): \(5/5\to2/5\); \(h=60\): \(5/5\to1/5\)). Figure 7 shows the multi-seed mid-margin \(\pi\) curves from Table 4.

![Figure 7. Daily multi-seed mid-margin pi vs horizon.](paper_figures/fig3_daily_decision_pi_horizon.png)

[Open PNG](paper_figures/fig3_daily_decision_pi_horizon.png) · [GitHub](https://github.com/mkuma93/DeepSequence/blob/main/paper_figures/fig3_daily_decision_pi_horizon.png)

*Figure 7. Daily multi-seed mid-margin decision \(\pi\) at \(C_{\mathrm{loyalty}}=0.25\) (higher is better; mean ± std over seeds \(42\)–\(46\)). TST leads at short lead times; DeepSequence dominates \(h=28/60\).*

**Portfolio takeaway.** Short replenishment → temporal transformer; long replenishment → DeepSequence—once a modest loyalty / switching cost prevents “always under-forecast” from winning on paper. The multi-seed mid-\(\pi\) pattern matches the seed-42 matrix.

### 5.3 Public Car Parts (monthly; domain mismatch)

**Table 5.** Car Parts all-model bake-off, seed 42 (locked stack) plus per-series Prophet.

| Horizon | TSB | DeepSequence | Prophet | TST | LightGBM | Best |
|--------:|----:|-------------:|--------:|----:|---------:|:-----|
| \(h=1\) | **0.850** | 0.882 | 0.916 | 0.887 | 0.889 | TSB |
| \(h=2\) | **0.767** | 0.778 | 0.836 | 0.789 | 0.832 | TSB |
| \(h=6\) | 0.834 | 0.834 | **0.827** | 0.866 | 0.890 | Prophet (≈DS/TSB) |

Prophet is competitive at \(h=6\) but loses short horizons to TSB and DeepSequence—as expected for a local additive model on short intermittent monthly history without shared pooling.

**Table 6.** Car Parts multi-seed IWMAE (seeds \(42\)–\(46\); DeepSequence / TSB / LightGBM).

| Horizon | DeepSequence | TSB | LightGBM |
|--------:|-------------:|----:|---------:|
| \(h=1\) | \(0.842\pm0.012\) | \(\mathbf{0.815\pm0}\) | \(0.874\pm0.005\) |
| \(h=2\) | \(0.733\pm0.009\) | \(\mathbf{0.703\pm0}\) | \(0.838\pm0.005\) |
| \(h=6\) | \(\mathbf{0.769\pm0.004}\) | \(0.787\pm0\) | \(0.877\pm0.012\) |

**Protocol note.** Table 5 (seed-42 bake-off; DeepSequence/TSB \(h=6\) ≈ 0.834) and Table 6 (multi-seed; DeepSequence \(0.769\pm0.004\) vs TSB 0.787) use **different reporting conventions** (raw IWMAE versus primary rounded IWMAE in the multi-seed orchestrator). Rankings within each table are self-consistent; do not mix absolute levels across tables.

TSB is seed-invariant (classical). DeepSequence’s long-horizon (\(h=6\)) IWMAE edge over TSB is stable across seeds on the multi-seed (rounded) table, but mid-margin \(\pi\) with \(C_{\mathrm{loyalty}}=0.25\) still favors **TSB** on all horizons (\(0/5\) DeepSequence mid-\(\pi\) wins at \(h=6\))—reinforcing domain mismatch versus covariate-rich daily retail. Prefer TSB (then SBA/Croston) as the short-horizon monthly default; treat DeepSequence as a structural long-horizon IWMAE competitor when a neural panel model is required. Prophet confirms that a structural baseline is present but does not overturn TSB on this panel. Figure 8 summarizes Tables 5–6.

![Figure 8. Car Parts IWMAE vs horizon.](paper_figures/fig2_carparts_iwmae_horizon.png)

[Open PNG](paper_figures/fig2_carparts_iwmae_horizon.png) · [GitHub](https://github.com/mkuma93/DeepSequence/blob/main/paper_figures/fig2_carparts_iwmae_horizon.png)

*Figure 8. Left: Car Parts multi-seed IWMAE (mean ± std; DeepSequence / TSB / LightGBM). Right: seed-42 bake-off including per-series Prophet. Do not mix absolute levels across panels (Section 5.3 protocol note).*

### 5.4 Daily Prophet subset (protocol note)

On a **150-SKU** evenly spaced subset of the locked daily list (at most four origins per SKU), per-series Prophet reports IWMAE \(h=1/28/60\) = **2.68 / 4.90 / 3.34**. These are **not** comparable to the 800-SKU global DeepSequence table (different panel slice and origin density). The run documents that a tractable daily Prophet protocol is available; a full 800-SKU daily Prophet bake-off remains future work.

### 5.5 Novelty ablations

**Table 7.** Daily one-step IWMAE, tabular DeepSequence-only, seed 42. Gate is the dominant intermittent novelty at one step; other factors sit inside single-seed noise.

| Arm | IWMAE | \(\Delta\) vs Full |
|-----|------:|-------------------:|
| Full (Level-1 + mixer + mono + gate; cross off) | 4.156 | — |
| −context mixer | 4.081 | −0.075 |
| −Level-1 selection attn | 4.113 | −0.043 |
| −mono | 4.191 | +0.036 |
| −gate | **4.578** | **+0.422** |
| +cross | 4.097 | −0.059 |

**Table 8.** Daily recursive multi-horizon IWMAE, DeepSequence-only, seed 42. Long horizons isolate the claimed novelties; Full wins \(h=28/60\).

| Arm | \(h=1\) | \(h=28\) | \(h=60\) |
|-----|--------:|---------:|---------:|
| **Full** | 4.088 | **6.451** | **3.930** |
| −context mixer | **4.012** | 6.556 | 4.014 |
| −Level-1 selection attn | 4.142 | 6.657 | 4.113 |
| −mono | 4.046 | 6.535 | 3.999 |
| +cross | 4.049 | 6.642 | 4.137 |

**What each novelty buys (long horizons, single seed).**

| Novelty | Evidence |
|---------|----------|
| Level-1 selection attention | Removing it raises \(h=28/60\) by ≈0.21 / 0.18 |
| Context mixer | Removing it raises \(h=28/60\) by ≈0.11 / 0.08 (helps long; short \(h=1\) can look better without it) |
| Softplus mono maps | Removing them raises \(h=28/60\) by ≈0.08 / 0.07 |
| Occurrence gate | One-step \(\Delta\) ≈ +0.42 IWMAE when removed |
| Cross-network layers | Enabling them hurts long \(h\); keep off |

Ablations are single-seed; they motivate the locked stack but do not replace multi-seed Tables 2 and 4. Figure 9 visualizes Tables 7–8 (gate at one step; long-horizon Full vs structural ablations).

![Figure 9. Novelty ablation IWMAE.](paper_figures/fig4_novelty_ablation.png)

[Open PNG](paper_figures/fig4_novelty_ablation.png) · [GitHub](https://github.com/mkuma93/DeepSequence/blob/main/paper_figures/fig4_novelty_ablation.png)

*Figure 9. Left: one-step DeepSequence novelty ablation (seed 42); −gate dominates the IWMAE gap. Right: recursive \(h=28/60\) ablations (gate omitted; MH arms without gate). Full wins both long horizons.*

### 5.6 Qualitative forecasts

To illustrate intermittent shape (long zero runs, sparse spikes), we dump actual vs forecast traces for three locked intermittent SKUs on each panel after a dedicated small-panel retrain (seed 42; 40 SKUs drawn from the locked lists with the plot SKUs forced in; 8 epochs). These figures are **qualitative**—not a substitute for the locked 800-SKU tables above.

**Daily (DeepSequence vs TST).** Figure 10 shows one-step test-window forecasts; Figure 11 shows recursive rollout from a locked test origin for short (\(h{=}1..7\)) and longer (\(h{=}1..28\)) horizons.

![Figure 10. Daily one-step forecasts (intermittent SKUs).](paper_figures/fig_forecast_daily_onestep.png)

[Open PNG](paper_figures/fig_forecast_daily_onestep.png) · [GitHub](https://github.com/mkuma93/DeepSequence/blob/main/paper_figures/fig_forecast_daily_onestep.png)

*Figure 10. Daily enterprise panel: actual vs DeepSequence vs TST on the test window for three intermittent locked SKUs.*

![Figure 11. Daily recursive forecasts (short and long).](paper_figures/fig_forecast_daily_recursive.png)

[Open PNG](paper_figures/fig_forecast_daily_recursive.png) · [GitHub](https://github.com/mkuma93/DeepSequence/blob/main/paper_figures/fig_forecast_daily_recursive.png)

*Figure 11. Daily recursive forecasts from one test origin per SKU (\(h{=}1..7\) and \(h{=}1..28\)).*

**Car Parts (DeepSequence vs TSB).** Figure 12 shows one-step monthly test forecasts; Figure 13 shows recursive \(h{=}1..2\) and \(h{=}1..6\) from the pre-test origin (same origin convention as the Car Parts MH bake-off).

![Figure 12. Car Parts one-step forecasts.](paper_figures/fig_forecast_carparts_onestep.png)

[Open PNG](paper_figures/fig_forecast_carparts_onestep.png) · [GitHub](https://github.com/mkuma93/DeepSequence/blob/main/paper_figures/fig_forecast_carparts_onestep.png)

*Figure 12. Monash Car Parts: actual vs DeepSequence vs TSB on the six-month test window.*

![Figure 13. Car Parts recursive forecasts.](paper_figures/fig_forecast_carparts_recursive.png)

[Open PNG](paper_figures/fig_forecast_carparts_recursive.png) · [GitHub](https://github.com/mkuma93/DeepSequence/blob/main/paper_figures/fig_forecast_carparts_recursive.png)

*Figure 13. Car Parts recursive forecasts from the pre-test origin (\(h{=}1..2\) and \(h{=}1..6\)).*

---

## 6. Discussion

**Framing.** DeepSequence is a multi-series extension of Prophet-style decomposition for intermittent panels, with hierarchical attention, regime-aware mixing, gating, and monotone maps as the architectural payload—not a claim that it is first on IWMAE at every horizon. The per-series Prophet control (Section 5.3–5.4) makes that framing falsifiable.

**When to use what (portfolio).**

| Setting | Prefer |
|---------|--------|
| Daily, short lead time / short \(h\) | Temporal transformer (accuracy); check loyalty \(\pi\) |
| Daily, long lead time (\(h\gtrsim 28\)) | **DeepSequence** (IWMAE + loyalty \(\pi\)) |
| Monthly short spare-parts, weak covariates | **TSB** (then SBA/Croston); Prophet alone is not enough |
| Monthly longer horizon (\(h=6\)) accuracy | DeepSequence competitive / best IWMAE; \(\pi\) may still favor TSB |
| Ranking protocol | IWMAE + underforecast + loyalty-aware \(\pi\); do not rank on all-day MAE alone |

**Why loyalty matters.** LightGBM’s low holding from under-forecasting wins low-margin \(\pi\) when \(C_{\mathrm{loyalty}}=0\). A modest switching cost restores the cost of missed demand; long lead-time daily \(\pi\) then aligns with DeepSequence’s long-horizon accuracy—stable across five training seeds on the daily panel. On Car Parts, loyalty does not overturn TSB’s mid-\(\pi\) dominance.

**Structural inductive bias.** The long-horizon pattern is consistent with the hypothesis that explicit trend / seasonality / holiday / regressor structure helps when recursive rollouts compound, whereas short-horizon intermittent accuracy often reduces to recent occurrence dynamics that classical methods and temporal transformers already capture well.

---

## 7. Limitations and future work

**Limitations.** Enterprise results are panel-specific and cannot be released as raw data. Novelty ablations are single-seed. Daily Prophet is a 150-SKU subset, not the locked 800-SKU panel. Sequence baselines are lite adaptations sharing the gated head where applicable. Car Parts is short monthly history without a rich retail calendar. Decision economics use error proxies, not full inventory simulation. Hierarchical product-tree reconciliation is out of scope. Prophet versus DeepSequence also differs in protocol (local per-series fit versus global multi-series training). Seed-42 bake-off tables and multi-seed summary tables may use different IWMAE field conventions on Car Parts (Section 5.3); rankings should be read within table.

**Future work.** (i) Fuller daily Prophet on the locked 800-SKU panel under comparable origin density. (ii) Optional monthly evaluation at \(h=12\) where series length permits. (iii) Multi-seed novelty ablations. (iv) Stronger inventory simulation and empirically calibrated loyalty costs. (v) Hierarchical reconciliation across product trees.

---

## 8. Conclusion

We reframed intermittent neural forecasting as **Prophet-style structure at panel scale**. DeepSequence’s first-class architectural contributions are hierarchical attention inside and across Prophet-like experts, a context-aware component mixer, an occurrence–magnitude gate \(\hat{y}=p\cdot b\), and monotone softplus structural maps.

Empirically, under locked defaults and five training seeds on a locked SKU panel, DeepSequence shows a **stable long-horizon** accuracy role (daily \(h=28/60\); Car Parts \(h=6\) IWMAE) and a **loyalty-aware** decision role at long daily lead times, while short horizons often favor a temporal transformer or TSB—and Car Parts mid-\(\pi\) remains TSB. Softsign expert outputs and cross-layers off are supporting defaults, not the headline claim. We recommend a **portfolio** deployment story and intermittent metrics that do not let under-forecasting win by default.

---

## References

Boylan, J. E., & Syntetos, A. A. (2021). *Intermittent Demand Forecasting: Context, Methods and Applications*. Wiley.

Challu, C., Olivares, K. G., Oreshkin, B. N., Garza, F., Mergenthaler-Canseco, M., & Dubrawski, A. (2023). N-HiTS: Neural hierarchical interpolation for time series forecasting. *Proceedings of the AAAI Conference on Artificial Intelligence*, 37(6), 6989–6997.

Croston, J. D. (1972). Forecasting and stock control for intermittent demands. *Operational Research Quarterly*, 23(3), 289–303.

Godahewa, R., Bergmeir, C., Webb, G. I., Hyndman, R. J., & Montero-Manso, P. (2021). Monash Time Series Forecasting Archive. *NeurIPS Datasets and Benchmarks*. (Car Parts: Zenodo 4656021.)

Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679–688.

Januschowski, T., Gasthaus, J., Wang, Y., Salinas, D., Flunkert, V., Bohlke-Schneider, M., & Callot, L. (2020). Criteria for classifying forecasting methods. *International Journal of Forecasting*, 36(1), 167–177.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30.

Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-task learning using uncertainty to weigh losses. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 7482–7491.

Kolassa, S. (2016). Evaluating predictive count data distributions in retail sales forecasting. *International Journal of Forecasting*, 32(3), 788–803.

Kourentzes, N. (2013). Intermittent demand forecasts with neural networks. *International Journal of Production Economics*, 143(1), 198–206.

Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. *International Journal of Forecasting*, 37(4), 1748–1764.

Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). M5 accuracy competition: Results, findings, and conclusions. *International Journal of Forecasting*, 38(4), 1346–1364.

Montero-Manso, P., & Hyndman, R. J. (2021). Principles and algorithms for forecasting groups of time series: Locality and globality. *International Journal of Forecasting*, 37(4), 1632–1653.

Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A time series is worth 64 words: Long-term forecasting with transformers. *International Conference on Learning Representations (ICLR)*.

Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2020). N-BEATS: Neural basis expansion analysis for interpretable time series forecasting. *International Conference on Learning Representations (ICLR)*.

Prestwich, S., Rossi, R., Armagan Tarim, S., & Hnich, B. (2014). Mean-based error measures for intermittent demand forecasting. *International Journal of Production Research*, 52(22), 6782–6791.

Salinas, D., Flunkert, V., Gasthaus, J., & Januschowski, T. (2020). DeepAR: Probabilistic forecasting with autoregressive recurrent networks. *International Journal of Forecasting*, 36(3), 1181–1191.

Scott, S. L., & Varian, H. R. (2014). Predicting the present with Bayesian structural time series. *International Journal of Mathematical Modelling and Numerical Optimisation*, 5(1–2), 4–23.

Syntetos, A. A., & Boylan, J. E. (2001). On the bias of intermittent demand estimates. *International Journal of Production Economics*, 71(1–3), 457–466.

Syntetos, A. A., & Boylan, J. E. (2005). The accuracy of intermittent demand estimates. *International Journal of Forecasting*, 21(2), 303–314.

Syntetos, A. A., Babai, Z., & Gardner, E. S., Jr. (2015). Forecasting intermittent inventory demands: Simple parametric methods vs. bootstrapping. *Journal of Business Research*, 68(8), 1746–1752.

Taylor, S. J., & Letham, B. (2018). Forecasting at scale. *The American Statistician*, 72(1), 37–45.

Teunter, R. H., Syntetos, A. A., & Babai, M. Z. (2011). Intermittent demand: Linking forecasting to inventory obsolescence. *European Journal of Operational Research*, 214(3), 606–615.

Turkmen, A. C., Januschowski, T., Wang, Y., & Smola, A. J. (2021). Forecasting intermittent and sparse time series: A unified generative modeling approach. *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(11), 9905–9913.

Wallström, P., & Segerstedt, A. (2010). Evaluation of forecasting error measurements and techniques for intermittent demand. *International Journal of Production Economics*, 128(2), 625–636.

---

## Appendix A. Software and reproducibility

Implementation and locked evaluation artifacts accompany this preprint:

| Artifact | Location |
|----------|----------|
| Package | `deepsequence_hierarchical_attention/` |
| Feature configuration | `feature_config.yaml` |
| Synthetic demo | `examples/v16_deepsequence_example.ipynb` |
| Training config sample | `examples/training_config.sample.json` |
| Locked daily multi-horizon (all models) | `ab_runs/reclaim/daily_mh_1_60_level1_cross_off_all_models.json` |
| Prophet Car Parts (monthly) | `ab_runs/reclaim/prophet_carparts/carparts_mh_1_2_6.json` |
| Prophet daily subset | `ab_runs/reclaim/prophet_daily/daily_subset150_h1_28_60.json` |
| Novelty ablations | `ab_runs/reclaim/ablate_novelty/` |
| Daily loyalty economics | `ab_runs/reclaim/daily_decision_economics_level1_cross_off_loyalty.json` |
| Car Parts multi-horizon (+ LightGBM) | `ab_runs/reclaim/carparts_mh_1_2_6_level1_cross_off_lgbm.json` |
| Car Parts loyalty economics | `ab_runs/reclaim/carparts_decision_economics_level1_cross_off_loyalty.json` |
| Daily multi-seed summary | `ab_runs/reclaim/multiseed/daily_multiseed_long_loyalty_summary.json` |
| Car Parts multi-seed summary | `ab_runs/reclaim/multiseed/carparts_multiseed_long_loyalty_summary.json` |
| Figures | `paper_figures/` |

Repository: [https://github.com/mkuma93/DeepSequence](https://github.com/mkuma93/DeepSequence)

**Reproduction sketch.**

```bash
pip install -e ".[dev]"
export DEEPSEQUENCE_DATA_DIR=/path/to/local/panel

# Locked daily recursive multi-horizon
python examples/eval_multihorizon_compare.py \
  --data_dir "$DEEPSEQUENCE_DATA_DIR" \
  --max_skus 800 --epochs 10 --seed 42 --horizon 60

# Public Car Parts
python examples/public_data/prepare_carparts.py
python examples/eval_public_carparts_mh_all.py --max_skus 800 --epochs 10 --seed 42
```

Exact locked-stack settings (softsign outputs, monotone maps, context mixer, cross-layers off) match the `ds_stack` fields in the JSON artifacts above.

---

## Appendix B. Notation

| Symbol | Meaning |
|--------|---------|
| \(y\) | Demand |
| \(z=\mathbf{1}[y>0]\) | Occurrence |
| \(b\) | Magnitude (softplus) |
| \(p\) | Occurrence probability |
| \(\hat{y}=p\cdot b\) | Final forecast |
| \(e_i\) | SKU embedding (optional) |
| \(c_{i,t}\) | Lag / intermittent context for Level-2 mixer |
| \(U, H\) | Underage / holding proxies in \(\pi\) |
| \(C_{\mathrm{loyalty}}\) | Scenario switching / loyalty cost |
| \(h\) | Forecast horizon (recursive rollout steps) |

---

## Appendix C. Figure index

For reliable local viewing when markdown preview fails: open [`paper_figures/VIEW.html`](paper_figures/VIEW.html) or see [`PAPER_figures.md`](PAPER_figures.md).

| File | Role in this preprint |
|------|------------------------|
| `paper_figures/VIEW.html` | Browser gallery of Figures 1–13 (recommended local viewer) |
| `PAPER_figures.md` | Markdown figure gallery with open / GitHub links |
| `paper_figures/fig_m1_changepoint_selection.png` | Figure 1 — changepoint selection (`ChangepointReLU`) |
| `paper_figures/fig_m2_monotone_softplus.png` | Figure 2 — softplus-PWL monotone maps |
| `paper_figures/fig_m3_level1_attention.png` | Figure 3 — Level-1 selection attention |
| `paper_figures/fig_m4_context_mixer.png` | Figure 4 — context-aware Level-2 mixer |
| `paper_figures/fig_m5_architecture.png` | Figure 5 — end-to-end architecture (code-faithful labels; shared \(e_i\); \(\hat{y}=p\cdot b\)) |
| `paper_figures/fig1_daily_iwmae_horizon.png` | Figure 6 — daily multi-seed IWMAE vs horizon |
| `paper_figures/fig3_daily_decision_pi_horizon.png` | Figure 7 — daily multi-seed mid-\(\pi\) vs horizon |
| `paper_figures/fig2_carparts_iwmae_horizon.png` | Figure 8 — Car Parts IWMAE (+ Prophet bake-off panel) |
| `paper_figures/fig4_novelty_ablation.png` | Figure 9 — novelty ablations |
| `paper_figures/fig_forecast_daily_onestep.png` | Figure 10 — daily qualitative one-step forecasts |
| `paper_figures/fig_forecast_daily_recursive.png` | Figure 11 — daily qualitative recursive forecasts |
| `paper_figures/fig_forecast_carparts_onestep.png` | Figure 12 — Car Parts qualitative one-step forecasts |
| `paper_figures/fig_forecast_carparts_recursive.png` | Figure 13 — Car Parts qualitative recursive forecasts |
| `paper_figures/make_method_diagrams.py` | Regenerates Figures 1–5 |
| `paper_figures/make_results_figures.py` | Regenerates Figures 6–9 from locked JSON |
| `paper_figures/make_forecast_line_plots.py` | Regenerates Figures 10–13 (+ JSON dumps) |
| `paper_figures/fig_architecture_ds.png` | Alias of Figure 5 |
| `paper_figures/fig_hierarchical_attention_internals.png` | Prior combined L1/L2 schematic |
| `paper_figures/fig_changepoint_monotone.png` | Prior combined changepoint/mono schematic |
| `paper_figures/fig_decision_economics_by_lead_time.png` | Prior / partial economics (no loyalty lock) |
| `paper_figures/fig_decision_economics_cost_vs_r.png` | Prior economics vs critical ratio |
| `paper_figures/fig7_public_carparts_iwmae.png` | Prior one-step Car Parts figure |
| `paper_figures/fig0_architecture.png` … `fig6_*.png` | Prior-protocol figures (Appendix D) |

---

## Appendix D. Prior protocol (not primary claims)

Earlier drafts emphasized **one-step DeepSequence IWMAE ≈ 4.004 as a headline #1 result** and **direct multi-horizon DeepSequence best at \(h=7/14\)** under a previous feature and model protocol. Those results remain in repository JSON artifacts and older `paper_figures/fig1`–`fig6` plots but are **not** primary claims of this preprint. The locked hierarchical-attention / cross-off recursive tables in Section 5 supersede them for the multi-series Prophet + portfolio narrative.

---

## Citation notes (author check before arXiv)

Classic references above (Croston, 1972; Syntetos and Boylan, 2001/2005; Teunter et al., 2011; Taylor and Letham, 2018; Salinas et al., 2020; Lim et al., 2021; Oreshkin et al., 2020; Challu et al., 2023; Nie et al., 2023; Ke et al., 2017; Godahewa et al., 2021; Boylan and Syntetos, 2021) are standard. Before arXiv / IJF submission, verify volume/page details against publisher records and convert to IJF house style. Optional additions an author may wish to insert after local library check: further intermittent neural surveys post-2021, and any preferred citation for IWMAE as used in the accompanying codebase if a prior published definition exists.
