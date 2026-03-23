# White Paper: Operation Overwatch v9.3
**Hybrid Multi-Target Ensemble with Neural Latent Feature Extraction**

---

## BLUF (Bottom Line Up Front)

Operation Overwatch v9.3 is a production-hardened refinement of the v9.2 quantitative trading framework for the Numerai tournament. Key upgrades include a critical data integrity fix to the Denoising Autoencoder (DAE) training procedure, a corrected model object handling path for daily inference, noise layer naming accuracy, GPU memory optimization, and the addition of three institutional-grade operational modules: an Automated Weekly Scorecard, an Ablation Study Harness, and a Professional Monitoring Suite.

---

## 1. Architectural Overview

Overwatch v9.3 utilizes a heterogeneous ensemble designed to capture non-linear relationships while maintaining rigorous risk neutrality. The system blends three distinct modeling methodologies:

- **Deep Tree Ensemble:** Multi-target LightGBM and XGBoost models.
- **Neural Tactician:** A DAE-initialized Residual MLP (ResMLP).
- **Binary Alpha:** A high-regularization Ridge Classifier for tail-event detection.

Core architecture is unchanged from v9.2. All v9.3 changes are correctness fixes and operational additions.

---

## 2. Component Architecture

### 2.1 Multi-Target Tree Strategy

To increase predictive diversity and reduce variance, the tree-based models target a weighted blend of four specific v5.2 objectives:

| Target | Weight | Role |
|--------|--------|------|
| `target_ender_20` | 40% | Primary |
| `target_cyrusd_20` | 25% | Auxiliary |
| `target_teager2b_20` | 20% | Auxiliary |
| `target_victor_20` | 15% | Auxiliary |

By training on multiple horizons and styles, the ensemble gains resilience against regime shifts in the Numerai meta-market.

### 2.2 Neural Tactician (DAE v3)

The DAE provides denoised latent features for the downstream ResMLP. v9.3 corrects a critical training integrity issue present in v9.2.

**Architecture:**
- 2-layer encoder: 1024 → 64 (SiLU + BatchNorm). 64-dim bottleneck retained from v9.2 — 128-dim was found to introduce noise that degraded Neural Sharpe.
- Noise layer: `GaussianNoise` (σ = 0.1) applies additive Gaussian corruption during training to prevent identity mapping. Note: this layer was previously labeled `SwapNoise` in v9.2; the implementation was always additive Gaussian. The label has been corrected.

**v9.3 Data Integrity Fix — DAE Training Scope:**
In v9.2, the DAE was fitted on `X_all`, a concatenation of training and validation feature sets. This caused the encoder to implicitly observe validation-era data during pretraining, leaking future information into the latent representations and inflating in-sample neural metrics. In v9.3, the DAE trains exclusively on the training split (`X_train`). This restores clean separation between training and validation and ensures latent features are not contaminated by forward-looking signal.

### 2.3 Signal Restoration: Binary Alpha

Validation diagnostics revealed that PCA-transformed features lost approximately **54% of signal** (CORR drop from 0.683 to 0.310). v9.2 restored the Binary Alpha component to raw features; this is unchanged in v9.3.

- **Logic:** A Ridge Classifier (α = 10.0) identifies stocks with a high probability of exceeding the `0.5` threshold.
- **Integration:** Provides a 5% tactical tilt to the default blend and a 40% weight to the `KZ_BINARY_N20` slot.

---

## 3. Risk Management & Neutralization

The Risk Engine utilizes a 50-component PCA to identify and neutralize common factor exposures.

| Model Slot | Neutralization Ratio | Primary Blend | Target Profile |
|------------|---------------------|---------------|----------------|
| `KZ_CORE_N00` | 0% | Default | Maximum Raw Returns |
| `KZ_DEF_N90` | 90% | Default | High Sharpe / Low Drawdown |
| `KZ_BINARY_N20` | 20% | Binary | Tail-Event Capture |

---

## 4. Tactical Implementation (Workflow)

The model follows an execution schedule optimized for Google Colab (T4 GPU) with cached weights on Google Drive.

**Saturday — Full Training**
> 5-fold Purged Walk-Forward Cross-Validation. Includes DAE training (train split only), multi-target tree fitting, and risk vector generation. ~25–30 minute execution. Cells 1–8.

**Tuesday–Friday — Daily Inference**
> Frozen weights loaded from Drive. Daily live data submitted in ~2 minutes. Cells 1–5, then Cell 9.

**Sunday/Monday — Weekly Scorecard**
> Pulls `daily_model_performances` for each slot via the Numerai API, estimates market regime from top-50 leaderboard dispersion, and appends results to `Overwatch/scorecard.csv` with cumulative stats and regime tags. Cells 1–5, then Cell 11.

**Week 12 — Ablation Study**
> 6-variant component test on the same purged validation split to isolate contribution of each architectural component. Cells 1–8, then Cell 12. See Section 6.

**Weekly (alongside Scorecard) — Professional Monitoring Suite**
> 5-panel institutional diagnostic suite tracking process quality independent of tournament outcomes. Cell 13. See Section 5.

---

## 5. Monitoring & Observability

v9.3 introduces a Professional Monitoring Suite (Cell 13) providing institutional-grade process diagnostics:

| Panel | Metric | Purpose |
|-------|--------|---------|
| Predicted vs. Realized | Correlation of submissions to resolved targets | Signal validity |
| Exposure Monitor | PCA + feature-group factor loadings | Unintended bet detection |
| Turnover Tracker | Rank correlation between consecutive submissions | Signal stability |
| Drawdown Context | Current vs. bootstrapped historical drawdown distribution | Regime vs. structural break |
| Ensemble Disagreement | Component agreement score per submission | Confidence / fragility signal |

The suite operates on saved submission CSVs in Drive. First run bootstraps the drawdown distribution from validation data (~10 min additional). Subsequent runs are incremental.

Two utility functions are available at any time:
- `archive_submission()` — run after every submission to maintain the monitoring log.
- `status_dashboard()` — quick health check on demand.

---

## 6. Ablation Study Harness

At Week 12, Cell 12 runs 6 configurations against the same purged validation split to systematically measure component contributions:

| Config | Description |
|--------|-------------|
| 1 | Trees Only — LGB+XGB on `target_ender_20` |
| 2 | Trees + DAE Latent → Trees (844 features) |
| 3 | Full Ensemble minus Binary Alpha |
| 4 | Full v9.3 Baseline (control) |
| 5 | Single Target Ensemble (`ender_20` only) |
| 6 | Equal Weight Targets (25/25/25/25) |

**Output:** `Overwatch/ablation_results.csv` + summary table + inter-configuration correlation matrix.

---

## 7. Statistical Validation

| Parameter | Configuration |
|-----------|--------------|
| **Feature Set** | Medium (v5.2) |
| **CV Strategy** | Purged Walk-Forward (Purge Gap = 4 eras, 5 folds) |
| **Regularization** | Heavy ℓ₂ penalty (α = 10.0) on Binary Alpha to mitigate high dimensionality (780 raw features) |
| **DAE Training Scope** | Train split only (v9.3 integrity fix) |

---

## 8. Version History

### v9.3 (Current)
- **FIX (Critical):** DAE data leak — autoencoder now trains on train split only. v9.2 used `X_all`, leaking validation-era data into latent representations.
- **FIX:** `GaussianNoise` renamed from `SwapNoise`. Implementation was always additive Gaussian — label corrected for accuracy.
- **FIX:** `predict_multi_target` now handles both `Booster` (daily inference) and `LGBMRegressor` (training) object types correctly.
- **OPT:** Removed `t_X_all` tensor (~3GB GPU memory savings).
- **ADD:** Cell 11 — Automated Weekly Scorecard with regime tagging.
- **ADD:** Cell 12 — Ablation Study Harness (6-variant, Week 12).
- **ADD:** Cell 13 — Professional Monitoring Suite (5-panel institutional diagnostics).

### v9.2
- Multi-Target Tree Ensemble with corrected v5.2 targets.
- DAE bottleneck reverted to 64 dimensions (128-dim degraded Neural Sharpe).
- Binary Alpha restored to raw features (PCA reduced CORR from 0.683 to 0.310).
- Binary Alpha regularization increased to α = 10.0 for 780-feature raw input.

### v9.1
- Introduced 3-layer encoder and 128-dim DAE bottleneck (subsequently reverted).
- Deep tree params established (10,000 max trees, lr = 0.005, min_child = 500).

### v9.0
- Initial Combined Arms architecture: LGB + XGB + DAE ResMLP + Binary Alpha.
- 3-slot risk neutralization framework (N00 / N90 / N20).
