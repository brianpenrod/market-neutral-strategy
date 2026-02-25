# White Paper: Operation Overwatch v9.2

**Hybrid Multi-Target Ensemble with Neural Latent Feature Extraction**

---

## BLUF (Bottom Line Up Front)

Operation Overwatch v9.2 represents a tactical refinement of the v9.1 quantitative trading framework for the Numerai tournament. Key upgrades include a Multi-Target Tree Ensemble using corrected v5.2 targets, a re-optimized Denoising Autoencoder (DAE) bottleneck for enhanced Neural Sharpe, and the restoration of Binary Alpha signal integrity by reverting to raw feature inputs.

---

## 1. Architectural Overview

Overwatch v9.2 utilizes a heterogeneous ensemble designed to capture non-linear relationships while maintaining rigorous risk neutrality. The system blends three distinct modeling methodologies:

- **Deep Tree Ensemble:** Multi-target LightGBM and XGBoost models.
- **Neural Tactician:** A DAE-initialized Residual MLP (ResMLP).
- **Binary Alpha:** A high-regularization Ridge Classifier for tail-event detection.

---

## 2. Key Enhancements & Component Analysis

### 2.1 Multi-Target Tree Strategy

To increase predictive diversity and reduce variance, the tree-based models now target a weighted blend of four specific v5.2 objectives:

| Target | Weight | Role |
|--------|--------|------|
| `target_ender_20` | 40% | Primary |
| `target_cyrusd_20` | 25% | Auxiliary |
| `target_teager2b_20` | 20% | Auxiliary |
| `target_victor_20` | 15% | Auxiliary |

By training on multiple horizons and styles, the ensemble gains resilience against regime shifts in the Numerai meta-market.

### 2.2 Neural Refinement (DAE v2)

Data-driven testing in v9.1 indicated that a 128-dimension latent space was "too wide," introducing noise that degraded the Sharpe ratio. v9.2 reverts the bottleneck to **64 dimensions**.

- **Architecture:** 2-layer encoder (1024 → 64) using SiLU activation and Batch Normalization.
- **Objective:** Reconstruction of the "medium" feature set to extract a compressed, denoised signal for the downstream ResMLP.

### 2.3 Signal Restoration: Binary Alpha

Validation diagnostics revealed that PCA-transformed features lost approximately **54% of the signal** (dropping CORR from 0.683 to 0.310). v9.2 restores the Binary Alpha component to raw features.

- **Logic:** A Ridge Classifier (α = 10.0) identifies stocks with a high probability of exceeding the `0.5` threshold.
- **Integration:** This component provides a 5% "tactical tilt" to the default blend and a 40% weight to the `KZ_BINARY_N20` slot.

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

The model follows a bifurcated execution schedule to optimize compute resources on Google Colab (T4 GPU):

**Saturday — Full Training**
> 5-fold Purged Walk-Forward Cross-Validation. Includes DAE training, multi-target tree fitting, and risk vector generation. ~25–30 minute execution.

**Tuesday–Friday — Daily Inference**
> Weights are cached in Google Drive. Daily live data is processed through the frozen architecture for submission in ~2 minutes.

---

## 5. Statistical Validation

| Parameter | Configuration |
|-----------|--------------|
| **Feature Set** | Medium (v5.2) |
| **CV Strategy** | Purged Walk-Forward (Purge Gap = 4 eras) |
| **Regularization** | Heavy ℓ₂ penalty on Binary Alpha to mitigate the high dimensionality (780 features) of the raw input |
