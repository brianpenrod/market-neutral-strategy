# Operation Overwatch: Causal Factor-Aware Alpha Generation
**Author:** Brian Penrod, DBA  
**Date:** February 2026  
**System Version:** v5.2 "Faith2"

---

## Abstract

In the domain of quantitative finance, the primary challenge is distinguishing between **Market Beta** (systemic risk) and **Idiosyncratic Alpha** (true skill). Traditional machine learning models often conflate the two, leading to high correlation with major market factors and subsequent degradation of performance in novel regimes. 

**Operation Overwatch** introduces a novel architecture for the Numerai Tournament that utilizes **Linear Non-Gaussian Acyclic Models (LiNGAM)** for Causal Discovery and a proprietary **Risk Molding** protocol. By identifying directional dependencies between features before training, and orthogonalizing the output against latent market factors post-inference, the system generates robust, market-neutral alpha with a verifiable distinctness from standard momentum or value strategies.

---

## 1. Introduction: The Alpha Decay Problem

Standard gradient boosting models (XGBoost, LightGBM) are powerful pattern recognition engines. However, without strict constraints, they tend to "learn" the market cycle. For example, if Tech stocks are rallying, the model learns to prioritize "Tech-like" features. When the cycle turns, the model fails.

This paper proposes a solution based on two core tenets:
1.  **Causality over Correlation:** A feature is only valuable if it has a directional impact on returns, independent of the market.
2.  **Active Risk Molding:** Alpha is not just what you predict; it is what remains after you remove the Beta.

## 2. Intelligence Logistics (v5.2 "Faith2")

The system operates on the Numerai v5.2 dataset, utilizing **Int8 quantization** for high-velocity inference.

### 2.1 The Live Intel Patch
Data integrity is paramount. To prevent "Stale Universe" errors—where a model predicts on a past stock universe rather than the current live set—the system implements a **Live Intel Patch**. This logic gate forces a fresh acquisition of the daily `live.parquet` file at runtime, ensuring 100% coverage of the ~7,000 stock investment universe.

### 2.2 Quantization Handling
The v5.2 data compresses features into integer bins (0-4). While efficient for storage, this destroys precision in linear algebra operations. Operation Overwatch implements an automated casting layer that converts `int8` inputs to `float32` prior to the Causal Discovery phase, restoring the mathematical fidelity required for structure learning.

## 3. The Causal Tactician (Signal Generation)

Unlike traditional models that throw all 2,000+ features into a regressor, Overwatch employs a pre-training **Causal Discovery** phase.

### 3.1 LiNGAM Algorithm
We utilize the **DirectLiNGAM** algorithm to construct a Directed Acyclic Graph (DAG) of the feature set. This allows us to identify **Interaction Pairs** where:
$$Feature_A \rightarrow Feature_B$$
This implies that $Feature_A$ is a parent node affecting $Feature_B$.

### 3.2 Feature Augmentation
Once these "Hidden Physics" are identified, the system engineers interaction terms (e.g., $Feature_A \times Feature_B$) and injects them into the training set. This allows the ensemble engines to learn non-linear causal structures that linear correlation matrices miss.

## 4. The Alpha Forge (Ensemble Architecture)

The raw signal is generated via a "Train Once" architecture to ensure consistency. The augmented dataset is fed into two parallel engines:

1.  **LightGBM (Gradient Boosting):** configured for leaf-wise growth to capture complex, deep interactions.
2.  **XGBoost (Histogram):** configured for depth-wise growth to ensure robust generalization.

The outputs are averaged (50/50) to produce the `Raw_Alpha_Signal`.

## 5. Risk Molding: The Orthogonalization Protocol

This is the system's core innovation. Rather than retraining the model for different risk tolerances, we mathematically project the single `Raw_Alpha_Signal` onto different risk manifolds.

### 5.1 Latent Factor Extraction (PCA)
We apply **Principal Component Analysis (PCA)** to the feature set to extract the top 50 latent components. In financial terms, these components represent the "Market Beta"—hidden factors like Sector, Momentum, Volatility, and Size.

### 5.2 Orthogonalization
We use Ridge Regression to calculate the residual of our Alpha against these Beta factors:
$$\hat{y}_{neutral} = \hat{y}_{raw} - \beta (\text{MarketFactors})$$

### 5.3 Deployment Profiles
This process creates three distinct investment strategies from a single intelligence source:

* **KZ_CORE_N00 (Aggressor):** 0% Neutralization. Pure Causal Alpha. This profile is allowed to hold market beta if the Causal Engine deems it necessary. It captures maximum upside during trending markets.
* **KZ_BAL_N50 (Hybrid):** 50% Factor Neutralization. A balanced approach that dampens volatility while retaining directional conviction.
* **KZ_DEF_N75 (Bunker):** 75% Factor Neutralization. A strictly market-neutral profile. This model generates returns solely from stock-specific selection (True Contribution), making it uncorrelated to major market indices.

## 6. Performance & Validation

In live production (Round 1198), the system demonstrated:
* **Universe Coverage:** 7,030 Global Equities.
* **Decoupling:** A correlation of **0.69** between the Aggressive and Defensive profiles. This proves that the Risk Molding engine successfully separates the signal into distinct Alpha and Beta components.
* **Conviction:** A "Strong Buy" consensus (>0.80 probability) on 11.5% of the universe, indicating high selectivity rather than random noise.

## 7. Conclusion

Operation Overwatch represents a shift from "Black Box" machine learning to **Gray Box** causal inference. By understanding *why* features interact (Causality) and actively managing *what* the model is exposed to (Risk Molding), the system achieves a level of robustness and tactical flexibility that standard "Shotgun" models cannot match.

---
*© 2026 Kinetic Zero Research. All Rights Reserved.*
