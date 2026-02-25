project: Operation Overwatchversion: 9.2status: Deployment Readyclassification: Quantitative Alpha Researchtags:numeraiquantitative-financemachine-learningrisk-managementOperation Overwatch v9.2BLUF (Bottom Line Up Front)Operation Overwatch v9.2 is a production-grade quantitative trading framework designed for the Numerai tournament. It utilizes a hybrid multi-target ensemble integrating deep gradient boosting, neural latent feature extraction, and high-regularization binary classification to navigate the v5.2 "Sunshine" data regime.Mission ProfileThis repository serves as the primary research and execution engine for Operation Iron Triad. Developed by a retired U.S. Army Special Forces CSM with a DBA in Finance, this system transitions professional-grade risk management and tactical precision into the quantitative finance space.Objective: Maximize Sharpe and CORR on the Numerai leaderboard while maintaining strict factor neutrality.Operational Status: Active.Data Version: v5.2 (Medium Feature Set).Systems Mapgraph TD
    %% Data Acquisition Layer
    subgraph Data_Sync [Data Acquisition & Pre-Processing]
        A1[NumerAPI Connection] --> A2[v5.2 Parquet: Train/Live]
        A2 --> A3[Feature Selection: Medium Set]
        A3 --> A4[Fill Nulls / Type Casting]
    end

    %% Model Component Layer
    subgraph Signal_Generation [Multi-Model Signal Generation]
        A4 --> B1[Neural Tactician: DAE + ResMLP]
        A4 --> B2[Tree Ensemble: LGBM + XGBoost]
        A4 --> B3[Binary Alpha: Ridge Classifier]
        
        %% Neural Specifics
        B1 --> B1a[DAE 64-Dim Bottleneck]
        B1a --> B1b[5-Seed MLP Average]
        
        %% Tree Specifics
        B2 --> B2a[Multi-Target Ensemble]
        B2a --> B2b[Weighted Rank Normalization]
    end

    %% Aggregation & Risk Layer
    subgraph Tactical_Control [Risk Engine & Neutralization]
        B1b --> C1[Ensemble Blender]
        B2b --> C1
        B3 --> C1
        
        A4 --> C2[PCA Risk Vectorization: 50 Components]
        C1 --> C3[Factor Neutralization]
        C2 --> C3
    end

    %% Submission Layer
    subgraph Deployment [Model Slot Deployment]
        C3 --> D1[KZ_CORE_N00: 0% Neut]
        C3 --> D2[KZ_DEF_N90: 90% Neut]
        C3 --> D3[KZ_BINARY_N20: 20% Neut + Binary Tilt]
        
        D1 --> E1[NumerAPI Upload]
        D2 --> E1
        D3 --> E1
    end

    %% Styling
    style Signal_Generation fill:#1a1a1a,stroke:#333,stroke-width:2px,color:#fff
    style Tactical_Control fill:#2d3436,stroke:#00b894,stroke-width:2px,color:#fff
    style Deployment fill:#2d3436,stroke:#0984e3,stroke-width:2px,color:#fff
Architectural Framework1. Neural Tactician (DAE-MLP)Bottleneck: 64-dimension latent space (optimized for signal-to-noise ratio).Initialization: Denoising Autoencoder (DAE) with 0.1 swap noise.Prediction: 5-seed Residual MLP (ResMLP) ensemble using SiLU activations and Batch Normalization.2. Multi-Target Tree EnsembleModels: LightGBM and XGBoost (10,000 max trees, 0.005 learning rate).Targeting: Weighted exposure to four specific objectives:target_ender_20 (40%)target_cyrusd_20 (25%)target_teager2b_20 (20%)target_victor_20 (15%)3. Binary AlphaInput: Raw features (780 dimensions).Model: Ridge Classifier ($\alpha=10.0$).Function: Tail-event detection and tactical tilt for the KZ_BINARY_N20 model slot.Model Risk OverlayThis component treats model error as a primary risk factor, deploying institutional-grade mitigation strategies to ensure portfolio stability:Over-fitting Mitigation: Reverted neural latent bottleneck from 128 to 64 dimensions to force compressed, robust representation of the feature space.Regime Shift Defense: Ensemble targets ender, cyrusd, teager2b, and victor concurrently to reduce reliance on any single market regime.Signal Decay Guardrails: Ridge Classifier operates on raw feature inputs to capture tail events lost during PCA transformation.Exposure Control: A 50-component PCA generates risk vectors, facilitating Ridge-based factor neutralization to align final predictions with strict volatility bounds.Model Roster & Risk ConfigurationThe system deploys three distinct model slots to provide diversified exposure:Slot NameBlend TypeNeutralizationTactical FocusKZ_CORE_N00Default0%Raw Return AlphaKZ_DEF_N90Default90%Defense / Low VolatilityKZ_BINARY_N20Binary20%Tail-Event CaptureExecution SOPStandard Training Cycle (Saturdays)Synchronize: Download latest train.parquet and features.json.Train: Execute 5-fold Purged Walk-Forward CV (4-era gap).Validate: Review CORR, Sharpe, and Max Drawdown diagnostics.Persist: Save model weights to Google Drive.Daily Submission (Tue–Fri)Inference: Load cached weights from /Weights_v92/.Live Sync: Pull daily live.parquet.Neutralize: Apply 50-component PCA risk neutralization.Upload: Deploy predictions via NumerAPI.Technical SpecificationsEnvironment: Google Colab (Python 3.10+, T4/L4 GPU).Workflow Integration: Primary capture via Evernote; deep research and technical documentation managed in Obsidian.Dependencies: polars, lightgbm, xgboost, pytorch, numerapi.

## 👤 Author

**Brian Penrod, DBA**  
Retired U.S. Army Special Forces CSM | Doctor of Business Administration (Finance)

> “I combine military strategic planning with advanced quantitative finance to build systems that prioritize risk management, data integrity, and tactical execution.”
