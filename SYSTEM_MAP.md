# System Architecture: Twin-Engine Market Neutral Strategy
**Status:** Production (v2.2)
**Infrastructure:** Google Colab Pro+ / NVIDIA A100

```mermaid
graph TD
    %% --- STYLES ---
    classDef raw fill:#2d2d2d,stroke:#555,stroke-width:2px,color:#fff;
    classDef process fill:#0D47A1,stroke:#000,stroke-width:2px,color:#fff;
    classDef model fill:#1B5E20,stroke:#00cc96,stroke-width:2px,color:#fff;
    classDef logic fill:#b71c1c,stroke:#ef553b,stroke-width:2px,color:#fff;
    classDef output fill:#4a148c,stroke:#aa00ff,stroke-width:2px,color:#fff;

    %% --- DATA INGESTION ---
    subgraph INGESTION ["PHASE 1: INGESTION (Polars)"]
        Raw[Numerai v5.1 Parquet]:::raw -->|Lazy Load| Polars[Polars DataFrame]:::process
        Polars -->|Filter| Features[Medium Set ~780 Features]:::process
    end

    %% --- VALIDATION LOGIC ---
    subgraph VALIDATION ["PHASE 2: REGIME SEPARATION"]
        Features --> Split{Chronological Split}:::logic
        Split -->|Era 0 - 80%| Train[Training Set]:::raw
        Split -->|Era > 80%| Val[Validation Set]:::raw
        
        note1[STRICT CUTOFF<br/>No Look-Ahead Bias]:::logic
        Split -.-> note1
    end

    %% --- MODELING ---
    subgraph ENGINES ["PHASE 3: TWIN-ENGINE TRAINING"]
        Train -->|Gradient One-Side Sampling| LGBM[Engine 1: LightGBM]:::model
        Train -->|Histogram Method| XGB[Engine 2: XGBoost]:::model
        
        LGBM -.->|Deep Trees| Snipe(The Sniper)
        XGB -.->|Robust Splits| Spot(The Spotter)
    end

    %% --- ENSEMBLE ---
    subgraph EXECUTION ["PHASE 4: ENSEMBLE & DEPLOYMENT"]
        LGBM -->|Pred A| Mean((Weighted Average)):::process
        XGB -->|Pred B| Mean
        
        Mean -->|0.5 * A + 0.5 * B| Signal[Final Rank Signal]:::output
        Signal -->|Upload| API[Numerai API]:::output
    end

    %% --- LINKS ---
    Val -->|Backtest| Mean
