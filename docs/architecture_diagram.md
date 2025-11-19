# DeepSequence Architecture Diagram

## High-Level Architecture

```mermaid
graph TB
    subgraph Input["📊 Input Data"]
        TS[Time Series Data<br/>ds, StockCode, Quantity]
        EXOG[Exogenous Variables<br/>Price, Clusters, Holidays]
    end
    
    subgraph Seasonal["🌊 Seasonal Component"]
        TIME[Time Feature<br/>Extraction]
        EMBED_W[Weekly<br/>Embedding]
        EMBED_M[Monthly<br/>Embedding]
        EMBED_Y[Yearly<br/>Embedding]
        HIDDEN_S[Hidden Layers<br/>+ Dropout]
        OUT_S[Seasonal Output]
        
        TIME --> EMBED_W
        TIME --> EMBED_M
        TIME --> EMBED_Y
        EMBED_W --> HIDDEN_S
        EMBED_M --> HIDDEN_S
        EMBED_Y --> HIDDEN_S
        HIDDEN_S --> OUT_S
    end
    
    subgraph Regressor["📈 Regressor Component"]
        CAT[Categorical Features<br/>StockCode, Cluster]
        CONT[Context Variables<br/>Price, Lags]
        EMBED_R[Embedding Layer]
        LATTICE[Lattice Layer<br/>Constraints]
        HIDDEN_R[Hidden Layers<br/>+ Dropout]
        OUT_R[Regression Output]
        
        CAT --> EMBED_R
        CONT --> LATTICE
        EMBED_R --> HIDDEN_R
        LATTICE --> HIDDEN_R
        HIDDEN_R --> OUT_R
    end
    
    subgraph Combine["⚡ Combination Layer"]
        ADD[Additive/Multiplicative<br/>Mode]
        FINAL[Final Forecast<br/>ŷ]
        
        ADD --> FINAL
    end
    
    TS --> TIME
    TS --> CAT
    EXOG --> CONT
    
    OUT_S --> ADD
    OUT_R --> ADD
    
    style Input fill:#e1f5ff
    style Seasonal fill:#fff4e1
    style Regressor fill:#ffe1f5
    style Combine fill:#e1ffe1
```

## Detailed Component Architecture

```mermaid
graph LR
    subgraph SeasonalDetail["Seasonal Component Details"]
        direction TB
        D[Date: 2023-01-15]
        D --> WOM[Week of Month: 3]
        D --> WEEK[Week Number: 2]
        D --> MONTH[Month: 1]
        D --> QUARTER[Quarter: 1]
        D --> DOW[Day of Week: 0]
        D --> DOY[Day of Year: 15]
        
        WOM --> E1[Embed 10D]
        WEEK --> E2[Embed 10D]
        MONTH --> E3[Embed 10D]
        
        E1 --> CONCAT[Concatenate]
        E2 --> CONCAT
        E3 --> CONCAT
        
        CONCAT --> FC1[Dense + ReLU]
        FC1 --> DROP1[Dropout 0.1]
        DROP1 --> FC2[Dense + ReLU]
        FC2 --> S_OUT[Seasonal: σ_s]
    end
    
    subgraph RegressorDetail["Regressor Component Details"]
        direction TB
        SKU[StockCode: 20677]
        CLUST[Cluster: 9]
        PRICE[Price: 45.2]
        LAG[Lag Features]
        
        SKU --> ESKU[Embed 10D]
        CLUST --> ECLUST[Embed 10D]
        
        ESKU --> CONCAT2[Concatenate]
        ECLUST --> CONCAT2
        PRICE --> LAT[Lattice Layer<br/>Monotonic Constraints]
        LAG --> LAT
        
        LAT --> CONCAT2
        CONCAT2 --> FC3[Dense + ReLU]
        FC3 --> DROP2[Dropout 0.1]
        DROP2 --> FC4[Dense + ReLU]
        FC4 --> R_OUT[Trend: τ_r]
    end
    
    subgraph CombineDetail["Final Combination"]
        S_OUT --> COMBINER{Mode?}
        R_OUT --> COMBINER
        COMBINER -->|Additive| ADD_OUT[ŷ = σ_s + τ_r]
        COMBINER -->|Multiplicative| MULT_OUT[ŷ = σ_s × τ_r]
    end
    
    style SeasonalDetail fill:#fff4e1
    style RegressorDetail fill:#ffe1f5
    style CombineDetail fill:#e1ffe1
```

## Data Flow Through Network

```mermaid
flowchart TD
    START([Raw Data]) --> PREP[Data Preparation]
    
    PREP --> FEAT[Feature Engineering]
    FEAT --> |Time Features| SEAS_PATH[Seasonal Path]
    FEAT --> |Context Features| REG_PATH[Regressor Path]
    
    SEAS_PATH --> SEAS_EMBED[Temporal Embeddings<br/>Weekly/Monthly/Yearly]
    SEAS_EMBED --> SEAS_NN[Neural Network<br/>2-3 Hidden Layers]
    SEAS_NN --> SEAS_OUT[Seasonal Component<br/>σ_s ∈ ℝ]
    
    REG_PATH --> REG_EMBED[Feature Embeddings<br/>SKU/Cluster]
    REG_PATH --> REG_LAT[Lattice Layer<br/>Constraints Applied]
    REG_EMBED --> REG_CONCAT[Concatenate]
    REG_LAT --> REG_CONCAT
    REG_CONCAT --> REG_NN[Neural Network<br/>2-3 Hidden Layers]
    REG_NN --> REG_OUT[Trend Component<br/>τ_r ∈ ℝ]
    
    SEAS_OUT --> COMBINE{Combination<br/>Strategy}
    REG_OUT --> COMBINE
    
    COMBINE -->|α = additive| ADD[ŷ = σ_s + τ_r]
    COMBINE -->|α = multiplicative| MULT[ŷ = σ_s × τ_r]
    
    ADD --> LOSS[Loss: MAPE/MAE/MSE]
    MULT --> LOSS
    
    LOSS --> OPT[Optimizer: Adam]
    OPT --> BACK[Backpropagation]
    BACK --> UPDATE[Weight Update]
    UPDATE --> |Training Loop| SEAS_NN
    UPDATE --> |Training Loop| REG_NN
    
    LOSS --> |Validation| EARLY[Early Stopping]
    EARLY --> |Best Model| SAVE([Saved Model])
    
    style START fill:#e1f5ff
    style SAVE fill:#90EE90
    style LOSS fill:#FFB6C1
    style COMBINE fill:#FFD700
```

## Training Pipeline

```mermaid
sequenceDiagram
    participant Data
    participant Seasonal
    participant Regressor
    participant Model
    participant Optimizer
    
    Data->>Seasonal: Time series (dates, SKU IDs)
    Data->>Regressor: Exogenous vars (price, lags)
    
    activate Seasonal
    Seasonal->>Seasonal: Extract time features
    Seasonal->>Seasonal: Create embeddings
    Seasonal->>Seasonal: Pass through NN
    Seasonal-->>Model: Seasonal output σ_s
    deactivate Seasonal
    
    activate Regressor
    Regressor->>Regressor: Encode categoricals
    Regressor->>Regressor: Apply constraints
    Regressor->>Regressor: Pass through NN
    Regressor-->>Model: Trend output τ_r
    deactivate Regressor
    
    activate Model
    Model->>Model: Combine: ŷ = σ_s + τ_r
    Model->>Model: Calculate loss
    Model-->>Optimizer: Gradients
    deactivate Model
    
    activate Optimizer
    Optimizer->>Optimizer: Update weights (Adam)
    Optimizer-->>Seasonal: Updated params
    Optimizer-->>Regressor: Updated params
    deactivate Optimizer
    
    loop Training Epochs
        Data->>Model: Next batch
        Model->>Optimizer: Compute gradients
        Optimizer->>Model: Update weights
    end
    
    Model-->>Data: Final predictions
```

## Mathematical Formulation

```mermaid
graph TD
    subgraph Formula["DeepSequence Formula"]
        EQ1["ŷ = f(σ_s, τ_r | α)"]
        EQ2["σ_s = g_seasonal(t, embed_temporal)"]
        EQ3["τ_r = g_regressor(x, embed_context)"]
        EQ4["α ∈ {additive, multiplicative}"]
        
        EQ1 --> EQ2
        EQ1 --> EQ3
        EQ1 --> EQ4
        
        EQ2 --> T1["t: time features<br/>wom, week, month, quarter"]
        EQ3 --> X1["x: context features<br/>price, lags, clusters"]
    end
    
    style Formula fill:#f0f0f0
    style EQ1 fill:#FFD700
```

## Model Characteristics

| Component | Description | Key Features |
|-----------|-------------|--------------|
| **Seasonal** | Captures periodic patterns | • Weekly/Monthly/Yearly cycles<br/>• Temporal embeddings<br/>• Flexible architecture |
| **Regressor** | Models trends & external factors | • Exogenous variables<br/>• Constraint handling<br/>• Feature interactions |
| **Combination** | Merges components | • Additive (linear)<br/>• Multiplicative (non-linear)<br/>• Mode selection |
| **Training** | Optimization strategy | • MAPE/MAE/MSE loss<br/>• Adam optimizer<br/>• Early stopping |

## Key Innovations

```mermaid
mindmap
  root((DeepSequence))
    Prophet Inspired
      Seasonal Decomposition
      Additive Model
      Interpretability
    Deep Learning
      Neural Networks
      Embeddings
      Gradient Descent
    SKU Forecasting
      Multi-SKU Support
      Cluster Aware
      Intermittent Demand
    Flexibility
      Custom Activations
      Configurable Layers
      Constraint Handling
```

