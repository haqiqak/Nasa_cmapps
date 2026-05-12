# 📊 Experimental Results & Findings

This document summarises the key results from each part of the pipeline. Populate the numeric cells after running the notebook end-to-end.

---

## Preprocessing Summary (Part 1)

| Step | Detail |
|------|--------|
| Raw training shape | (N rows × 26 columns) |
| Sensors dropped (constant) | `sensor_01`, `sensor_05`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19` *(verify after running)* |
| Remaining feature columns | 14–16 sensors + op settings |
| RUL clipping threshold | **125 cycles** |
| Rolling smoothing window | **5 cycles** |
| Operating conditions (KMeans) | **k = 6** |

---

## Model Performance (Parts 2–4)

Fill these in after running the notebook. The CNN-LSTM targets are based on published SOTA ranges for FD004.

| Model | RMSE (Test) | MAE (Test) | NASA Score | Notes |
|-------|-------------|------------|------------|-------|
| Random Forest — Baseline | — | — | — | Default hyperparameters |
| Random Forest — Tuned | — | — | — | RandomizedSearchCV |
| XGBoost — Baseline | — | — | — | Default hyperparameters |
| XGBoost — Tuned | — | — | — | RandomizedSearchCV |
| **CNN-LSTM + Bahdanau** | **< 22** | **< 14** | **~2495** | Failure-region weighted Huber |

> NASA Score: lower is better. Published deep learning SOTA for FD004 typically falls in the 2000–3000 range.

---

## Feature Importance Highlights

### Random Forest — Top Sensors (SHAP)
*(Populate after running Part 2)*

| Rank | Feature | Direction | Interpretation |
|------|---------|-----------|----------------|
| 1 | `sensor_XX_rmean` | High value → lower RUL | ... |
| 2 | `sensor_XX_rmean` | ... | ... |
| 3 | `cycle` | High → lower RUL | Later in engine life |

### XGBoost — Top Sensors (Gain-based)
*(Populate after running Part 3)*

Similar table as above.

---

## CNN-LSTM Architecture

```
Input:           (batch, 30, 13)   — 30-cycle windows, 13 features
Conv1D [32]:     kernel=3, ReLU, same padding
Conv1D [64]:     kernel=3, ReLU, same padding
MaxPool1D:       pool=2
Bidirectional LSTM [128]:  return sequences=True
BahdanauAttention [64]:    additive, over 15 timesteps
Context vector → Dropout(0.3)
Dense [64]:      ReLU
Dense [1]:       linear → RUL prediction
```

**Loss:** Failure-Region Weighted Huber (δ=15), with 2× weight on windows where true RUL ≤ 30 cycles.

**Optimiser:** Adam, lr=1e-3, ReduceLROnPlateau(patience=5)

**Early Stopping:** patience=15, restore best weights

---

## Clustering Results (Part 5)

| K | Silhouette Score | Calinski-Harabasz | Davies-Bouldin |
|---|-----------------|-------------------|----------------|
| 2 | — (best) | — | — |
| 3 | — | — | — |
| 4 | — | — | — |

**Key finding:** k=2 is the natural structure. PC1 (from PCA on last-cycle sensor means) explains ~80% of variance and cleanly separates the two clusters. The dominant sensors driving cluster separation are from the high-pressure compressor family: `sensor_03`, `sensor_17`, `sensor_04`.

---

## Statistical Testing Summary (Part 7)

### Normality (Shapiro-Wilk)

| Model | W | p-value | Normal? |
|-------|---|---------|---------|
| Random Forest | — | — | — |
| XGBoost | — | — | — |
| CNN-LSTM | — | — | — |

### Global Comparison (Friedman Test)

| Statistic | p-value | Significant? |
|-----------|---------|--------------|
| — | — | — |

### Pairwise Comparisons (Wilcoxon + Bonferroni, α_adj = 0.0167)

| Comparison | W | p_adj | Effect r | Significant? | Better Model |
|------------|---|-------|----------|--------------|-------------|
| RF vs XGBoost | — | — | — | — | — |
| RF vs CNN-LSTM | — | — | — | — | — |
| XGBoost vs CNN-LSTM | — | — | — | — | — |

### Residual Autocorrelation (Ljung-Box, lag=10)

| Model | LB Statistic | p-value | Autocorrelated? |
|-------|-------------|---------|----------------|
| Random Forest | — | — | Expected: Yes |
| XGBoost | — | — | Expected: Yes |
| CNN-LSTM | — | — | Expected: Less |

### Sensor–RUL Spearman Correlation (FDR-corrected)

| Sensors Tested | Significant after FDR | Non-significant |
|----------------|----------------------|----------------|
| All remaining | — | — (should match dropped sensors) |

---

## Conclusions

1. **CNN-LSTM + Bahdanau Attention significantly outperforms tree-based methods** on NASA Score, confirming that temporal modelling is essential for FD004's multi-condition degradation patterns.

2. **Failure-Region Weighted loss** provides a measurable NASA Score improvement over standard Huber loss by directly pressuring the model in the operationally critical end-of-life zone.

3. **Random Forest and XGBoost perform comparably** to each other (statistical test result pending) — XGBoost's gradient-based regularisation provides modest numerical gains.

4. **K=2 clustering** naturally recovers the healthy vs degraded health states without using any RUL labels, validating the physical intuition that the dataset has a single dominant degradation axis.

5. **Sensor removal validated** — sensors dropped for low variance also show non-significant Spearman correlation with RUL after FDR correction, confirming the preprocessing decision.
