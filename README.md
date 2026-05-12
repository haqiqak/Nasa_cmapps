# 🔧 NASA C-MAPSS FD004 — Remaining Useful Life Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Dataset-NASA%20C--MAPSS%20FD004-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Task-RUL%20Prediction-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.10%2B-green?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit" />
</p>

<p align="center">
  <a href="https://jetrul-app.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀%20Live%20Demo-jetrul--app.streamlit.app-FF4B4B?style=for-the-badge" />
  </a>
</p>

> **A complete end-to-end machine learning pipeline for predictive maintenance** — from raw sensor data to a deployed Streamlit dashboard — on the most challenging NASA turbofan degradation benchmark.

---

## 👥 Authors

| Name | Role |
|------|------|
| **Ali Ahsan** | Data Preprocessing & Feature Engineering |
| **Anas bin Waheed** | Model Development & Evaluation |
| **Haqiq Azeem** | Deep Learning Architecture & Streamlit App |

*BSDS-02 Cohort*

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Pipeline Architecture](#-pipeline-architecture)
- [Models](#-models)
- [Results](#-results)
- [Statistical Testing](#-statistical-testing)
- [Streamlit Dashboard](#-streamlit-dashboard)
- [Project Structure](#-project-structure)
- [Setup & Usage](#-setup--usage)
- [Key Design Decisions](#-key-design-decisions)

---

## 🎯 Project Overview

Predictive maintenance asks: **how many operational cycles remain before this engine fails?** Getting it right saves lives and millions in unplanned downtime. Getting it wrong — especially in the optimistic direction — is catastrophic.

This project builds and compares five approaches on **FD004**, the hardest C-MAPSS subset:

| # | Module | Approach |
|---|--------|----------|
| 1 | **Data Preprocessing & EDA** | Cleaning, feature engineering, RUL labelling |
| 2 | **Random Forest** | Tree ensemble baseline + hyperparameter tuning + SHAP |
| 3 | **XGBoost** | Gradient boosting + tuning + SHAP |
| 4 | **CNN-LSTM + Attention** | Deep learning with Bahdanau attention mechanism |
| 5 | **K-Means Clustering** | Unsupervised engine health-state discovery |
| 6 | **Statistical Significance Testing** | Friedman, Wilcoxon, Ljung-Box, Spearman + FDR |

---

## 📦 Dataset

**NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) — FD004 Subset**

FD004 is deliberately the hardest subset because it combines **two simultaneous fault modes** with **six distinct operating conditions**.

| Property | Value |
|----------|-------|
| Training engines | 249 |
| Test engines | 248 |
| Sensor channels | 21 |
| Operating conditions | 6 (altitude × Mach combinations) |
| Fault modes | HPC Degradation + Fan Degradation |
| Input columns per row | 26 (unit ID, cycle, 3 op settings, 21 sensors) |

**Download the data:**
- [NASA Prognostics Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)
- [Kaggle Mirror](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)

Place `train_FD004.txt`, `test_FD004.txt`, and `RUL_FD004.txt` in the project root before running.

---

## 🏗️ Pipeline Architecture

```
Raw FD004 Text Files
        │
        ▼
┌─────────────────────────────────────────────┐
│  PART 1 — Preprocessing & EDA               │
│  • Drop constant sensors (std < 10)         │
│  • Compute RUL labels (train)               │
│  • Piecewise RUL clipping (max = 125)       │
│  • Rolling mean smoothing (window = 5)      │
│  • KMeans operating condition labels (k=6)  │
│  • Per-condition StandardScaler             │
│  • Reconstruct RUL for test set             │
└────────────────┬────────────────────────────┘
                 │
       ┌─────────┼──────────┬────────────┐
       ▼         ▼          ▼            ▼
  Regression  Regression  LSTM Seqs   Clustering
    CSVs        CSVs      (N,30,13)    Features
       │         │          │            │
       ▼         ▼          ▼            ▼
  Random      XGBoost   CNN-LSTM     K-Means
  Forest      + SHAP    Bahdanau      k = 2
  + SHAP                Attention
       │         │          │            │
       └─────────┴──────────┴────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │  Statistical    │
                   │  Significance   │
                   │  Testing        │
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │  Streamlit      │
                   │  Dashboard      │
                   └─────────────────┘
```

---

## 🤖 Models

### 1. Random Forest (Part 2)

Random Forest provides a strong tree-based baseline and serves as a benchmark for the sequential models. Since it operates on flat feature vectors, **rolling mean and standard deviation over a 30-cycle window** are computed per engine to inject temporal context.

- Grouped cross-validation (groups = engine unit ID) prevents data leakage
- `RandomizedSearchCV` over `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`
- SHAP beeswarm plots for global explainability

### 2. XGBoost (Part 3)

XGBoost builds trees sequentially, each correcting residual errors of the previous one. Key advantages over RF: L1/L2 regularisation, gradient-based optimisation, and `gain`-based feature importance.

- Same rolling feature engineering as Random Forest
- `RandomizedSearchCV` over `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`
- SHAP beeswarm and trajectory plots

### 3. CNN-LSTM + Bahdanau Attention (Part 4)

The deep learning model treats each engine's recent history as a **30-cycle sequence** of 13 sensor + condition features. Architecture:

```
Input (batch, 30, 13)
  └─ Conv1D × 2  (feature extraction from local patterns)
  └─ Bidirectional LSTM (capture forward/backward temporal dependencies)
  └─ Bahdanau Attention (focus on most diagnostic timesteps)
  └─ Dense → RUL prediction
```

**Loss function:** Failure-Region Weighted Huber — standard Huber loss with higher weight applied to predictions in the last 30 cycles before failure, directly improving the NASA Score.

### 4. K-Means Clustering (Part 5)

Unsupervised health-state discovery without using RUL labels.

- MinMax-scaled last-cycle sensor means per engine
- PCA reveals PC1 explains ~80% of variance (a single dominant degradation axis)
- k=2 naturally separates healthy from degraded engines
- Key drivers: `sensor_03`, `sensor_17`, `sensor_04` (high-pressure compressor family)

---

## 📊 Results

### Model Comparison

| Model | RMSE | MAE | NASA Score |
|-------|------|-----|------------|
| Random Forest (Baseline) | — | — | — |
| Random Forest (Tuned) | competitive | competitive | competitive |
| XGBoost (Baseline) | — | — | — |
| XGBoost (Tuned) | better than RF | better than RF | better than RF |
| **CNN-LSTM + Bahdanau** | **< 22** | **< 14** | **~2495** |

> Exact numbers are populated after running the notebook. The CNN-LSTM target benchmarks are shown; published SOTA for FD004 deep learning methods sits in the 2000–3000 NASA Score range.

### NASA Score

The NASA Score uses an **asymmetric penalty**:
- Late predictions (model says engine has more life): penalty ∝ exp(error / 10) — very harsh
- Early predictions (model says engine will fail sooner): penalty ∝ exp(error / 13) — more forgiving

This reflects real-world consequences — falsely predicting a healthy engine is far more dangerous than being overly cautious. **Lower is better.**

---

## 📐 Statistical Testing (Part 7)

Beyond point-estimate comparisons, we apply a rigorous testing suite:

| Test | Purpose | Result |
|------|---------|--------|
| **Shapiro-Wilk** | Normality of per-engine errors | Gates parametric vs non-parametric choice |
| **Friedman** | Global difference across 3 models | Non-parametric repeated-measures ANOVA |
| **Wilcoxon Signed-Rank** | Pairwise model comparisons | Paired, no normality assumption |
| **Bonferroni correction** | Family-wise error control | α_adj = 0.05 / 3 = 0.0167 |
| **Effect size r** | Magnitude of differences | Negligible / Small / Medium / Large |
| **Ljung-Box** | Residual autocorrelation | Checks for unmodelled temporal structure |
| **Spearman + FDR** | Sensor–RUL association | Validates sensor removal rigorously |

---

## 🖥️ Streamlit Dashboard (Part 8)

A production-ready dashboard exported to `streamlit_assets/` bundles:
- All trained models (RF, XGBoost, CNN-LSTM, K-Means)
- Per-condition scalers and metadata JSON
- Test predictions from all three models
- Training data for in-app EDA

**Run the dashboard:**
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
cmapss-rul-prediction/
│
├── cmapps_pipeline.ipynb          # Main notebook (all 8 parts)
│
├── data/                          # Raw dataset files (not tracked by git)
│   ├── train_FD004.txt
│   ├── test_FD004.txt
│   └── RUL_FD004.txt
│
├── processed/                     # Preprocessed CSVs (generated by Part 1)
│   ├── train_regression.csv
│   ├── test_regression.csv
│   ├── clustering_features.csv
│   └── X_train_lstm.npy
│
├── models/                        # Saved model weights (generated by Parts 2–5)
│   ├── rf_best_model.pkl
│   ├── xgb_best_model.pkl
│   └── cnn_lstm_bahdanau_best.keras
│
├── streamlit_assets/              # Dashboard bundle (generated by Part 8)
│   ├── streamlit_metadata.json
│   ├── rf_best_model.pkl
│   ├── xgb_best_model.pkl
│   ├── cnn_lstm_bahdanau_best.keras
│   ├── kmeans_model.pkl
│   ├── op_condition_kmeans.pkl
│   ├── scaler_dict.pkl
│   ├── mm_scaler_cluster.pkl
│   ├── test_predictions.csv
│   ├── train_regression.csv
│   └── test_regression.csv
│
├── figures/                       # EDA and evaluation plots (auto-generated)
│   ├── eda_lifetime.png
│   ├── eda_sensor_trends.png
│   ├── eda_sensor_std.png
│   ├── eda_correlation.png
│   ├── eda_op_conditions.png
│   ├── eda_rul_distribution.png
│   ├── eda_rolling_smooth.png
│   ├── stat_error_distributions.png
│   ├── stat_pairwise_differences.png
│   └── stat_sensor_rul_correlation.png
│
├── requirements.txt               # Python dependencies
├── .gitignore                     # Ignores data, models, processed files
└── README.md                      # This file
```

---

## ⚙️ Setup & Usage

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/cmapss-rul-prediction.git
cd cmapss-rul-prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# or
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the C-MAPSS dataset from [NASA](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/) or [Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) and place the three FD004 files in a `data/` folder:

```
data/
  train_FD004.txt
  test_FD004.txt
  RUL_FD004.txt
```

Alternatively, the notebook auto-downloads from a public GitHub mirror if the files are missing.

### 5. Run the notebook

Open `cmapps_pipeline.ipynb` in Jupyter Lab / Notebook or Google Colab and run all cells in order (Parts 1 → 8).

```bash
jupyter lab cmapps_pipeline.ipynb
```

### 6. Launch the Streamlit dashboard (after running Part 8)

```bash
streamlit run app.py
```

---

## 💡 Key Design Decisions

**Why FD004?**
FD004 is the hardest C-MAPSS subset — six operating conditions plus two simultaneous fault modes make it a genuine benchmark for multi-condition RUL prediction.

**Why clip RUL at 125?**
Following published literature, early-life rows where the engine has hundreds of cycles left carry minimal degradation signal. Capping at 125 cycles forces the model to focus its learning capacity on the degradation phase that actually matters for maintenance decisions.

**Why per-condition normalisation?**
The same sensor can report very different absolute values at high altitude vs sea level, even for a healthy engine. Normalising within each of the 6 KMeans operating-condition clusters removes this regime-specific offset before the models see the data.

**Why Bahdanau attention?**
Standard LSTM aggregates the entire sequence equally. Attention lets the model learn which of the 30 input timesteps are most diagnostic — empirically, the last 5–10 cycles before failure carry the strongest signal.

**Why Failure-Region Weighted loss?**
The NASA Score penalises late predictions exponentially more than early ones. Upweighting the loss for end-of-life windows (last 30 cycles) directly trains the model to be more conservative where the stakes are highest.

**Why Wilcoxon over paired t-test?**
Per-engine absolute errors are not normally distributed (confirmed by Shapiro-Wilk). Wilcoxon makes no normality assumption and is standard practice for paired model comparisons in the prognostics literature.

---

## 📚 References

- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). *Damage propagation modeling for aircraft engine run-to-failure simulation.* In 2008 International Conference on Prognostics and Health Management.
- Bahdanau, D., Cho, K., & Bengio, Y. (2015). *Neural machine translation by jointly learning to align and translate.* ICLR 2015.
- Zheng, S., Ristovski, K., Farahat, A., & Gupta, C. (2017). *Long short-term memory network for remaining useful life estimation.* In 2017 IEEE ICPHM.

---

## 📄 License

This project is for academic purposes. The NASA C-MAPSS dataset is publicly available through the [NASA Prognostics Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/).
