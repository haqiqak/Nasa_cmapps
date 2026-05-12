# 📁 Dataset Setup Guide

The NASA C-MAPSS dataset is publicly available but must be downloaded separately before running the notebook.

---

## Required Files

You need exactly three files from the **FD004** subset:

| File | Description | Typical Size |
|------|-------------|-------------|
| `train_FD004.txt` | Training engine sequences (run-to-failure) | ~7.5 MB |
| `test_FD004.txt` | Test engine sequences (truncated) | ~4.8 MB |
| `RUL_FD004.txt` | Ground-truth RUL for last cycle of each test engine | ~2 KB |

---

## Option A — NASA Official Repository

1. Visit: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
2. Find **"Turbofan Engine Degradation Simulation Data Set"**
3. Download `CMAPSSData.zip`
4. Extract and copy the three FD004 files to your project root (or `data/`)

---

## Option B — Kaggle Mirror (easier)

1. Visit: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
2. Click **Download** (requires free Kaggle account)
3. Extract `CMAPSSData.zip`
4. Copy the three FD004 files to your project root (or `data/`)

---

## Option C — Auto-download in notebook

The notebook (Part 1, Cell 1) automatically attempts to download from a public GitHub mirror:

```
https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData/
```

If that mirror is unavailable, it falls back to printing the manual download instructions above.

---

## File Placement

After downloading, place the files here:

```
cmapss-rul-prediction/
├── data/                   ← recommended (already in .gitignore)
│   ├── train_FD004.txt
│   ├── test_FD004.txt
│   └── RUL_FD004.txt
└── cmapps_pipeline.ipynb
```

Or directly in the project root alongside the notebook (the notebook looks for files in `.` by default).

---

## Data Format

Each line in `train_FD004.txt` and `test_FD004.txt` has **26 space-separated values**:

```
unit_id  cycle  op_set_1  op_set_2  op_set_3  sensor_01 ... sensor_21
```

`RUL_FD004.txt` has one integer per line — the true RUL at the last observed cycle for each test engine, in engine-ID order.

---

## Verifying the Download

Run this quick check in Python:

```python
import pandas as pd

COL_NAMES = (['unit_id', 'cycle', 'op_set_1', 'op_set_2', 'op_set_3']
             + [f'sensor_{i:02d}' for i in range(1, 22)])

train = pd.read_csv('data/train_FD004.txt', sep=r'\s+', header=None, names=COL_NAMES)
test  = pd.read_csv('data/test_FD004.txt',  sep=r'\s+', header=None, names=COL_NAMES)
rul   = pd.read_csv('data/RUL_FD004.txt',   sep=r'\s+', header=None, names=['RUL_true'])

print(f'Train: {train.shape}')   # Expected: ~61249 rows × 26 cols
print(f'Test:  {test.shape}')    # Expected: ~41214 rows × 26 cols
print(f'RUL entries: {len(rul)}')  # Expected: 248
```
