# Decision Tree Project — Instructions

## Project Structure

```
A1/
├── decisionTree.py   # Custom Decision Tree (from scratch)
├── metrics.py        # From-scratch evaluation metrics
├── datasets.py       # Dataset loaders (Adult, Synthetic, Imbalanced)
├── evaluate.py       # Full evaluation pipeline
├── compare.py        # sklearn vs Custom DT comparison
├── results.json      # Output: full evaluation results
└── comparison.json   # Output: side-by-side comparison results
```

---

## Prerequisites

```bash
pip install numpy scikit-learn
```

---

## How to Run

### 1. Validate datasets only
Loads all 3 datasets and checks shapes/types.
```bash
python datasets.py
```
**Expected output:**
```
PASS  Real-world (Adult)  shape=(5000, 14)  classes=[0, 1]
PASS  Synthetic (noisy)   shape=(2000, 15)  classes=[0, 1]
PASS  Imbalanced (95/5)   shape=(2000, 12)  classes=[0, 1]
ALL DATASET VALIDATIONS PASSED
```

---

### 2. Validate metrics only
Runs 6 unit tests for Accuracy, Precision, Recall, F1.
```bash
python metrics.py
```
**Expected output:**
```
PASS  Test 1 – Perfect binary predictions
PASS  Test 2 – All wrong binary predictions
PASS  Test 3 – Accuracy 3/4 = 0.75
PASS  Test 4 – Divide-by-zero safety
PASS  Test 5 – NaN handling
PASS  Test 6 – Multi-class perfect predictions
ALL SANITY CHECKS PASSED
```

---

### 3. Run full evaluation pipeline
Runs metric sanity checks + trains both models on all 3 datasets + prints table + saves `results.json`.
```bash
python evaluate.py
```

---

### 4. Run sklearn comparison (recommended entry point)
Trains Custom DT and sklearn DT on all 3 datasets, prints side-by-side table, saves `comparison.json`.
```bash
python compare.py
```
**Expected output:**
```
Dataset                Model               Accuracy  Precision   Recall       F1
Real-world (Adult)     Custom DT             0.8330     0.7826   0.7581   0.7702
Real-world (Adult)     Sklearn DT            0.8260     0.7764   0.7363   0.7558
Synthetic (noisy)      Custom DT             0.7625     0.7629   0.7629   0.7629
Synthetic (noisy)      Sklearn DT            0.7500     0.7524   0.7511   0.7517
Imbalanced (95/5)      Custom DT             0.9400     0.6876   0.6535   0.6701
Imbalanced (95/5)      Sklearn DT            0.9400     0.6436   0.5635   0.6009
```

---

## Troubleshooting

### OpenML fetch fails for Adult dataset
Clear the sklearn cache and retry:
```powershell
Remove-Item -Recurse -Force "$env:USERPROFILE\scikit_learn_data\openml" -ErrorAction SilentlyContinue
python compare.py
```
If OpenML remains unreachable, the loader automatically falls back to a synthetic 5000×14 dataset and labels it `"Real-world (Adult — fallback)"` so the pipeline keeps running.

---

## Notes

- All metrics (Accuracy, Precision, Recall, F1) are implemented **from scratch** — no `sklearn.metrics` used.
- `sklearn` is used **only** for model training (`DecisionTreeClassifier`) and data utilities (`train_test_split`, `make_classification`, `fetch_openml`).
- All random seeds are fixed to `42` for full reproducibility.
- Run all scripts from inside the `A1/` directory.
