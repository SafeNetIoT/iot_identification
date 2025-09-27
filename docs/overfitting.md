# Overfitting Analysis of RandomForest Model

## Objective
The goal of this analysis was to determine whether our RandomForest model was **overfitting** — i.e., learning training data too well while failing to generalize to unseen data.

---

## Methods

### 1. Train vs Validation/Test Accuracy
- After training, we evaluated the model on both:
  - **Training set**
  - **Held-out validation/test set**
- We compared the accuracies:
  - **Large gap (Train ≫ Test)** → strong overfitting  
  - **Small gap** → good generalization  

### 2. Cross-Validation (CV)
- We used **RandomizedSearchCV** with `cv=3–5` folds to estimate model performance across different splits of the training data.  
- Cross-validation helps detect overfitting because:
  - If the model memorizes a fold, accuracy drops on the other folds.
  - Variation between folds (`std_test_score`) reflects stability.

### 3. Saving Fold-Level Results
- By default, `RandomizedSearchCV` reports only mean scores.
- We extracted granular results from `rand_search.cv_results_`:
  ```python
  import pandas as pd
  cv_results = pd.DataFrame(rand_search.cv_results_)
  cols = [c for c in cv_results.columns if "split" in c and "test_score" in c]
  print(cv_results[cols + ["mean_test_score", "std_test_score"]])
  cv_results.to_csv("cv_results.csv", index=False)
  ```
- This allowed us to inspect **fold-level accuracies**, not just the mean.

---

## Results

### Accuracy Comparison
- **Train Accuracy:** `0.9985`  
- **Validation/Test Accuracy:** `0.9710`  
- Gap ≈ 2.7% → mild overfitting, but overall strong generalization.

### Cross-Validation Scores (sample candidates)
| split0 | split1 | split2 | mean | std |
|--------|--------|--------|------|-----|
| 0.9966 | 0.9966 | 0.9968 | 0.9967 | 0.00007 |
| 0.8911 | 0.8902 | 0.8910 | 0.8908 | 0.00038 |
| 0.9416 | 0.9433 | 0.9456 | 0.9435 | 0.00164 |

- Candidate 0 shows near-perfect scores across folds.  
- Candidate 2 is slightly weaker but consistent.  
- Standard deviation is very low, meaning the model is stable across folds.

### Observations
- Cross-validation mean ≈ 99.7%, but test accuracy ≈ 97.1%.  
- This indicates **slight overfitting**: the model performs a bit worse on the true test set than CV predicted.  
- Some individual classes (e.g. `echodot4`, `nest_cam`) had significantly lower F1-scores, suggesting **class-level imbalance or difficulty**, not just overfitting.

---

## Conclusions
- The RandomForest model shows **mild overfitting** (train ≈ 100% vs. test ≈ 97%).  
- Cross-validation confirmed that the model is stable across folds (low variance), but the true test set reveals a slightly larger drop in accuracy.  
- Overfitting is not severe, but attention may be needed for weaker-performing classes.

---

## Further analysis
1. Try restricting tree complexity (`max_depth`, `min_samples_leaf`) to reduce memorization.
2. Rebalance or augment underperforming classes.
3. Consider alternative models (e.g. gradient boosting) to improve generalization.
4. Use **Bayesian optimization** for more efficient hyperparameter tuning.