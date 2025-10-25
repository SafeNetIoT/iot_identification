# `TestPipeline` — IoT Feature Extraction and Interval Analysis

## Overview
`TestPipeline` automates **end-to-end evaluation** of IoT network flow feature extraction and model performance across multiple time-based collection intervals.  
It performs both **model accuracy testing** and **feature stability analysis** to determine how varying the packet-capture window affects classification and feature distributions.

The pipeline orchestrates:
1. **Feature extraction** from raw PCAPs via the `ExtractionPipeline`.
2. **Data preparation** using the `DatasetPreparation` class (encoding, scaling, splitting).
3. **Model training & evaluation** using `MultiClassModel` (Random Forest baseline).
4. **Interval-based testing**, comparing performance and feature stability across time windows.

---

## Implementation Summary

### Class: `TestPipeline`
```python
class TestPipeline:
    def __init__(self, verbose=True): ...
    def combine_csvs(self, collection_time): ...
    def test_intervals(self): ...
    def compare_time_intervals(self, cache, test="ttest", alpha=0.05): ...
    def test_windows(self): ...
```

### Key Methods

#### `combine_csvs(collection_time)`
Aggregates features extracted from all devices for a given collection interval.  
Returns a unified DataFrame ready for model training.

#### `test_intervals()`
Trains and evaluates the classifier for each configured time interval (as defined in `config.json → TIME_INTERVALS`).

- Uses the same preprocessing pipeline and model architecture per interval.
- Results (models and evaluation metrics) are saved to the output directory.

**Finding #1 — Model Accuracy Stability**  
Across all tested collection intervals, the trained classifier achieved **consistent accuracy (~97%)**, indicating that:
- The chosen feature set generalizes well across different time-window lengths.
- Shorter and longer capture windows provide comparable discriminative information for device identification.

**Conclusion:** Model performance is robust to window size within the tested intervals.

---

#### `compare_time_intervals(cache, test="ttest", alpha=0.05)`
Performs pairwise statistical comparison of feature distributions between time intervals.  
- Supports both **Student’s t-test** (`test="ttest"`) and **Kolmogorov–Smirnov** (`test="ks"`) for non-parametric comparison.  
- Returns:
  - `results_df`: all pairwise tests (interval₁, interval₂, feature, p-value, significance)
  - `summary`: aggregated counts of how many interval pairs show significant differences per feature

#### `test_windows()`
Runs the **feature stability test** across all collection intervals defined in the configuration.

**Finding #2 — Feature Distribution Drift**
Pairwise Kolmogorov–Smirnov tests revealed that several features exhibit **significant distributional shifts** across time windows, even though model accuracy remains stable.

| Feature | # of Significant Differences |
|----------|------------------------------|
| `dur` | 28 |
| `iat_fwd_std` | 28 |
| `iat_fwd_mean` | 27 |
| `iat_cv_fwd` | 26 |
| `pkts_fwd` | 26 |
| `pkts_bwd` | 25 |
| `small_pkt_ratio_fwd` | 25 |
| `bytes_fwd` | 24 |
| `bytes_bwd` | 23 |
| `sport` | 23 |

**Interpretation:**  
Duration and inter-arrival-time-based features are highly time-dependent, showing significant drift across intervals.  
Despite this, classification accuracy remained stable, suggesting that other features (e.g., ratios, entropy) compensate for these fluctuations.

**Conclusion:**  
While model accuracy is window-invariant, several core temporal features exhibit **non-stationary distributions**, which should be monitored or normalized in production pipelines.

## Example Usage

### 1. Accuracy Test
```python
from src.identification.test_pipeline import TestPipeline

pipeline = TestPipeline()
pipeline.test_intervals()  # trains & evaluates for all intervals
```

### 2. Feature Stability Test
```python
pipeline = TestPipeline()
pipeline.test_windows()  # compares distributions between intervals
```

---