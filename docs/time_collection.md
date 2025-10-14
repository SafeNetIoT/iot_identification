# TestPipeline — IoT Feature Extraction and Evaluation

## Overview
`TestPipeline` automates the **end-to-end testing** of feature extraction and model evaluation across multiple packet capture (`.pcap`) files and time intervals.  

It performs:
1. Feature extraction from raw PCAPs  
2. Time-based collection windowing  
3. Dataset preparation, scaling, and splitting  
4. Model training, evaluation, and saving    

---
## Example Usage

```python
from src.identification.test_pipeline import TestPipeline

pipeline = TestPipeline()
pipeline.test_intervals()  # run full experiment
```

Or, to test a single PCAP and interval:

```python
pipeline = TestPipeline()
df = pipeline.extract("data/raw/alexa_swan_kettle/2023-10-19/sample.pcap", time_interval=30)
print(df.head())
```

---

## Implementation Notes
- Flow aggregation relies on the `five_tuple` (src/dst IPs and ports, protocol).  
- Designed for modular replacement with the `ExtractionPipeline` class for production use.  
- Compatible with both streaming and batch-based feature extraction modes.  
