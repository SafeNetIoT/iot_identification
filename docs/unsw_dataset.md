# Testing on the UNSW Syndey dataset
This document explains the process of testing the model of choice on the UNSW Sydney's dataset as well as the current results. 

### Setup
The dataset is of the order of magnitude of 10s of GB, which is the required memory for downloading the full dataset. Out of all devices in the dataset only the Netatmo weather station is present in our dataset, which means that only this device can be potentially detectable.

```bash
scripts/unsw_dataset.sh
```

```python
pytest -s tests/test_unsw_dataset.py
```

### Results
Currently our model cannot classify the devices in the dataset (to be percise only one device). This is likely a result of a mismatch in data collection between the unseen and training data.