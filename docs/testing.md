# Testing Overview

This document describes the testing strategy for the **BinaryModel** training and inference pipeline.
The tests are designed to ensure both **end-to-end correctness** and **modular reliability** of the system.

---

## Testing Philosophy

The test suite follows two complementary principles:

1. **Integration-first for model validation**
   The goal is to confirm that the **trained model behaves correctly as a whole** — producing accurate predictions, saving artifacts correctly, and handling unseen data gracefully.
   These tests evaluate the **end result** of the model workflow rather than internal implementation details.

2. **Focused unit tests for critical methods**
   Only a small set of key methods — such as data preparation, model training, and device addition — are unit tested.
   These tests mock out dependencies to verify **logic, inputs, and side effects** in isolation.

---

## Test Files

### test_binary_model.py

Integration tests verifying the saving logic and the performance on unseen data. Contains a unit test for handling loading non-existent.


### test_extraction.py

Integration test for the fast extraction pipeline.


### test_fast_pipeline.py

Unit test for ```add_device()``` verifying that it extracts features, labels sessions, samples false classes, and trains/saves one classifier per device.

---
