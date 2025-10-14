# Machine Learning Model Architecture

This document describes the design and relationships between the machine learning components used for IoT device identification. The system is structured around three main layers:

1. **BaseModel** – a reusable model wrapper built around scikit-learn.
2. **Manager** – an orchestrator that manages multiple BaseModel instances.
3. **BinaryModel** and **MultiClassModel** – task-specific managers for different identification strategies.

---

## 1. BaseModel

**Purpose:**  
Encapsulates a single scikit-learn model instance (e.g., RandomForestClassifier). Handles all tasks related to training, evaluation, and persistence of the model.

**Responsibilities:**
- Validate input dataset schema.
- Split data into training and testing sets.
- Apply feature scaling.
- Optionally rebalance datasets.
- Train and tune the model.
- Evaluate the model (accuracy, classification report, confusion matrix).
- Save the trained model and evaluation results to disk.

**Key Methods:**
- `split()` – splits data into train/test sets.
- `scale()` – standardizes features using `StandardScaler`.
- `train()` – trains the model.
- `evaluate()` – computes metrics.
- `save()` – persists the model, test data, and evaluation report.

The BaseModel is independent of any specific task type (binary vs. multiclass). It assumes that the input data has already been preprocessed and labeled.

---

## 2. Manager

**Purpose:**  
The Manager class coordinates multiple BaseModel instances. It provides higher-level workflow management, including preprocessing, training multiple models, and saving global results.

**Responsibilities:**
- Hold references to models, datasets, and evaluation results.
- Train all stored models.
- Save all models and their aggregated metrics.
- Serve as a superclass for BinaryModel and MultiClassModel.

**Key Methods:**
- `train_all()` – iterates through stored records, creates BaseModel objects, and performs training and evaluation.
- `save_all()` – writes models and summaries to disk.
- `create_model_directory()` – creates a date-based directory structure for model outputs.

**Data Structure:**  
The Manager stores each trained model, its metadata, and evaluation results using a lightweight `ModelRecord` dataclass:

```python
from dataclasses import dataclass

@dataclass
class ModelRecord:
    name: str
    model: object
    evaluation: dict
```

---

## 3. BinaryModel

**Purpose:**  
Implements the "one-vs-all" binary classification strategy for device identification.  
Each IoT device gets its own binary classifier that distinguishes between "this device" and "all other devices."

**Responsibilities:**
- Prepare training data for each device:
  - Positive class (current device data).
  - Negative class (samples from other devices).
- Instantiate and train a separate BaseModel for each device.
- Store evaluation results per device.

**Workflow:**
1. Iterate through all preprocessed device CSVs.
2. For each device:
   - Label it as positive (1).
   - Sample negative examples from other devices (label 0).
   - Train a BaseModel on this binary dataset.
3. Save all trained models and generate evaluation summaries.

**Example:**
```python
binary_model = BinaryModel()
binary_model.train_all()
binary_model.save_all()
```

---

## 4. MultiClassModel

**Purpose:**  
Implements a single classifier capable of identifying multiple device types directly (multi-class classification).

**Responsibilities:**
- Combine all preprocessed CSVs into a single labeled dataset.
- Label each record with its corresponding device name.
- Train one BaseModel across all devices.
- Save the trained model and evaluation report.

**Workflow:**
1. Read and combine all device CSVs.
2. Label each record with its device identifier.
3. Pass the combined dataset to a BaseModel instance.
4. Train, evaluate, and save the model.

**Example:**
```python
multi_model = MultiClassModel()
multi_model.preprocess()
multi_model.train_all()
multi_model.save_all()
```

---

## 5. Summary of Design

| Component | Type | Description |
|------------|------|-------------|
| BaseModel | Core class | Wraps a single ML model, handles training and evaluation. |
| Manager | Abstract layer | Coordinates multiple BaseModels. |
| BinaryModel | Derived from Manager | Trains multiple binary classifiers (device vs others). |
| MultiClassModel | Derived from Manager | Trains one multiclass classifier for all devices. |
| ModelRecord | Dataclass | Stores model metadata, evaluation results, and name. |

**Design Philosophy:**  
- **Single Responsibility:** Each class does one clear job.  
- **Reusability:** BaseModel can be reused for different tasks.  
- **Extensibility:** Managers can be extended for other identification strategies.  
- **Traceability:** Every training run creates a structured directory with evaluation logs and model files.
---
