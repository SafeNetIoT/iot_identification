# IoT Identification

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Installation](#installation)  
3. [Quickstart](#quickstart)
3. [Limitations and Further Research](#limitations-and-further-research)  

---

## Project Overview
The aim of this project is to develop a machine learning model to identify an IoT device based on DNS logs from a Wi-Fi access point.  

The repository proposes 2 mathematically equivalent **Random Forest classifiers**, achieving an accuracy of **97%**.
The first proposal is multi class random forest classifier, whereas the second implementation is an array of binary random forest classifiers. The purpose of the second model is to simplify adding classes to the model without retrainining the entire model. 

---

## Installation

### Prerequisites
- **Docker** and **Docker Compose**
- (Optional) **VS Code + Dev Containers** extension

### Clone the repo
```bash
git clone https://github.com/SafeNetIoT/iot_identification.git
cd iot_identification
```

### Start the dev environment
```bash
docker compose up --build
```
Runs the same container used in production and CI.  
Your code is mounted into `/app`, so changes persist.

### VS Code Users
Using VS Code Dev Containers gives you a fully pre-configured, reproducible development environment — with automatic Python setup, debugging, and dependency management — without installing anything locally.\

1. Install the **Dev Containers** extension.  
2. Open the repo in VS Code.  
3. Click **“Reopen in Container”**.  

---

## Quickstart

This project provides two types of ML models:

- **Binary Model** – an array of independent binary classifiers (one per device)
- **Multiclass Model** – a single classifier with one class per device

Architecturally, each individual model is the same (Random Forest).  
The difference lies in how models are **organized**.

The **Binary Model array** exists to make it easy to add new devices without retraining everything from scratch.  
Although it consists of multiple models, we refer to the structure simply as the *Binary Model*.

Because the Binary Model is the default and most commonly used, the rest of this documentation focuses on that architecture.

---

## Dataset Structure

After running the setup steps, a `data/raw/` directory should exist, containing the device-specific data.

Each device must have its own subdirectory, named after the desired class label, and containing one or more `.pcap` files.  
Intermediate directories (e.g., by date) are optional, because the program recursively searches for `.pcap` files.

Example structure:

```
data/raw/
    device_A/
        2024-01-01/session1.pcap
        2024-01-02/session2.pcap
    device_B/
        capture1.pcap
        capture2.pcap
```

> **Important:** Each capture session must follow the “on/off” experimental structure described in  
> https://inria.hal.science/hal-04777603v1/document  
> and sessions must be kept **isolated**.

---

## Training the Binary Model Array (Slow Pipeline)

The `BinaryModel` class supports end-to-end training of all binary Random Forest models:

```python
from src.ml.binary_model import BinaryModel

manager = BinaryModel()
manager.slow_train()
```

This trains one model per device and stores them inside a uniquely generated output directory within `models/`.

To customize the output directory:

```python
manager = BinaryModel(output_directory="your/directory")
manager.slow_train()
```

Training also writes evaluation metrics to `z_evaluation.json`.

---

## Adding a Device to the Binary Model Array (Fast Pipeline)

A new device can be incorporated into the Binary Model without retraining every model.

Before adding a device:

1. Load the existing model array  
2. Set the preprocessing cache  
3. Call `add_device()`

Example:

```python
from src.ml.binary_model import BinaryModel

manager = BinaryModel(
    output_directory="models/2025-11-27/binary_model",
    loading_dir="models/2025-11-27/binary_model"
)

manager.load_model()
manager.set_cache()
manager.add_device("alexa_swan_kettle2", "data/raw/alexa_swan_kettle/")
```

This:

- Extracts features for the new device  
- Trains a new binary classifier  
- Saves the new pickle file  
- Updates `z_evaluation.json`

---

## Model Under Test

To conveniently use a specific trained model, set:

```
model_under_test = "path/to/best/model"
```

in `config.py`.

This reduces boilerplate during prediction.

---

## Prediction Using the Binary Model Array

To load a previously trained model and run predictions on a `.pcap` file:

```python
from src.ml.binary_model import BinaryModel

manager = BinaryModel(loading_dir="models/2025-11-27/binary_model")
manager.predict("your/data.pcap")
```

To predict directly from scapy packets:

```python
from scapy.all import sniff
from src.ml.binary_model import BinaryModel
from config import settings

packets = sniff(iface="eth0", timeout=180)
model = BinaryModel(loading_dir=settings.model_under_test)

device = model.predict(packets)
if device is not None:
    print("Detected device:", device)
```

---

## Limitations and Further Research
- Potential **overfitting** in certain cases.  
- Data drift
- Model degredation with new classes (binary model)