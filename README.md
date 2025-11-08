# IoT Identification

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Installation](#installation)  
3. [Limitations and Further Research](#limitations-and-further-research)  

---

## Project Overview
The aim of this project is to develop a machine learning model to identify an IoT device based on DNS logs from a Wi-Fi access point.  

The repository proposes 2 mathematically equivalent **Random Forest classifiers**, achieving an accuracy of **97%**.
The first proposal is multi class random forest classifier, whereas the second implementation is an array of binary random forest classifiers. The purpose of the second model is to simplify adding classes to the model without retrainining the entire model. 

---

## Installation

### 1. Create a virtual environment  
**MacOS / Linux**:
```bash
python -m venv venv
source venv/bin/activate
```

**Windows**:
```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install the requirements  
```bash
pip install -r requirements.txt
```

### 3. Install the project in editable mode  
```bash
pip install -e .
```

### 4. Run the model with the default features  
```bash
python -m src.identification.feature_extraction.py
python -m src.identification.binary_model.py
```

---

## Limitations and Further Research
- Potential **overfitting** in certain cases.  
- Data drift
- Model degredation with new classes (binary model)