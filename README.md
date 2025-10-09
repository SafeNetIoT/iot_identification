# IoT Identification

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Project Structure](#project-structure)  
3. [Installation](#installation)  
4. [Limitations and Further Research](#limitations-and-further-research)  

---

## Project Overview
The aim of this project is to develop a machine learning model to identify an IoT device based on DNS logs from a Wi-Fi access point.  

The repository proposes 2 mathematically equivalent **Random Forest classifiers**, achieving an accuracy of **97%**.
The first proposal is multi class random forest classifier, whereas the second implementation is an array of binary random forest classifiers. The purpose of the second model is to simplify adding classes to the model without retrainining the entire model. 

---

## Project Structure
```
iot_identification/
├── data/
│   └── raw/ 
├── docs/
│   ├── feature_engineering.md
│   ├── overfitting.md
│   └── time_collection.md
├── src/
│   ├── identification/
│   │   ├── features/
│   │   │   ├── feature_extraction.py
│   │   │   ├── feature_menu.yml
│   │   │   ├── feature_validation.py
│   │   │   ├── features.txt
│   │   │   ├── flow_state.py
│   │   │   ├── flow_structs.py
│   │   │   └── stats.py
│   │   ├── binary_model.py
│   │   ├── collection_time_analysis.py
│   │   ├── dataset_preparation.py
│   │   └── multi_class_model.py
│   ├── lookup/
│   │   ├── lookup_results.csv
│   │   └── mac_lookup.py
│   └── utils.py
├── .gitignore
├── config.json
├── config.py
├── LICENSE
├── pyproject.toml
├── README.md
└── requirements.txt
```

- **data/raw/**: Contains raw DNS logs. These are processed into `data/processed/` (created automatically).  
- **docs/**: Contains extended documentation (e.g., `feature_engineering.md`).  
- **src/**: Source code.  
  - **identification/**:  
    - `feature_menu.yml`: Initial set of considered features.  
    - `feature_extraction.py`: Extracts specified features and creates a CSV for each pcap file in the raw data.  
    - `feature_validation.py`: Evaluates feature stability and correlations, rejecting unhelpful features (automatically and manually). Detailed selection steps are explained in `feature_selection.md`.  
    - `features.txt`: List of final chosen features.  
    - `model.py`: Prepares processed data, selects features, trains, and tests the Random Forest classifier.  
  - **lookup/**:  
    - `mac_lookup.py`: Looks up the device manufacturer based on MAC address.  
    - `lookup_results.csv`: Accuracy results of the API tested on known devices.  
  - `utils.py`: Utility functions.  
- **models/**: Trained models are saved here once generated.  
- **config.json / config.py**: Configuration files.  
- **pyproject.toml / requirements.txt**: Project dependencies.  

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
python src/identification/feature_extraction.py
python src/identification/model.py
```

---

## Limitations and Further Research
- Potential **overfitting** in certain cases.  
- DATA DRIFT