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

## Limitations and Further Research
- Potential **overfitting** in certain cases.  
- Data drift
- Model degredation with new classes (binary model)