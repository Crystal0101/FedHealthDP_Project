# FedHealthDP Project

## Project Overview

The FedHealthDP Project aims to train disease prediction models using federated learning with differential privacy. This approach leverages data from the MIMIC-III and Kaggle databases to create robust and privacy-preserving predictive models for cancer risk assessment.

## Main Functional Modules

1. **Data Preprocessing (scripts/preprocess.py)**:
    - Load data from sources
    - Handle missing values, normalization, and feature engineering

2. **Model Definition (models/model.py)**:
    - Define deep learning model structure
    - Support federated learning and differential privacy architecture

3. **Differential Privacy Optimizer (models/dp_optimizer.py)**:
    - Implement differential privacy optimization algorithms
    - Ensure data privacy during training

4. **Federated Learning Implementation (scripts/federated_learning.py)**:
    - Core logic for federated learning
    - Manage model updates and aggregation across multiple clients

5. **Training Script (scripts/train.py)**:
    - Train models, including local and federated training
    - Support configurable training parameters

6. **Evaluation Script (scripts/evaluate.py)**:
    - Evaluate model performance
    - Generate reports and visualization results

7. **Utility Functions (utils/data_utils.py and utils/eval_utils.py)**:
    - Common data processing and evaluation functions for use by other modules

## Directory Structure

```plaintext
FedHealthDP_Project/
│
├── README.md              # Project introduction and instructions
├── config.yaml            # Configuration file
├── requirements.txt       # List of dependencies
├── main.py                # Project entry point
│
├── data/                  # Data directory
│   ├── kaggle/            # Kaggle datasets
│   └── mimiciii/          # MIMIC-III datasets
│
├── models/                # Models directory
│   ├── __init__.py        # Initialization file
│   ├── dp_optimizer.py    # Differential privacy optimizer implementation
│   └── model.py           # Model definition
│
├── scripts/               # Scripts directory
│   ├── preprocess.py      # Data preprocessing script
│   ├── train.py           # Training script
│   ├── federated_learning.py # Federated learning implementation script
│   └── evaluate.py        # Model evaluation script
│
├── test/                  # Test directory
│   ├── test_evaluate.py   # Evaluation tests
│   ├── test_federated.py  # Federated learning tests
│   ├── test_preprocess.py # Preprocessing tests
│   └── test_train.py      # Training tests
│
└── utils/                 # Utilities directory
    ├── __init__.py        # Initialization file
    ├── data_utils.py      # Data processing utilities
    └── eval_utils.py      # Evaluation utilities

