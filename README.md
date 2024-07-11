# FedHealthDP Project

## 项目简介

The FedHealthDP Project aims to train disease prediction models using federated learning with differential privacy, leveraging data from the MIMIC-III database.


## 目录结构

- `data/`：包含所有数据文件
- `models/`：模型定义和差分隐私优化器
- `scripts/`：数据预处理、训练、评估和联邦学习脚本
- `utils/`：辅助函数和工具类代码
- `requirements.txt`：项目依赖的Python库
- `config.yaml`：项目配置文件

## 使用方法

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. Environment Setup

```bash
export DATA_DIR=/path/to/data
export MODEL_DIR=/path/to/models
```

3. Running the Project
Data Preprocessing:

```bash
python scripts/preprocess_data.py --input $DATA_DIR/raw --output $DATA_DIR/processed
```

Local Model Training:

```bash
python scripts/train_local_model.py --data $DATA_DIR/processed --model $MODEL_DIR/local_model
```

Federated Learning:

```bash
python scripts/federated_learning.py --data $DATA_DIR/processed --model $MODEL_DIR/federated_model
```
Evaluation:

```bash
python scripts/evaluate_model.py --model $MODEL_DIR/federated_model --data $DATA_DIR/proc
```

