# FedHealthDP Project

## 项目简介

The FedHealthDP Project aims to train disease prediction models using federated learning with differential privacy, leveraging data from the MIMIC-III database.


## 目录结构

FedHealthDP_Project/
│
├── README.md              # 项目简介和说明
├── config.yaml            # 配置文件
├── requirements.txt       # 依赖项列表
├── main.py                # 项目入口文件
│
├── data/                  # 数据目录
│   ├── kaggle/            # Kaggle数据集
│   └── mimiciii/          # MIMIC-III数据集
│
├── models/                # 模型目录
│   ├── __init__.py        # 初始化文件
│   ├── dp_optimizer.py    # 差分隐私优化器实现
│   └── model.py           # 模型定义
│
├── scripts/               # 脚本目录
│   ├── preprocess.py      # 数据预处理脚本
│   ├── train.py           # 训练脚本
│   ├── federated_learning.py # 联邦学习实现脚本
│   └── evaluate.py        # 模型评估脚本
│
├── test/                  # 测试目录
│   ├── test_evaluate.py   # 评估测试
│   ├── test_federated.py  # 联邦学习测试
│   ├── test_preprocess.py # 预处理测试
│   └── test_train.py      # 训练测试
│
└── utils/                 # 工具目录
    ├── __init__.py        # 初始化文件
    ├── data_utils.py      # 数据处理工具
    └── eval_utils.py      # 评估工具

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

