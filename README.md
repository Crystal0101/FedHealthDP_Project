# FedHealthDP Project

## 项目简介

The FedHealthDP Project aims to train disease prediction models using federated learning with differential privacy, leveraging data from the MIMIC-III and Kaggle database.


## 目录结构

```plaintext
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
```

## 主要功能模块

1. 数据预处理 (scripts/preprocess.py):

从数据源加载数据。
处理缺失值、标准化和特征工程。

2. 模型定义 (models/model.py):

定义深度学习模型结构。
支持联邦学习和差分隐私的模型架构。

3. 差分隐私优化器 (models/dp_optimizer.py):

实现差分隐私的优化算法。
确保在训练过程中保护数据隐私。

4. 联邦学习实现 (scripts/federated_learning.py):

实现联邦学习的核心逻辑。
管理多个客户端的模型更新和聚合。

5. 训练脚本 (scripts/train.py):

训练模型，包括本地训练和联邦训练。
支持配置化训练参数。

6. 评估脚本 (scripts/evaluate.py):

评估模型性能，生成报告和可视化结果。
7. 工具函数 (utils/data_utils.py 和 utils/eval_utils.py):

常用数据处理和评估函数，供其他模块调用。

## 使用方法

1. 安装依赖：

```bash
pip install -r requirements.txt
```

