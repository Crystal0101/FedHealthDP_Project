# FedHealthDP Project

## 项目简介

该项目旨在使用差分隐私的联邦学习技术训练疾病预测模型，数据来自于MIMIC-III数据库。

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

2. 运行数据预处理脚本：

```bash
python3.9 scripts/preprocess.py
```

3. 训练本地模型：

```bash
python3.9 scripts/train.py
```

4. 运行联邦学习过程：

```bash
python3.9 scripts/federated_learning.py
```

5. 评估模型：

```bash
python scripts/evaluate.py
```





