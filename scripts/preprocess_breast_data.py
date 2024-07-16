import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os

# 指定需要创建的目录列表
directories = [
    '/content/FedHealthDP_Project/FedHealthDP_Project/data/cleaned',
    '/content/FedHealthDP_Project/FedHealthDP_Project/data/processed',
    '/content/FedHealthDP_Project/FedHealthDP_Project/data/split',
    '/content/FedHealthDP_Project/FedHealthDP_Project/data/eda'  # 添加EDA图表保存目录
]

# 遍历目录列表，确保每个目录都被创建
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)

# 读取数据集，指定分隔符为制表符
breast_cancer_data = pd.read_csv('/content/FedHealthDP_Project/FedHealthDP_Project/data/merge/breast_cancer.csv', sep='\t')

# Kaggle数据集处理

# 处理缺失值
breast_cancer_data.dropna(inplace=True)

# 处理重复数据
breast_cancer_data.drop_duplicates(inplace=True)

# 检查和处理离群值和异常值
# 乳腺癌数据集
numeric_columns_breast = breast_cancer_data.select_dtypes(include=[np.number]).columns
Q1 = breast_cancer_data[numeric_columns_breast].quantile(0.25)
Q3 = breast_cancer_data[numeric_columns_breast].quantile(0.75)
IQR = Q3 - Q1
breast_cancer_data = breast_cancer_data[~((breast_cancer_data[numeric_columns_breast] < (Q1 - 1.5 * IQR)) | (breast_cancer_data[numeric_columns_breast] > (Q3 + 1.5 * IQR))).any(axis=1)]

# EDA过程
def eda_analysis(data, numeric_columns, title_prefix):
    # 描述性统计
    print(f"{title_prefix} 描述性统计:")
    print(data.describe())

    # 相关性矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(data[numeric_columns].corr(), annot=True, cmap='coolwarm')
    plt.title(f'{title_prefix} 相关性矩阵')
    plt.savefig(f'/content/FedHealthDP_Project/FedHealthDP_Project/data/eda/{title_prefix}_correlation_matrix.png', bbox_inches='tight')  # 保存图表
    plt.show()

    # 每个数值变量的分布
    for column in numeric_columns:
        plt.figure(figsize=(10, 5))
        sns.histplot(data[column], kde=True)
        plt.title(f'{title_prefix} - {column} 的分布')
        plt.show()

# 进行EDA分析
eda_analysis(breast_cancer_data, numeric_columns_breast, "breast_correlation_matrix.png")

# 保存处理后的Kaggle数据集
breast_cancer_data.to_csv('/content/FedHealthDP_Project/FedHealthDP_Project/data/cleaned/breast_cancer_data_cleaned.csv', index=False, sep='\t')

# 读取清理后的Kaggle数据集，指定分隔符为制表符
breast_cancer_data = pd.read_csv('/content/FedHealthDP_Project/FedHealthDP_Project/data/cleaned/breast_cancer_data_cleaned.csv', sep='\t')

# 清理列名，去除可能的前后空格
breast_cancer_data.columns = breast_cancer_data.columns.str.strip()

# 乳腺癌特征选择
breast_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 
                   'mean smoothness', 'mean compactness', 'mean concavity', 
                   'mean concave points', 'mean symmetry', 'mean fractal dimension']
X_breast = breast_cancer_data[breast_features]
y_breast = breast_cancer_data['target']

# 存储乳腺癌数据集的特征和标签
X_breast.to_csv('/content/FedHealthDP_Project/FedHealthDP_Project/data/processed/breast_cancer_features.csv', index=False)
y_breast.to_csv('/content/FedHealthDP_Project/FedHealthDP_Project/data/processed/breast_cancer_labels.csv', index=False)

# 读取乳腺癌特征和标签数据
X_breast = pd.read_csv('/content/FedHealthDP_Project/FedHealthDP_Project/data/processed/breast_cancer_features.csv')
y_breast = pd.read_csv('/content/FedHealthDP_Project/FedHealthDP_Project/data/processed/breast_cancer_labels.csv')

# 分割乳腺癌数据集
X_breast_train, X_breast_test, y_breast_train, y_breast_test = train_test_split(X_breast, y_breast, test_size=0.2, random_state=42)

# 保存分割后的数据
X_breast_train.to_csv('/content/FedHealthDP_Project/FedHealthDP_Project/data/split/breast_cancer_X_train.csv', index=False)
X_breast_test.to_csv('/content/FedHealthDP_Project/FedHealthDP_Project/data/split/breast_cancer_X_test.csv', index=False)
y_breast_train.to_csv('/content/FedHealthDP_Project/FedHealthDP_Project/data/split/breast_cancer_y_train.csv', index=False)
y_breast_test.to_csv('/content/FedHealthDP_Project/FedHealthDP_Project/data/split/breast_cancer_y_test.csv', index=False)
