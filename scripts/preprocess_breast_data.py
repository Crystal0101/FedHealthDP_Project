import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os
import logging
import pickle

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 创建必要的目录
def create_directories(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"Created directory: {directory}")

# 读取数据集
def read_data(file_path, sep='\t'):
    try:
        data = pd.read_csv(file_path, sep=sep)
        logging.info(f"Read data from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error reading data from {file_path}: {e}")
        return None

# 数据预处理
def preprocess_data(data):
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    Q1 = data[numeric_columns].quantile(0.25)
    Q3 = data[numeric_columns].quantile(0.75)
    IQR = Q3 - Q1
    data = data[~((data[numeric_columns] < (Q1 - 1.5 * IQR)) | (data[numeric_columns] > (Q3 + 1.5 * IQR))).any(axis=1)]
    return data

# EDA分析
def eda_analysis(data, numeric_columns, title_prefix, output_dir):
    logging.info(f"{title_prefix} 描述性统计:")
    logging.info(data.describe())

    plt.figure(figsize=(12, 10))
    sns.heatmap(data[numeric_columns].corr(), annot=True, cmap='coolwarm')
    plt.title(f'{title_prefix} 相关性矩阵')
    plt.savefig(os.path.join(output_dir, f'{title_prefix}_correlation_matrix.png'), bbox_inches='tight')
    plt.show()

    for column in numeric_columns:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        plt.figure(figsize=(10, 6))
        sns.histplot(data[column], kde=True)
        plt.title(f'{title_prefix} {column} 分布')
        plt.savefig(os.path.join(output_dir, f'{title_prefix}_{column}_distribution.png'), bbox_inches='tight')
        plt.show()

# 主函数
def main():
    directories = [
        '/content/FedHealthDP_Project/FedHealthDP_Project/data/cleaned',
        '/content/FedHealthDP_Project/FedHealthDP_Project/data/processed',
        '/content/FedHealthDP_Project/FedHealthDP_Project/data/split',
        '/content/FedHealthDP_Project/FedHealthDP_Project/data/eda'
    ]
    create_directories(directories)

    breast_cancer_data = read_data('/content/FedHealthDP_Project/FedHealthDP_Project/data/merge/breast_cancer.csv')
    if breast_cancer_data is not None:
        breast_cancer_data = preprocess_data(breast_cancer_data)
        eda_analysis(breast_cancer_data, breast_cancer_data.select_dtypes(include=[np.number]).columns, "breast", '/content/FedHealthDP_Project/FedHealthDP_Project/data/eda')
        breast_cancer_data.to_csv('/content/FedHealthDP_Project/FedHealthDP_Project/data/cleaned/breast_cancer_data_cleaned.csv', index=False, sep='\t')

        breast_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension']
        X_breast = breast_cancer_data[breast_features]
        y_breast = breast_cancer_data['target']

        # 保存特征列列表
        with open('/content/FedHealthDP_Project/FedHealthDP_Project/data/processed/breast_cancer_feature_columns.pkl', 'wb') as f:
            pickle.dump(breast_features, f)

        X_breast.to_csv('/content/FedHealthDP_Project/FedHealthDP_Project/data/processed/breast_cancer_features.csv', index=False)
        y_breast.to_csv('/content/FedHealthDP_Project/FedHealthDP_Project/data/processed/breast_cancer_labels.csv', index=False)

        X_breast_train, X_breast_test, y_breast_train, y_breast_test = train_test_split(X_breast, y_breast, test_size=0.2, random_state=42)

        X_breast_train.to_csv('/content/FedHealthDP_Project/FedHealthDP_Project/data/split/breast_cancer_X_train.csv', index=False)
        X_breast_test.to_csv('/content/FedHealthDP_Project/FedHealthDP_Project/data/split/breast_cancer_X_test.csv', index=False)
        y_breast_train.to_csv('/content/FedHealthDP_Project/FedHealthDP_Project/data/split/breast_cancer_y_train.csv', index=False)
        y_breast_test.to_csv('/content/FedHealthDP_Project/FedHealthDP_Project/data/split/breast_cancer_y_test.csv', index=False)

if __name__ == "__main__":
    main()
