import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
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

# 特征选择
def feature_selection(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': importances})
    feature_importance = feature_importance.sort_values(by='importance', ascending=False)
    return feature_importance

# 特征工程
def feature_engineering(data, target_column):
    # 分离目标列
    target = data[target_column]
    data = data.drop(columns=[target_column])
    
    # One-Hot Encoding for categorical features
    categorical_columns = data.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        encoder = OneHotEncoder(sparse=False, drop='first')
        encoded_features = encoder.fit_transform(data[categorical_columns])
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))
        data = data.drop(columns=categorical_columns).reset_index(drop=True)
        data = pd.concat([data, encoded_df], axis=1)
    
    # Standardization for numerical features
    scaler = StandardScaler()
    numerical_columns = data.select_dtypes(include=[np.number]).columns
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    
    # 重新加入目标列
    data[target_column] = target.values
    
    return data

# 降维
def apply_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X)
    principal_df = pd.DataFrame(data=principal_components, columns=[f'PC{i}' for i in range(1, n_components + 1)])
    return principal_df, pca

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

    lung_cancer_data = read_data('/content/FedHealthDP_Project/FedHealthDP_Project/data/kaggle/lung_cancer.csv')
    if lung_cancer_data is not None:
        lung_cancer_data = preprocess_data(lung_cancer_data)
        eda_analysis(lung_cancer_data, lung_cancer_data.select_dtypes(include=[np.number]).columns, "lung_cancer", '/content/FedHealthDP_Project/FedHealthDP_Project/data/eda')
        
        # 特征工程
        lung_cancer_data = feature_engineering(lung_cancer_data, 'LUNG_CANCER')
        
        # 确保在特征工程后 'LUNG_CANCER' 列仍然存在
        if 'LUNG_CANCER' not in lung_cancer_data.columns:
            logging.error("Column 'LUNG_CANCER' is missing after feature engineering.")
            return

        # 保存清洗和标准化后的数据
        lung_cancer_data.to_csv('/content/FedHealthDP_Project/FedHealthDP_Project/data/cleaned/lung_cancer_cleaned.csv', index=False)

        # 特征选择
        X = lung_cancer_data.drop(columns=['LUNG_CANCER'])
        y = lung_cancer_data['LUNG_CANCER']
        feature_importance = feature_selection(X, y)
        logging.info(f"Feature importance:\n{feature_importance}")

        # 选择最重要的前10个特征
        selected_features = feature_importance['feature'].head(10).tolist()
        X = X[selected_features]

        # 保存特征列列表
        with open('/content/FedHealthDP_Project/FedHealthDP_Project/data/processed/lung_cancer_feature_columns.pkl', 'wb') as f:
            pickle.dump(selected_features, f)

        # 降维
        principal_df, pca = apply_pca(X, n_components=2)
        principal_df['LUNG_CANCER'] = y.values

        # 可视化 PCA 结果
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=principal_df, x='PC1', y='PC2', hue='LUNG_CANCER', palette='viridis')
        plt.title('PCA of Lung Cancer Data')
        plt.savefig('/content/FedHealthDP_Project/FedHealthDP_Project/data/eda/pca_lung_cancer.png')
        plt.show()

        # 分割数据集为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 保存分割后的数据集
        X_train.to_csv('/content/FedHealthDP_Project/FedHealthDP_Project/data/split/lung_cancer_X_train.csv', index=False)
        X_test.to_csv('/content/FedHealthDP_Project/FedHealthDP_Project/data/split/lung_cancer_X_test.csv', index=False)
        y_train.to_csv('/content/FedHealthDP_Project/FedHealthDP_Project/data/split/lung_cancer_y_train.csv', index=False)
        y_test.to_csv('/content/FedHealthDP_Project/FedHealthDP_Project/data/split/lung_cancer_y_test.csv', index=False)

if __name__ == "__main__":
    main()
