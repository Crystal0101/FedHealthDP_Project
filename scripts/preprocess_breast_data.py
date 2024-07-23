import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 创建必要的目录
output_dirs = [
    '/content/FedHealthDP_Project/data/cleaned',
    '/content/FedHealthDP_Project/data/processed',
    '/content/FedHealthDP_Project/data/eda',
    '/content/FedHealthDP_Project/data/split'
]
for dir_path in output_dirs:
    os.makedirs(dir_path, exist_ok=True)

def read_data(file_path):
    try:
        data = pd.read_csv(file_path)
        logging.info(f"数据读取成功: {file_path}")
        return data
    except Exception as e:
        logging.error(f"数据读取失败: {e}")
        raise

def clean_data(data):
    try:
        data = data.drop(columns=['id'])
        data = data.dropna()  # 删除缺失值行，如果数据量大可以使用填充方法
        logging.info("数据清洗成功")
        return data
    except Exception as e:
        logging.error(f"数据清洗失败: {e}")
        raise

def eda_analysis(data, dataset_index):
    try:
        # 描述性统计
        logging.info("数据描述性统计:")
        logging.info(data.describe())

        # 绘制目标变量的分布
        plt.figure(figsize=(10, 6))
        sns.countplot(data['diagnosis'])
        plt.title('Diagnosis Distribution', fontsize=16)
        plt.xlabel('Diagnosis', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.savefig(f'/content/FedHealthDP_Project/data/eda/diagnosis_distribution_{dataset_index}.png')
        plt.close()

        # 绘制数值特征的直方图
        data.hist(bins=20, figsize=(20, 15))
        plt.tight_layout()
        plt.savefig(f'/content/FedHealthDP_Project/data/eda/numerical_features_histogram_{dataset_index}.png')
        plt.close()

        # 绘制数值特征的箱线图
        plt.figure(figsize=(20, 15))
        sns.boxplot(data=data.select_dtypes(include=[np.number]))
        plt.title('Numerical Features Boxplot', fontsize=16)
        plt.xticks(rotation=90, fontsize=12)
        plt.tight_layout()
        plt.savefig(f'/content/FedHealthDP_Project/data/eda/numerical_features_boxplot_{dataset_index}.png')
        plt.close()

        # 相关矩阵热力图
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        corr_matrix = data[numeric_columns].corr()
        plt.figure(figsize=(16, 12))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=.5)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        plt.title('Correlation Matrix Heatmap', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'/content/FedHealthDP_Project/data/eda/breast_correlation_heatmap_{dataset_index}.png')
        plt.close()
        logging.info("EDA分析完成")
    except Exception as e:
        logging.error(f"EDA分析失败: {e}")
        raise

def feature_engineering(data):
    try:
        # 标签编码
        le = LabelEncoder()
        data['diagnosis'] = le.fit_transform(data['diagnosis'])
        
        # 标准化数值特征
        scaler = StandardScaler()
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
        
        logging.info("特征工程完成")
        return data
    except Exception as e:
        logging.error(f"特征工程失败: {e}")
        raise

def feature_selection(data, target_column='diagnosis', k=10):
    try:
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # 选择k个最佳特征
        selector = SelectKBest(score_func=f_classif, k=k)
        X_new = selector.fit_transform(X, y)
        
        # 保留选择的特征列名
        selected_features = X.columns[selector.get_support()]
        pd.Series(selected_features).to_csv('/content/FedHealthDP_Project/data/split/breast_features.csv', index=False)
        X = pd.DataFrame(X_new, columns=selected_features)
        
        # 重新添加目标列
        data = pd.concat([X, y.reset_index(drop=True)], axis=1)
        
        logging.info("特征选择完成")
        return data, selected_features
    except Exception as e:
        logging.error(f"特征选择失败: {e}")
        raise

def balance_data(data, target_column='diagnosis'):
    try:
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        smote_enn = SMOTEENN(random_state=42)
        X_resampled, y_resampled = smote_enn.fit_resample(X, y)
        
        balanced_data = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=target_column)], axis=1)
        
        logging.info("数据平衡处理完成")
        return balanced_data
    except Exception as e:
        logging.error(f"数据平衡处理失败: {e}")
        raise

def split_data(data, selected_features, target_column='diagnosis', test_size=0.2, random_state=42):
    try:
        X = data[selected_features]
        y = data[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # 保存分割后的数据
        X_train.to_csv('/content/FedHealthDP_Project/data/split/breast_cancer_X_train.csv', index=False)
        X_test.to_csv('/content/FedHealthDP_Project/data/split/breast_cancer_X_test.csv', index=False)
        y_train.to_csv('/content/FedHealthDP_Project/data/split/breast_cancer_y_train.csv', index=False, header=True)
        y_test.to_csv('/content/FedHealthDP_Project/data/split/breast_cancer_y_test.csv', index=False, header=True)
        
        logging.info("数据分割完成")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"数据分割失败: {e}")
        raise

def main():
    try:
        # 数据读取
        data_files = [
            '/content/FedHealthDP_Project/data/kaggle/breast_cancer_0.csv',
            '/content/FedHealthDP_Project/data/kaggle/breast_cancer_1.csv',
            '/content/FedHealthDP_Project/data/kaggle/breast_cancer_2.csv'
        ]
        
        for i, file in enumerate(data_files):
            data = read_data(file)
            
            # 数据清洗
            data = clean_data(data)
            
            # 暂存目标列
            y = data['diagnosis'].copy()
            
            # EDA分析
            eda_analysis(data, i)
            
            # 特征工程
            data = feature_engineering(data)
            
            # 恢复目标列
            data['diagnosis'] = y
            
            # 特征选择
            data, selected_features = feature_selection(data)
            
            # 数据平衡
            data = balance_data(data)
            
            # 数据分割
            X_train, X_test, y_train, y_test = split_data(data, selected_features)
            
            # 保存处理后的数据，增加后缀以区分
            processed_data_path_train = f'/content/FedHealthDP_Project/data/processed/breast_cleaned_data_train_{i}.csv'
            processed_data_path_test = f'/content/FedHealthDP_Project/data/processed/breast_cleaned_data_test_{i}.csv'
            data.to_csv(processed_data_path_train, index=False)
            X_train.to_csv(f'/content/FedHealthDP_Project/data/split/breast_cancer_X_train_{i}.csv', index=False)
            X_test.to_csv(f'/content/FedHealthDP_Project/data/split/breast_cancer_X_test_{i}.csv', index=False)
            y_train.to_csv(f'/content/FedHealthDP_Project/data/split/breast_cancer_y_train_{i}.csv', index=False, header=True)
            y_test.to_csv(f'/content/FedHealthDP_Project/data/split/breast_cancer_y_test_{i}.csv', index=False, header=True)
            
            logging.info(f"数据处理和保存完成 for dataset {i}")
    except Exception as e:
        logging.error(f"数据处理流程失败: {e}")

if __name__ == "__main__":
    main()
