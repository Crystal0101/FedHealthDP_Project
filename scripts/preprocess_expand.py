import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 创建必要的目录
os.makedirs('/content/FedHealthDP_Project/data/cleaned', exist_ok=True)
os.makedirs('/content/FedHealthDP_Project/data/processed', exist_ok=True)
os.makedirs('/content/FedHealthDP_Project/data/eda', exist_ok=True)
os.makedirs('/content/FedHealthDP_Project/data/split', exist_ok=True)

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

def eda_analysis(data):
    try:
        # 描述性统计
        logging.info("数据描述性统计:")
        logging.info(data.describe())
        
        # 相关矩阵热力图
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        corr_matrix = data[numeric_columns].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix Heatmap')
        plt.savefig('/content/FedHealthDP_Project/data/eda/breast_correlation_heatmap.png')
        plt.show()
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
        selected_features.to_series().to_csv('/content/FedHealthDP_Project/data/split/breast_features.csv', index=False)
        X = pd.DataFrame(X_new, columns=selected_features)
        
        # 重新添加目标列
        data = pd.concat([X, y.reset_index(drop=True)], axis=1)
        
        logging.info("特征选择完成")
        return data, selected_features
    except Exception as e:
        logging.error(f"特征选择失败: {e}")
        raise

def visualize_data(data):
    try:
        sns.pairplot(data, hue='diagnosis')
        plt.savefig('/content/FedHealthDP_Project/data/eda/breast_pairplot.png')
        plt.show()
        logging.info("数据可视化完成")
    except Exception as e:
        logging.error(f"数据可视化失败: {e}")
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
        data = read_data('/content/FedHealthDP_Project/data/kaggle/expanded_breast_cancer.csv')
        
        # 数据清洗
        data = clean_data(data)
        
        # 暂存目标列
        y = data['diagnosis'].copy()
        
        # EDA分析
        eda_analysis(data)
        
        # 特征工程
        data = feature_engineering(data)
        
        # 恢复目标列
        data['diagnosis'] = y
        
        # 特征选择
        data, selected_features = feature_selection(data)
        
        # 数据可视化
        visualize_data(data)
        
        # 数据分割
        X_train, X_test, y_train, y_test = split_data(data, selected_features)
        
        # 保存处理后的数据
        processed_data_path = '/content/FedHealthDP_Project/data/processed/breast_cleaned_data.csv'
        data.to_csv(processed_data_path, index=False)
        
        logging.info("数据处理和保存完成")
    except Exception as e:
        logging.error(f"数据处理流程失败: {e}")

if __name__ == "__main__":
    main()
