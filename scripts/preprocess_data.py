import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os

# 指定需要创建的目录列表
directories = [
    '/content/FedHealthDP_Project/data/cleaned',
    '/content/FedHealthDP_Project/data/processed',
    '/content/FedHealthDP_Project/data/split'
]

# 遍历目录列表，确保每个目录都被创建
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)

# 读取数据集
# 读取Kaggle数据集
breast_cancer_data = pd.read_csv('/content/FedHealthDP_Project/data/kaggle/breast_cancer.csv')
lung_cancer_data = pd.read_csv('/content/FedHealthDP_Project/data/kaggle/lung_cancer.csv')
prostate_cancer_data = pd.read_csv('/content/FedHealthDP_Project/data/kaggle/prostate_cancer.csv')

# Kaggle数据集处理

# 处理缺失值
breast_cancer_data.dropna(inplace=True)
lung_cancer_data.replace('?', np.nan, inplace=True)
lung_cancer_data.fillna(lung_cancer_data.median(numeric_only=True), inplace=True)
prostate_cancer_data.dropna(inplace=True)

# 处理重复数据
breast_cancer_data.drop_duplicates(inplace=True)
lung_cancer_data.drop_duplicates(inplace=True)
prostate_cancer_data.drop_duplicates(inplace=True)

# 检查和处理离群值和异常值
# 乳腺癌数据集
numeric_columns_breast = breast_cancer_data.select_dtypes(include=[np.number]).columns
Q1 = breast_cancer_data[numeric_columns_breast].quantile(0.25)
Q3 = breast_cancer_data[numeric_columns_breast].quantile(0.75)
IQR = Q3 - Q1
breast_cancer_data = breast_cancer_data[~((breast_cancer_data[numeric_columns_breast] < (Q1 - 1.5 * IQR)) | (breast_cancer_data[numeric_columns_breast] > (Q3 + 1.5 * IQR))).any(axis=1)]

# 肺癌数据集
numeric_columns_lung = lung_cancer_data.select_dtypes(include=[np.number]).columns
Q1 = lung_cancer_data[numeric_columns_lung].quantile(0.25)
Q3 = lung_cancer_data[numeric_columns_lung].quantile(0.75)
IQR = Q3 - Q1
lung_cancer_data = lung_cancer_data[~((lung_cancer_data[numeric_columns_lung] < (Q1 - 1.5 * IQR)) | (lung_cancer_data[numeric_columns_lung] > (Q3 + 1.5 * IQR))).any(axis=1)]

# 前列腺癌数据集
numeric_columns_prostate = prostate_cancer_data.select_dtypes(include=[np.number]).columns
Q1 = prostate_cancer_data[numeric_columns_prostate].quantile(0.25)
Q3 = prostate_cancer_data[numeric_columns_prostate].quantile(0.75)
IQR = Q3 - Q1
prostate_cancer_data = prostate_cancer_data[~((prostate_cancer_data[numeric_columns_prostate] < (Q1 - 1.5 * IQR)) | (prostate_cancer_data[numeric_columns_prostate] > (Q3 + 1.5 * IQR))).any(axis=1)]

# 评估变量分布
for column in numeric_columns_breast:
    plt.figure(figsize=(10, 5))
    sns.histplot(breast_cancer_data[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()

for column in numeric_columns_lung:
    plt.figure(figsize=(10, 5))
    sns.histplot(lung_cancer_data[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()

for column in numeric_columns_prostate:
    plt.figure(figsize=(10, 5))
    sns.histplot(prostate_cancer_data[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()

# 保存处理后的Kaggle数据集
breast_cancer_data.to_csv('/content/FedHealthDP_Project/data/cleaned/breast_cancer_data_cleaned.csv', index=False)
lung_cancer_data.to_csv('/content/FedHealthDP_Project/data/cleaned/lung_cancer_data_cleaned.csv', index=False)
prostate_cancer_data.to_csv('/content/FedHealthDP_Project/data/cleaned/prostate_cancer_data_cleaned.csv', index=False)


# 读取清理后的Kaggle数据集，指定分隔符为制表符
breast_cancer_data = pd.read_csv('/content/FedHealthDP_Project/data/cleaned/breast_cancer_data_cleaned.csv', sep='\t')
lung_cancer_data = pd.read_csv('/content/FedHealthDP_Project/data/cleaned/lung_cancer_data_cleaned.csv', sep='\t')
prostate_cancer_data = pd.read_csv('/content/FedHealthDP_Project/data/cleaned/prostate_cancer_data_cleaned.csv', sep='\t')

# 清理列名，去除可能的前后空格
breast_cancer_data.columns = breast_cancer_data.columns.str.strip()
lung_cancer_data.columns = lung_cancer_data.columns.str.strip()
prostate_cancer_data.columns = prostate_cancer_data.columns.str.strip()

# 乳腺癌特征选择
breast_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
                   'smoothness_mean', 'compactness_mean', 'concavity_mean', 
                   'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
X_breast = breast_cancer_data[breast_features]
y_breast = breast_cancer_data['diagnosis']

# 肺癌特征选择
lung_features = ['AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 
                 'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE', 
                 'ALLERGY', 'WHEEZING', 'ALCOHOL CONSUMING', 
                 'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 
                 'CHEST PAIN']
X_lung = lung_cancer_data[lung_features]
y_lung = lung_cancer_data['LUNG_CANCER'].apply(lambda x: 1 if x == 'YES' else 0)

# 前列腺癌特征选择
prostate_features = ['radius', 'texture', 'perimeter', 'area', 
                     'smoothness', 'compactness', 'symmetry', 
                     'fractal_dimension']
X_prostate = prostate_cancer_data[prostate_features]
y_prostate = prostate_cancer_data['diagnosis_result'].apply(lambda x: 1 if x == 'M' else 0)

# 存储乳腺癌数据集的特征和标签
X_breast.to_csv('/content/FedHealthDP_Project/data/processed/breast_cancer_features.csv', index=False)
y_breast.to_csv('/content/FedHealthDP_Project/data/processed/breast_cancer_labels.csv', index=False)

# 存储肺癌数据集的特征和标签
X_lung.to_csv('/content/FedHealthDP_Project/data/processed/lung_cancer_features.csv', index=False)
y_lung.to_csv('/content/FedHealthDP_Project/data/processed/lung_cancer_labels.csv', index=False)

# 存储前列腺癌数据集的特征和标签
X_prostate.to_csv('/content/FedHealthDP_Project/data/processed/prostate_cancer_features.csv', index=False)
y_prostate.to_csv('/content/FedHealthDP_Project/data/processed/prostate_cancer_labels.csv', index=False)

# 读取乳腺癌特征和标签数据
X_breast = pd.read_csv('/content/FedHealthDP_Project/data/processed/breast_cancer_features.csv')
y_breast = pd.read_csv('/content/FedHealthDP_Project/data/processed/breast_cancer_labels.csv')

# 分割乳腺癌数据集
X_breast_train, X_breast_test, y_breast_train, y_breast_test = train_test_split(X_breast, y_breast, test_size=0.2, random_state=42)

# 保存分割后的数据
X_breast_train.to_csv('/content/FedHealthDP_Project/data/split/breast_cancer_X_train.csv', index=False)
X_breast_test.to_csv('/content/FedHealthDP_Project/data/split/breast_cancer_X_test.csv', index=False)
y_breast_train.to_csv('/content/FedHealthDP_Project/data/split/breast_cancer_y_train.csv', index=False)
y_breast_test.to_csv('/content/FedHealthDP_Project/data/split/breast_cancer_y_test.csv', index=False)

# 读取肺癌特征和标签数据
X_lung = pd.read_csv('/content/FedHealthDP_Project/data/processed/lung_cancer_features.csv')
y_lung = pd.read_csv('/content/FedHealthDP_Project/data/processed/lung_cancer_labels.csv')

# 分割肺癌数据集
X_lung_train, X_lung_test, y_lung_train, y_lung_test = train_test_split(X_lung, y_lung, test_size=0.2, random_state=42)

# 保存分割后的数据
X_lung_train.to_csv('/content/FedHealthDP_Project/data/split/lung_cancer_X_train.csv', index=False)
X_lung_test.to_csv('/content/FedHealthDP_Project/data/split/lung_cancer_X_test.csv', index=False)
y_lung_train.to_csv('/content/FedHealthDP_Project/data/split/lung_cancer_y_train.csv', index=False)
y_lung_test.to_csv('/content/FedHealthDP_Project/data/split/lung_cancer_y_test.csv', index=False)

# 读取前列腺癌特征和标签数据
X_prostate = pd.read_csv('/content/FedHealthDP_Project/data/processed/prostate_cancer_features.csv')
y_prostate = pd.read_csv('/content/FedHealthDP_Project/data/processed/prostate_cancer_labels.csv')

# 分割前列腺癌数据集
X_prostate_train, X_prostate_test, y_prostate_train, y_prostate_test = train_test_split(X_prostate, y_prostate, test_size=0.2, random_state=42)

# 保存分割后的数据
X_prostate_train.to_csv('/content/FedHealthDP_Project/data/split/prostate_cancer_X_train.csv', index=False)
X_prostate_test.to_csv('/content/FedHealthDP_Project/data/split/prostate_cancer_X_test.csv', index=False)
y_prostate_train.to_csv('/content/FedHealthDP_Project/data/split/prostate_cancer_y_train.csv', index=False)
y_prostate_test.to_csv('/content/FedHealthDP_Project/data/split/prostate_cancer_y_test.csv', index=False)
