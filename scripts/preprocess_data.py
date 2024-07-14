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
breast_cancer_data = pd.read_csv('/content/FedHealthDP_Project/data/kaggle/breast_cancer_data.csv')
lung_cancer_data = pd.read_csv('/content/FedHealthDP_Project/data/kaggle/lung_cancer_data.csv')
prostate_cancer_data = pd.read_csv('/content/FedHealthDP_Project/data/kaggle/prostate_cancer_data.csv')

# 读取MIMIC-III数据集
admissions = pd.read_csv('/content/FedHealthDP_Project/data/mimiciii/fixed_ADMISSIONS.csv')
chartevents = pd.read_csv('/content/FedHealthDP_Project/data/mimiciii/fixed_CHARTEVENTS.csv')
diagnoses_icd = pd.read_csv('/content/FedHealthDP_Project/data/mimiciii/fixed_DIAGNOSES_ICD.csv')
labevents = pd.read_csv('/content/FedHealthDP_Project/data/mimiciii/fixed_LABEVENTS.csv')
patients = pd.read_csv('/content/FedHealthDP_Project/data/mimiciii/fixed_PATIENTS.csv')
prescriptions = pd.read_csv('/content/FedHealthDP_Project/data/mimiciii/fixed_PRESCRIPTIONS.csv')
procedures_icd = pd.read_csv('/content/FedHealthDP_Project/data/mimiciii/fixed_PROCEDURES_ICD.csv')

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

# MIMIC-III数据集处理

# 处理缺失值
admissions.dropna(inplace=True)
chartevents.dropna(inplace=True)
diagnoses_icd.dropna(inplace=True)
labevents.dropna(inplace=True)
patients.dropna(inplace=True)
prescriptions.dropna(inplace=True)
procedures_icd.dropna(inplace=True)

# 处理重复数据
admissions.drop_duplicates(inplace=True)
chartevents.drop_duplicates(inplace=True)
diagnoses_icd.drop_duplicates(inplace=True)
labevents.drop_duplicates(inplace=True)
patients.drop_duplicates(inplace=True)
prescriptions.drop_duplicates(inplace=True)
procedures_icd.drop_duplicates(inplace=True)

# 检查和处理离群值和异常值
def remove_outliers(df, numeric_columns):
    Q1 = df[numeric_columns].quantile(0.25)
    Q3 = df[numeric_columns].quantile(0.75)
    IQR = Q3 - Q1
    return df[~((df[numeric_columns] < (Q1 - 1.5 * IQR)) | (df[numeric_columns] > (Q3 + 1.5 * IQR))).any(axis=1)]

numeric_columns_admissions = admissions.select_dtypes(include=[np.number]).columns
admissions = remove_outliers(admissions, numeric_columns_admissions)

numeric_columns_chartevents = chartevents.select_dtypes(include=[np.number]).columns
chartevents = remove_outliers(chartevents, numeric_columns_chartevents)

numeric_columns_diagnoses_icd = diagnoses_icd.select_dtypes(include=[np.number]).columns
diagnoses_icd = remove_outliers(diagnoses_icd, numeric_columns_diagnoses_icd)

numeric_columns_labevents = labevents.select_dtypes(include=[np.number]).columns
labevents = remove_outliers(labevents, numeric_columns_labevents)

numeric_columns_patients = patients.select_dtypes(include=[np.number]).columns
patients = remove_outliers(patients, numeric_columns_patients)

numeric_columns_prescriptions = prescriptions.select_dtypes(include=[np.number]).columns
prescriptions = remove_outliers(prescriptions, numeric_columns_prescriptions)

numeric_columns_procedures_icd = procedures_icd.select_dtypes(include=[np.number]).columns
procedures_icd = remove_outliers(procedures_icd, numeric_columns_procedures_icd)

# 保存处理后的MIMIC-III数据集
admissions.to_csv('/content/FedHealthDP_Project/data/cleaned/admissions_cleaned.csv', index=False)
chartevents.to_csv('/content/FedHealthDP_Project/data/cleaned/chartevents_cleaned.csv', index=False)
diagnoses_icd.to_csv('/content/FedHealthDP_Project/data/cleaned/diagnoses_icd_cleaned.csv', index=False)
labevents.to_csv('/content/FedHealthDP_Project/data/cleaned/labevents_cleaned.csv', index=False)
patients.to_csv('/content/FedHealthDP_Project/data/cleaned/patients_cleaned.csv', index=False)
prescriptions.to_csv('/content/FedHealthDP_Project/data/cleaned/prescriptions_cleaned.csv', index=False)
procedures_icd.to_csv('/content/FedHealthDP_Project/data/cleaned/procedures_icd_cleaned.csv', index=False)



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

# 读取清理后的MIMIC-III数据集
admissions = pd.read_csv('/content/FedHealthDP_Project/data/cleaned/admissions_cleaned.csv')
chartevents = pd.read_csv('/content/FedHealthDP_Project/data/cleaned/chartevents_cleaned.csv')
diagnoses_icd = pd.read_csv('/content/FedHealthDP_Project/data/cleaned/diagnoses_icd_cleaned.csv')
labevents = pd.read_csv('/content/FedHealthDP_Project/data/cleaned/labevents_cleaned.csv')
patients = pd.read_csv('/content/FedHealthDP_Project/data/cleaned/patients_cleaned.csv')
prescriptions = pd.read_csv('/content/FedHealthDP_Project/data/cleaned/prescriptions_cleaned.csv')
procedures_icd = pd.read_csv('/content/FedHealthDP_Project/data/cleaned/procedures_icd_cleaned.csv')

# 提取患者特征：年龄、性别、入院次数、糖尿病诊断和平均血糖水平
patient_features = patients[['subject_id', 'gender', 'dob']]
admission_features = admissions[['subject_id', 'admittime', 'dischtime', 'diagnosis']]
diagnosis_features = diagnoses_icd[['subject_id', 'icd9_code']]
lab_features = labevents[['subject_id', 'itemid', 'value']]
prescription_features = prescriptions[['subject_id', 'drug']]
procedure_features = procedures_icd[['subject_id', 'icd9_code']]

# 计算年龄
admission_features['admittime'] = pd.to_datetime(admission_features['admittime'])
patient_features['dob'] = pd.to_datetime(patient_features['dob'])
patient_features = patient_features.merge(admission_features.groupby('subject_id')['admittime'].first().reset_index(), on='subject_id')
patient_features['age'] = patient_features['admittime'].dt.year - patient_features['dob'].dt.year

# 假设我们感兴趣的ICD代码是糖尿病（ICD-9代码250.xx）
diabetes_diagnosis = diagnosis_features[diagnosis_features['icd9_code'].astype(str).str.startswith('250')]

# 计算每个病人的入院次数
admission_count = admission_features.groupby('subject_id').size().reset_index(name='admission_count')

# 计算每个病人的处方药物数量
prescription_count = prescription_features.groupby('subject_id').size().reset_index(name='prescription_count')

# 计算每个病人的手术次数
procedure_count = procedure_features.groupby('subject_id').size().reset_index(name='procedure_count')

# 将所有特征合并到一个DataFrame
mimic_features = patient_features.merge(admission_count, on='subject_id', how='left')
mimic_features = mimic_features.merge(prescription_count, on='subject_id', how='left')
mimic_features = mimic_features.merge(procedure_count, on='subject_id', how='left')
mimic_features['diabetes'] = mimic_features['subject_id'].isin(diabetes_diagnosis['subject_id']).astype(int)

# 处理实验室结果
# 假设我们感兴趣的实验室项目ID是血糖水平（假设itemid=50809）
glucose_levels = lab_features[lab_features['itemid'] == 50809]
glucose_avg = glucose_levels.groupby('subject_id')['value'].mean().reset_index(name='avg_glucose')

# 合并实验室结果
mimic_features = mimic_features.merge(glucose_avg, on='subject_id', how='left')

# 填充缺失值
mimic_features.fillna({'admission_count': 0, 'prescription_count': 0, 'procedure_count': 0, 'avg_glucose': mimic_features['avg_glucose'].mean()}, inplace=True)

# 存储处理后的MIMIC-III特征数据
mimic_features.to_csv('/content/FedHealthDP_Project/data/processed/mimic_features.csv', index=False)

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

# 读取 DIAGNOSES_ICD 数据
diagnoses_icd = pd.read_csv('/content/FedHealthDP_Project/data/cleaned/diagnoses_icd_cleaned.csv')

# 假设癌症相关的 ICD 代码以 'C' 或 'D0-D4' 开头（具体可以根据实际情况调整）
cancer_icd_codes = diagnoses_icd[diagnoses_icd['icd9_code'].astype(str).str.startswith(('140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196', '197', '198', '199'))]

# 标记患癌症的患者
cancer_patients = cancer_icd_codes['subject_id'].unique()

# 读取患者特征数据
patients = pd.read_csv('/content/FedHealthDP_Project/data/cleaned/patients_cleaned.csv')

# 添加是否患癌症的标签列
patients['has_cancer'] = patients['subject_id'].isin(cancer_patients).astype(int)

# 假设我们已经提取了其他特征数据，并将其与患者特征合并
mimic_features = pd.read_csv('/content/FedHealthDP_Project/data/processed/mimic_features.csv')

# 合并是否患癌症的标签
mimic_features = mimic_features.merge(patients[['subject_id', 'has_cancer']], on='subject_id', how='left')

# 使用 'has_cancer' 列作为标签列
y_mimic = mimic_features['has_cancer']
X_mimic = mimic_features.drop(columns=['has_cancer', 'subject_id'])

# 分割数据集
from sklearn.model_selection import train_test_split

X_mimic_train, X_mimic_test, y_mimic_train, y_mimic_test = train_test_split(X_mimic, y_mimic, test_size=0.2, random_state=42)

# 创建保存分割数据的目录
os.makedirs('/content/FedHealthDP_Project/data/split', exist_ok=True)

# 保存分割后的数据
X_mimic_train.to_csv('/content/FedHealthDP_Project/data/split/mimic_X_train.csv', index=False)
X_mimic_test.to_csv('/content/FedHealthDP_Project/data/split/mimic_X_test.csv', index=False)
y_mimic_train.to_csv('/content/FedHealthDP_Project/data/split/mimic_y_train.csv', index=False)
y_mimic_test.to_csv('/content/FedHealthDP_Project/data/split/mimic_y_test.csv', index=False)