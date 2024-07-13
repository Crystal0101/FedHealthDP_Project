import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from datetime import datetime
import pickle

# 定义模型类
class CancerModel(nn.Module):
    def __init__(self, input_shape):
        super(CancerModel, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.linear_relu_stack(x)

# 数据标准化和转换非数值列
def load_and_process_data(file_path, y_file_path, y_column_name, all_feature_columns, categorical_columns=None, date_columns=None):
    data = pd.read_csv(file_path)
    y_data = pd.read_csv(y_file_path)

    if categorical_columns:
        for col in categorical_columns:
            if col in data.columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
                ohe = OneHotEncoder(sparse_output=False)
                encoded_features = ohe.fit_transform(data[[col]])
                encoded_features_df = pd.DataFrame(encoded_features, columns=[f"{col}_{int(i)}" for i in range(encoded_features.shape[1])])
                data = pd.concat([data, encoded_features_df], axis=1).drop(col, axis=1)

    if date_columns:
        for col in date_columns:
            if col in data.columns:
                data[col] = pd.to_datetime(data[col]).apply(lambda x: (datetime.now() - x).days / 365.25)

    scaler = StandardScaler()
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

    if y_column_name not in y_data.columns:
        raise ValueError(f"Column {y_column_name} not found in {y_file_path}")

    y = torch.tensor(y_data[y_column_name].apply(lambda x: 1 if x == 'M' or x == 'YES' else 0).values, dtype=torch.float32)
    
    # 只选择存在的特征列
    existing_features = [col for col in all_feature_columns if col in data.columns]
    missing_features = [col for col in all_feature_columns if col not in data.columns]
    
    # 对不存在的特征列进行填充
    for col in missing_features:
        data[col] = 0

    data = data[all_feature_columns]

    print(f"Data loaded from {file_path} with {data.shape[1]} features.")

    return torch.tensor(data.values, dtype=torch.float32), y

# 评估模型的函数
def evaluate_model(model_path, input_shape, X_test, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CancerModel(input_shape).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        inputs = X_test.to(device)
        targets = y_test.to(device)
        outputs = model(inputs)
        predictions = (outputs >= 0.5).float()

    accuracy = accuracy_score(targets.cpu(), predictions.cpu())
    precision = precision_score(targets.cpu(), predictions.cpu())
    recall = recall_score(targets.cpu(), predictions.cpu())
    f1 = f1_score(targets.cpu(), predictions.cpu())

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

# 加载特征列
with open('/content/FedHealthDP_Project/feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

print(f"Loaded feature columns with {len(feature_columns)} features.")

# 加载数据并评估模型
try:
    # 乳腺癌数据评估
    X_breast_test, y_breast_test = load_and_process_data(
        '/content/FedHealthDP_Project/data/split/breast_cancer_X_test.csv',
        '/content/FedHealthDP_Project/data/split/breast_cancer_y_test.csv',
        'diagnosis', all_feature_columns=feature_columns)
    evaluate_model('/content/FedHealthDP_Project/models/global_model.pth', len(feature_columns), X_breast_test, y_breast_test)

    # 肺癌数据评估
    X_lung_test, y_lung_test = load_and_process_data(
        '/content/FedHealthDP_Project/data/split/lung_cancer_X_test.csv',
        '/content/FedHealthDP_Project/data/split/lung_cancer_y_test.csv',
        'LUNG_CANCER', all_feature_columns=feature_columns)
    evaluate_model('/content/FedHealthDP_Project/models/global_model.pth', len(feature_columns), X_lung_test, y_lung_test)

    # 前列腺癌数据评估
    X_prostate_test, y_prostate_test = load_and_process_data(
        '/content/FedHealthDP_Project/data/split/prostate_cancer_X_test.csv',
        '/content/FedHealthDP_Project/data/split/prostate_cancer_y_test.csv',
        'diagnosis_result', all_feature_columns=feature_columns)
    evaluate_model('/content/FedHealthDP_Project/models/global_model.pth', len(feature_columns), X_prostate_test, y_prostate_test)

    # MIMIC数据评估
    X_mimic_test, y_mimic_test = load_and_process_data(
        '/content/FedHealthDP_Project/data/split/mimic_X_test.csv',
        '/content/FedHealthDP_Project/data/split/mimic_y_test.csv',
        'has_cancer', all_feature_columns=feature_columns,
        categorical_columns=['gender'],
        date_columns=['dob', 'admittime'])
    evaluate_model('/content/FedHealthDP_Project/models/global_model.pth', len(feature_columns), X_mimic_test, y_mimic_test)

except Exception as e:
    print(f"An error occurred: {e}")
