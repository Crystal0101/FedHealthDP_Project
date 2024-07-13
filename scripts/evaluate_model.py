import pandas as pd
import torch
import torch.nn as nn  # 添加此行以导入 nn 模块
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from datetime import datetime

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
def load_and_process_data(file_path, y_file_path, y_column_name, categorical_columns=None, date_columns=None):
    data = pd.read_csv(file_path)
    y_data = pd.read_csv(y_file_path)

    if categorical_columns:
        for col in categorical_columns:
            if col in data.columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
                ohe = OneHotEncoder(sparse_output=False)
                encoded_features = ohe.fit_transform(data[[col]])  # 直接使用返回的 NumPy 数组
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
    return torch.tensor(data.values, dtype=torch.float32), y

# 评估模型的函数
def evaluate_model(model_path, X_test, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CancerModel(X_test.shape[1]).to(device)
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

# 加载数据并评估模型
try:
    # 乳腺癌数据评估
    X_breast_test, y_breast_test = load_and_process_data(
        '/content/FedHealthDP_Project/data/split/breast_cancer_X_test.csv',
        '/content/FedHealthDP_Project/data/split/breast_cancer_y_test.csv',
        'diagnosis')
    evaluate_model('/content/FedHealthDP_Project/models/breast_cancer_model.pth', X_breast_test, y_breast_test)

    # 肺癌数据评估
    X_lung_test, y_lung_test = load_and_process_data(
        '/content/FedHealthDP_Project/data/split/lung_cancer_X_test.csv',
        '/content/FedHealthDP_Project/data/split/lung_cancer_y_test.csv',
        'LUNG_CANCER')
    evaluate_model('/content/FedHealthDP_Project/models/lung_cancer_model.pth', X_lung_test, y_lung_test)

    # 前列腺癌数据评估
    X_prostate_test, y_prostate_test = load_and_process_data(
        '/content/FedHealthDP_Project/data/split/prostate_cancer_X_test.csv',
        '/content/FedHealthDP_Project/data/split/prostate_cancer_y_test.csv',
        'diagnosis_result')
    evaluate_model('/content/FedHealthDP_Project/models/prostate_cancer_model.pth', X_prostate_test, y_prostate_test)

    # MIMIC数据评估
    X_mimic_test, y_mimic_test = load_and_process_data(
        '/content/FedHealthDP_Project/data/split/mimic_X_test.csv',
        '/content/FedHealthDP_Project/data/split/mimic_y_test.csv',
        'has_cancer',  # 请确认这个是正确的列名
        categorical_columns=['gender'],
        date_columns=['dob', 'admittime'])
    evaluate_model('/content/FedHealthDP_Project/models/mimic_model.pth', X_mimic_test, y_mimic_test)

except Exception as e:
    print(f"An error occurred: {e}")