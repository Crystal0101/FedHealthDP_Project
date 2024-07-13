import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset
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
                ohe = OneHotEncoder(sparse_output=False)  # 直接输出 NumPy 数组
                encoded_features = ohe.fit_transform(data[[col]])  # 无需调用 toarray()
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

# 定义模型训练函数
def train_model(X_train, y_train, model_save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CancerModel(X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(10):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    torch.save(model.state_dict(), model_save_path)
    return model

# 加载数据并训练模型
try:
    # 乳腺癌数据
    X_breast_train, y_breast_train = load_and_process_data(
        '/content/FedHealthDP_Project/data/split/breast_cancer_X_train.csv',
        '/content/FedHealthDP_Project/data/split/breast_cancer_y_train.csv',
        'diagnosis')
    breast_model = train_model(X_breast_train, y_breast_train, '/content/FedHealthDP_Project/models/breast_cancer_model.pth')

    # 肺癌数据
    X_lung_train, y_lung_train = load_and_process_data(
        '/content/FedHealthDP_Project/data/split/lung_cancer_X_train.csv',
        '/content/FedHealthDP_Project/data/split/lung_cancer_y_train.csv',
        'LUNG_CANCER')
    lung_model = train_model(X_lung_train, y_lung_train, '/content/FedHealthDP_Project/models/lung_cancer_model.pth')

    # 前列腺癌数据
    X_prostate_train, y_prostate_train = load_and_process_data(
        '/content/FedHealthDP_Project/data/split/prostate_cancer_X_train.csv',
        '/content/FedHealthDP_Project/data/split/prostate_cancer_y_train.csv',
        'diagnosis_result')
    prostate_model = train_model(X_prostate_train, y_prostate_train, '/content/FedHealthDP_Project/models/prostate_cancer_model.pth')

    # MIMIC数据
    X_mimic_train, y_mimic_train = load_and_process_data(
        '/content/FedHealthDP_Project/data/split/mimic_X_train.csv',
        '/content/FedHealthDP_Project/data/split/mimic_y_train.csv',
        'has_cancer',  # 请确认这个是正确的列名
        categorical_columns=['gender'],
        date_columns=['dob', 'admittime'])
    mimic_model = train_model(X_mimic_train, y_mimic_train, '/content/FedHealthDP_Project/models/mimic_model.pth')

except Exception as e:
    print(f"An error occurred: {e}")
