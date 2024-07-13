import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset
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

# 定义差分隐私噪声函数
def add_dp_noise(gradients, sensitivity, epsilon):
    noise = np.random.laplace(0, sensitivity / epsilon, size=gradients.shape)
    return gradients + noise

# 联邦学习客户端类
class FederatedLearningClient:
    def __init__(self, model, data, target, epochs=1, batch_size=32):
        self.model = model
        self.data = data
        self.target = target
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters())

    def train(self):
        self.model.train()
        dataset = TensorDataset(self.data, self.target)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.epochs):
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets.unsqueeze(1))
                loss.backward()
                self.optimizer.step()

        return self.model.state_dict()

# 联邦学习服务器类
class FederatedLearningServer:
    def __init__(self, model, dp_epsilon, dp_sensitivity):
        self.model = model
        self.dp_epsilon = dp_epsilon
        self.dp_sensitivity = dp_sensitivity

    def aggregate_updates(self, client_updates):
        aggregated_update = {}
        for k in client_updates[0].keys():
            aggregated_update[k] = sum(client_update[k] for client_update in client_updates) / len(client_updates)
            aggregated_update[k] = torch.tensor(add_dp_noise(aggregated_update[k].cpu().numpy(), self.dp_sensitivity, self.dp_epsilon), dtype=torch.float32).to(client_updates[0][k].device)

        return aggregated_update

    def update_model(self, aggregated_update):
        self.model.load_state_dict(aggregated_update)

# 示例使用
def federated_training(clients, server, rounds):
    for round in range(rounds):
        client_updates = [client.train() for client in clients]
        aggregated_update = server.aggregate_updates(client_updates)
        server.update_model(aggregated_update)
        print(f'Round {round + 1} completed.')

# 定义统一特征列
feature_columns = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
                   'smoothness_mean', 'compactness_mean', 'concavity_mean', 
                   'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                   'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 
                   'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE', 
                   'ALLERGY', 'WHEEZING', 'ALCOHOL CONSUMING', 
                   'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 
                   'CHEST PAIN', 'gender_0', 'gender_1', 'dob', 'admittime']

# 保存特征列
with open('/content/FedHealthDP_Project/feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)

# 加载数据并初始化客户端和服务器
try:
    # 乳腺癌数据
    X_breast_train, y_breast_train = load_and_process_data(
        '/content/FedHealthDP_Project/data/split/breast_cancer_X_train.csv',
        '/content/FedHealthDP_Project/data/split/breast_cancer_y_train.csv',
        'diagnosis', all_feature_columns=feature_columns)
    breast_model = CancerModel(X_breast_train.shape[1])
    breast_client = FederatedLearningClient(breast_model, X_breast_train, y_breast_train)

    # 肺癌数据
    X_lung_train, y_lung_train = load_and_process_data(
        '/content/FedHealthDP_Project/data/split/lung_cancer_X_train.csv',
        '/content/FedHealthDP_Project/data/split/lung_cancer_y_train.csv',
        'LUNG_CANCER', all_feature_columns=feature_columns)
    lung_model = CancerModel(X_lung_train.shape[1])
    lung_client = FederatedLearningClient(lung_model, X_lung_train, y_lung_train)

    # 前列腺癌数据
    X_prostate_train, y_prostate_train = load_and_process_data(
        '/content/FedHealthDP_Project/data/split/prostate_cancer_X_train.csv',
        '/content/FedHealthDP_Project/data/split/prostate_cancer_y_train.csv',
        'diagnosis_result', all_feature_columns=feature_columns)
    prostate_model = CancerModel(X_prostate_train.shape[1])
    prostate_client = FederatedLearningClient(prostate_model, X_prostate_train, y_prostate_train)

    # MIMIC数据
    X_mimic_train, y_mimic_train = load_and_process_data(
        '/content/FedHealthDP_Project/data/split/mimic_X_train.csv',
        '/content/FedHealthDP_Project/data/split/mimic_y_train.csv',
        'has_cancer', all_feature_columns=feature_columns,
        categorical_columns=['gender'],
        date_columns=['dob', 'admittime'])
    mimic_model = CancerModel(X_mimic_train.shape[1])
    mimic_client = FederatedLearningClient(mimic_model, X_mimic_train, y_mimic_train)

    # 初始化服务器
    global_model = CancerModel(X_breast_train.shape[1])  # 确保模型输入大小一致
    server = FederatedLearningServer(global_model, dp_epsilon=0.1, dp_sensitivity=1.0)

    # 创建客户端列表
    clients = [breast_client, lung_client, prostate_client, mimic_client]

    # 运行联邦学习训练
    federated_training(clients, server, rounds=10)

    # 保存全局模型
    torch.save(global_model.state_dict(), '/content/FedHealthDP_Project/models/global_model.pth')

except Exception as e:
    print(f"An error occurred: {e}")
