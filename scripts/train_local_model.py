import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import ParameterGrid
from imblearn.over_sampling import ADASYN
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
import pickle

# 定义更复杂的模型
class DeeperNN(nn.Module):
    def __init__(self, input_shape):
        super(DeeperNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

# 数据加载函数
def load_data(file_path, y_file_path):
    X = pd.read_csv(file_path)
    y = pd.read_csv(y_file_path)
    
    # 打印数据类型和头几行数据
    print("Data types of X:")
    print(X.dtypes)
    print("First few rows of X:")
    print(X.head())
    
    # 打印数据类型和头几行标签数据
    print("Data types of y:")
    print(y.dtypes)
    print("First few rows of y:")
    print(y.head())
    
    # 转换数据为数值类型，处理任何非数值列
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # 检查标签数据是否为字符串类型
    if y.dtypes[0] == 'object':
        y = y.applymap(lambda x: 1 if x.lower() == 'yes' else 0)
    
    y = y.squeeze()  # 确保 y 是一维数组
    y = torch.tensor(y.values.astype(np.float32), dtype=torch.float32)
    
    print(f"Data loaded from {file_path} with {X.shape[1]} features.")
    return torch.tensor(X.values.astype(np.float32)), y

# 数据可视化
def visualize_data_distribution(data, title):
    plt.figure(figsize=(12, 6))
    plt.boxplot(data)
    plt.title(title)
    plt.xticks(rotation=90)
    plt.show()

# 样本平衡
def balance_data(X, y):
    unique, counts = np.unique(y.numpy(), return_counts=True)
    print(f"Class distribution before balancing: {dict(zip(unique, counts))}")
    if len(unique) > 1:
        smote = ADASYN(random_state=42)
        X_res, y_res = smote.fit_resample(X.numpy(), y.numpy())
        return torch.tensor(X_res, dtype=torch.float32), torch.tensor(y_res, dtype=torch.float32)
    else:
        print("Skipping balancing as there is only one class in target variable.")
        return X, y

# 超参数调优函数
def hyperparameter_tuning(param_grid, model_class, X_train, y_train, input_shape):
    best_params = {}
    best_score = float('-inf')
    param_combinations = list(ParameterGrid(param_grid))
    
    for params in param_combinations:
        model = model_class(input_shape)
        optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        criterion = nn.BCELoss()
        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)
        
        for epoch in range(params['epochs']):
            model.train()
            epoch_loss = 0
            for inputs, targets in loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets.unsqueeze(1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            scheduler.step(epoch_loss / len(loader))
        
        # 评估模型
        model.eval()
        with torch.no_grad():
            outputs = model(X_train)
            predictions = (outputs >= 0.5).float()
            accuracy = (predictions == y_train.unsqueeze(1)).sum().item() / len(y_train)
        
        if accuracy > best_score:
            best_score = accuracy
            best_params = params
    
    return best_params, best_score

# 定义差分隐私噪声函数
def add_dp_noise(gradients, sensitivity, epsilon):
    noise = np.random.laplace(0, sensitivity / epsilon, size=gradients.shape)
    return gradients + noise

# 计算模型性能指标
def compute_metrics(outputs, targets, threshold=0.5):
    predictions = (outputs >= threshold).float()
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, zero_division=0)
    recall = recall_score(targets, predictions, zero_division=0)
    f1 = f1_score(targets, predictions, zero_division=0)
    roc_auc = roc_auc_score(targets, outputs)
    return accuracy, precision, recall, f1, roc_auc

# 选择最佳阈值
def find_best_threshold(outputs, targets):
    precisions, recalls, thresholds = precision_recall_curve(targets, outputs)
    f1_scores = 2 * precisions * recalls / (precisions + recalls)
    best_threshold = thresholds[np.argmax(f1_scores)]
    return best_threshold

# 联邦学习客户端类
class FederatedLearningClient:
    def __init__(self, model, data, target, epochs=1, batch_size=32, dp_sensitivity=0.1, dp_epsilon=1.0):
        self.model = model
        self.data = data
        self.target = target
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.5)
        self.dp_sensitivity = dp_sensitivity
        self.dp_epsilon = dp_epsilon
        self.best_threshold = 0.5

    def train(self):
        self.model.train()
        dataset = TensorDataset(self.data, self.target)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            epoch_loss = 0
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets.unsqueeze(1))
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            self.scheduler.step(epoch_loss / len(loader))
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss/len(loader):.4f}")

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.data.to(self.device)).cpu()
            self.best_threshold = find_best_threshold(outputs, self.target)
            print(f"Best threshold found: {self.best_threshold:.4f}")

        return self.model.state_dict()

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.data.to(self.device)).cpu()
            metrics = compute_metrics(outputs, self.target, threshold=self.best_threshold)
        return metrics

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
        print(f"Round {round + 1} started.")
        client_updates = [client.train() for client in clients if len(torch.unique(client.target)) > 1]
        if client_updates:
            aggregated_update = server.aggregate_updates(client_updates)
            server.update_model(aggregated_update)
            print(f"Round {round + 1} completed.")
            
            for i, client in enumerate(clients):
                metrics = client.evaluate()
                print(f"Client {i+1} evaluation metrics - Accuracy: {metrics[0]:.4f}, Precision: {metrics[1]:.4f}, Recall: {metrics[2]:.4f}, F1 Score: {metrics[3]:.4f}, ROC AUC: {metrics[4]:.4f}")
        else:
            print(f'Round {round + 1} skipped due to insufficient class variety in targets.')

# 加载特征列
with open('/content/FedHealthDP_Project/FedHealthDP_Project/data/processed/breast_cancer_feature_columns.pkl', 'rb') as f:
    breast_feature_columns = pickle.load(f)
with open('/content/FedHealthDP_Project/FedHealthDP_Project/data/processed/lung_cancer_feature_columns.pkl', 'rb') as f:
    lung_feature_columns = pickle.load(f)

# 超参数范围
param_grid = {
    'lr': [0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64],
    'epochs': [10, 20, 30]
}

# 加载数据并初始化客户端和服务器
try:
    # 乳腺癌数据
    X_breast_train, y_breast_train = load_data(
        '/content/FedHealthDP_Project/FedHealthDP_Project/data/split/breast_cancer_X_train.csv',
        '/content/FedHealthDP_Project/FedHealthDP_Project/data/split/breast_cancer_y_train.csv'
    )
    if len(torch.unique(y_breast_train)) > 1:
        X_breast_train, y_breast_train = balance_data(X_breast_train, y_breast_train)
        visualize_data_distribution(pd.DataFrame(X_breast_train.numpy(), columns=breast_feature_columns), 'Breast Cancer Data Distribution')
        breast_models = [DeeperNN(X_breast_train.shape[1])]
        breast_clients = []
        for model in breast_models:
            best_params, best_score = hyperparameter_tuning(param_grid, type(model), X_breast_train, y_breast_train, X_breast_train.shape[1])
            breast_clients.append(FederatedLearningClient(model, X_breast_train, y_breast_train, epochs=best_params['epochs'], batch_size=best_params['batch_size']))
    else:
        breast_clients = []

    # 肺癌数据
    X_lung_train, y_lung_train = load_data(
        '/content/FedHealthDP_Project/FedHealthDP_Project/data/split/lung_cancer_X_train.csv',
        '/content/FedHealthDP_Project/FedHealthDP_Project/data/split/lung_cancer_y_train.csv'
    )
    if len(torch.unique(y_lung_train)) > 1:
        X_lung_train, y_lung_train = balance_data(X_lung_train, y_lung_train)
        visualize_data_distribution(pd.DataFrame(X_lung_train.numpy(), columns=lung_feature_columns), 'Lung Cancer Data Distribution')
        lung_models = [DeeperNN(X_lung_train.shape[1])]
        lung_clients = []
        for model in lung_models:
            best_params, best_score = hyperparameter_tuning(param_grid, type(model), X_lung_train, y_lung_train, X_lung_train.shape[1])
            lung_clients.append(FederatedLearningClient(model, X_lung_train, y_lung_train, epochs=best_params['epochs'], batch_size=best_params['batch_size']))
    else:
        lung_clients = []

    # 初始化服务器
    global_model = DeeperNN(X_breast_train.shape[1])
    server = FederatedLearningServer(global_model, dp_epsilon=1.0, dp_sensitivity=0.1)

    # 创建客户端列表，去除 None 的客户端
    clients = breast_clients + lung_clients

    # 运行联邦学习训练
    federated_training(clients, server, rounds=10)

    # 保存全局模型
    torch.save(global_model.state_dict(), '/content/FedHealthDP_Project/FedHealthDP_Project/models/global_model.pth')

except Exception as e:
    print(f"An error occurred: {e}")
