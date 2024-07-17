import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, confusion_matrix, classification_report, roc_curve, auc
from scipy.stats import uniform, randint

# 定义更复杂的模型
class DeeperNN(nn.Module):
    def __init__(self, input_shape):
        super(DeeperNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

# 数据预处理函数
def preprocess_data(file_path, y_file_path, y_column_name, all_feature_columns, categorical_columns=None):
    data = pd.read_csv(file_path)
    y_data = pd.read_csv(y_file_path)

    print("Data columns:", data.columns)
    print("Target columns:", y_data.columns)
    print("Feature columns:", all_feature_columns)

    if categorical_columns:
        for col in categorical_columns:
            if col in data.columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])

    scaler = StandardScaler()
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

    if y_column_name not in y_data.columns:
        raise ValueError(f"Column {y_column_name} not found in {y_file_path}")

    # 确保目标列中的所有正类值被正确映射为 1
    y = y_data[y_column_name].apply(lambda x: 1 if x in ['M', 'YES', 1] else 0).values
    y = torch.tensor(y.astype(np.float32), dtype=torch.float32)

    existing_features = [col for col in all_feature_columns if col in data.columns]
    if len(existing_features) == 0:
        raise ValueError("No matching feature columns found in the data.")

    print("Matching feature columns:", existing_features)

    data = data[existing_features]
    
    # 确保所有数据都是数值类型并转换为浮点数格式
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.fillna(0)  # 填充 NaN 值

    print(f"Data loaded from {file_path} with {data.shape[1]} features.")

    return torch.tensor(data.values.astype(np.float32)), y

# 数据增强方法
def balance_data(X, y, method='SMOTE'):
    unique, counts = np.unique(y.numpy(), return_counts=True)
    print(f"Class distribution before balancing: {dict(zip(unique, counts))}")
    if len(unique) > 1:
        if method == 'ADASYN':
            sampler = ADASYN(random_state=42)
        else:
            sampler = SMOTE(random_state=42)
        X_res, y_res = sampler.fit_resample(X.numpy(), y.numpy())
        return torch.tensor(X_res, dtype=torch.float32), torch.tensor(y_res, dtype=torch.float32)
    else:
        print("Skipping balancing as there is only one class in target variable.")
        return X, y

# 特征选择
def select_features(X, y, num_features=10):
    print("Performing feature selection...")
    mi = mutual_info_classif(X.numpy(), y.numpy(), discrete_features='auto')
    selected_features = np.argsort(mi)[-num_features:]  # 选择互信息最高的特征
    if selected_features.size == 0:
        # 如果没有特征被选择，使用随机森林进行特征选择
        print("No features selected using mutual information. Using RandomForest for feature selection.")
        model = RandomForestClassifier(random_state=42)
        model.fit(X.numpy(), y.numpy())
        importances = model.feature_importances_
        selected_features = np.argsort(importances)[-num_features:]

    print(f"Selected features: {selected_features}")
    if selected_features.size == 0:
        return X  # 如果没有特征被选择，返回原始特征
    return X[:, selected_features]

# 超参数优化
def hyperparameter_tuning(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    param_distributions = {
        'n_estimators': randint(50, 300),
        'max_depth': randint(3, 15),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'max_features': uniform(0.1, 0.9)
    }
    
    opt = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=32,
        cv=3,
        scoring='accuracy',
        random_state=42
    )
    
    opt.fit(X_train.numpy(), y_train.numpy())
    return opt.best_params_

# 动态调整差分隐私噪声
def add_dp_noise(gradients, sensitivity, epsilon, dynamic_factor=1.0):
    noise_scale = sensitivity / (epsilon * dynamic_factor)
    noise = np.random.laplace(0, noise_scale, size=gradients.shape)
    return gradients + noise

# 联邦学习客户端类
class FederatedLearningClient:
    def __init__(self, model, data, target, epochs=1, batch_size=32, dp_sensitivity=0.1, dp_epsilon=1.0, dp_dynamic_factor=1.0):
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
        self.dp_dynamic_factor = dp_dynamic_factor
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
    def __init__(self, model, dp_epsilon, dp_sensitivity, dp_dynamic_factor=1.0):
        self.model = model
        self.dp_epsilon = dp_epsilon
        self.dp_sensitivity = dp_sensitivity
        self.dp_dynamic_factor = dp_dynamic_factor

    def aggregate_updates(self, client_updates):
        aggregated_update = {}
        for k in client_updates[0].keys():
            aggregated_update[k] = sum(client_update[k] for client_update in client_updates) / len(client_updates)
            aggregated_update[k] = torch.tensor(add_dp_noise(aggregated_update[k].cpu().numpy(), self.dp_sensitivity, self.dp_epsilon, self.dp_dynamic_factor), dtype=torch.float32).to(client_updates[0][k].device)

        return aggregated_update

    def update_model(self, aggregated_update):
        self.model.load_state_dict(aggregated_update)

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

# 联邦学习训练函数
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

# 加载数据并初始化客户端和服务器
try:
    # 乳腺癌数据
    feature_columns = ['radius_mean', 'perimeter_mean', 'area_mean', 'concavity_mean',
                       'concave points_mean', 'radius_worst', 'perimeter_worst', 
                       'area_worst', 'concavity_worst', 'concave points_worst']
    
    X_breast_train, y_breast_train = preprocess_data(
        '/content/FedHealthDP_Project/data/split/breast_cancer_X_train.csv',
        '/content/FedHealthDP_Project/data/split/breast_cancer_y_train.csv',
        'diagnosis', all_feature_columns=feature_columns)

    # 打印目标变量的分布
    print("Target variable distribution after preprocessing:", torch.unique(y_breast_train, return_counts=True))

    if X_breast_train.shape[1] == 0:
        raise ValueError("No features selected for breast cancer data.")

    X_breast_train = select_features(X_breast_train, y_breast_train, num_features=10)

    if len(torch.unique(y_breast_train)) > 1:
        X_breast_train, y_breast_train = balance_data(X_breast_train, y_breast_train, method='SMOTE')
        breast_models = [DeeperNN(X_breast_train.shape[1])]
        breast_clients = []
        for model in breast_models:
            best_params = hyperparameter_tuning(X_breast_train, y_breast_train)
            breast_clients.append(FederatedLearningClient(model, X_breast_train, y_breast_train, epochs=30, batch_size=32))
    else:
        breast_clients = []

    # 初始化服务器
    global_model = DeeperNN(X_breast_train.shape[1])
    server = FederatedLearningServer(global_model, dp_epsilon=1.0, dp_sensitivity=0.1, dp_dynamic_factor=1.0)

    # 创建客户端列表，去除 None 的客户端
    clients = breast_clients

    # 运行联邦学习训练
    federated_training(clients, server, rounds=10)

    # 保存全局模型
    torch.save(global_model.state_dict(), '/content/FedHealthDP_Project/models/global_model.pth')

except Exception as e:
    print(f"An error occurred: {e}")
