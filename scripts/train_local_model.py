import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# 初始化日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

    logging.info("Data columns: %s", data.columns)
    logging.info("Target columns: %s", y_data.columns)
    logging.info("Feature columns: %s", all_feature_columns)

    if categorical_columns:
        for col in categorical_columns:
            if col in data.columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(col)

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

    logging.info("Matching feature columns: %s", existing_features)

    data = data[existing_features]
    
    # 确保所有数据都是数值类型并转换为浮点数格式
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.fillna(0)  # 填充 NaN 值

    logging.info(f"Data loaded from {file_path} with {data.shape[1]} features.")

    return data, torch.tensor(data.values.astype(np.float32)), y

# 数据增强方法
def balance_data(X, y, method='SMOTE'):
    unique, counts = np.unique(y.numpy(), return_counts=True)
    logging.info(f"Class distribution before balancing: {dict(zip(unique, counts))}")
    if len(unique) > 1:
        if method == 'ADASYN':
            sampler = ADASYN(random_state=42)
        else:
            sampler = SMOTE(random_state=42)
        X_res, y_res = sampler.fit_resample(X.numpy(), y.numpy())
        return torch.tensor(X_res, dtype=torch.float32), torch.tensor(y_res, dtype=torch.float32)
    else:
        logging.info("Skipping balancing as there is only one class in target variable.")
        return X, y

# 进行特征选择方法比较实验
def compare_feature_selection_methods(X, y):
    X_df = pd.DataFrame(X.numpy())  # 将 Tensor 转换为 DataFrame
    y_series = pd.Series(y.numpy())  # 将 Tensor 转换为 Series
    
    results = {}

    # 方法 1: 互信息
    mi = mutual_info_classif(X_df, y_series)
    mi_selected_features = np.argsort(mi)[-10:]  # 选择互信息最高的特征
    results['mutual_info'] = mi_selected_features

    # 方法 2: 随机森林
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_df, y_series)
    rf_selected_features = np.argsort(rf.feature_importances_)[-10:]
    results['random_forest'] = rf_selected_features

    return results

# 特征选择
def select_features(X, y, num_features=10):
    logging.info("Performing feature selection...")
    X_df = pd.DataFrame(X.numpy())  # 将 Tensor 转换为 DataFrame
    y_series = pd.Series(y.numpy())  # 将 Tensor 转换为 Series

    mi = mutual_info_classif(X_df, y_series, discrete_features='auto')
    selected_features = np.argsort(mi)[-num_features:]  # 选择互信息最高的特征
    if selected_features.size == 0:
        # 如果没有特征被选择，使用随机森林进行特征选择
        logging.info("No features selected using mutual information. Using RandomForest for feature selection.")
        model = RandomForestClassifier(random_state=42)
        model.fit(X_df, y_series)
        importances = model.feature_importances_
        selected_features = np.argsort(importances)[-num_features:]

    logging.info(f"Selected features: {selected_features}")
    if selected_features.size == 0:
        return X  # 如果没有特征被选择，返回原始特征

    return X[:, selected_features]

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

# 动态调整差分隐私噪声
def add_dp_noise(gradients, sensitivity, epsilon, dynamic_factor=1.0):
    noise_scale = sensitivity / (epsilon * dynamic_factor)
    noise = np.random.laplace(0, noise_scale, size=gradients.shape)
    return gradients + noise
    
# 联邦学习客户端类
class FederatedLearningClient:
    def __init__(self, model, data, target, epochs=1, batch_size=32, dp_sensitivity=0.1, dp_epsilon=1.0, dp_dynamic_factor=1.0, val_split=0.2):
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

        # 划分训练集和验证集
        val_size = int(len(data) * val_split)
        train_size = len(data) - val_size
        dataset = TensorDataset(self.data, self.target)
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])
        
    def train(self):
        self.model.train()
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        train_precisions = []
        val_precisions = []
        train_recalls = []
        val_recalls = []
        train_f1s = []
        val_f1s = []
        train_roc_aucs = []
        val_roc_aucs = []
        learning_rates = []

        for epoch in range(self.epochs):
            epoch_train_loss = 0
            correct_train = 0
            total_train = 0
            all_train_outputs = []
            all_train_targets = []

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets.unsqueeze(1))
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()
                all_train_outputs.append(outputs.cpu().detach())
                all_train_targets.append(targets.cpu().detach())
                predicted = (outputs >= 0.5).float()
                total_train += targets.size(0)
                correct_train += (predicted == targets.unsqueeze(1)).sum().item()
            
            epoch_train_loss /= len(train_loader)
            train_losses.append(epoch_train_loss)
            train_accuracies.append(correct_train / total_train)
            learning_rates.append(self.optimizer.param_groups[0]['lr'])
            all_train_outputs = torch.cat(all_train_outputs)
            all_train_targets = torch.cat(all_train_targets)
            train_metrics = compute_metrics(all_train_outputs, all_train_targets)
            train_precisions.append(train_metrics[1])
            train_recalls.append(train_metrics[2])
            train_f1s.append(train_metrics[3])
            train_roc_aucs.append(train_metrics[4])

            # 验证
            self.model.eval()
            epoch_val_loss = 0
            correct_val = 0
            total_val = 0
            all_val_outputs = []
            all_val_targets = []

            with torch.no_grad():
                for val_inputs, val_targets in val_loader:
                    val_inputs, val_targets = val_inputs.to(self.device), val_targets.to(self.device)
                    val_outputs = self.model(val_inputs)
                    val_loss = self.criterion(val_outputs, val_targets.unsqueeze(1))
                    epoch_val_loss += val_loss.item()
                    all_val_outputs.append(val_outputs.cpu().detach())
                    all_val_targets.append(val_targets.cpu().detach())
                    predicted = (val_outputs >= 0.5).float()
                    total_val += val_targets.size(0)
                    correct_val += (predicted == val_targets.unsqueeze(1)).sum().item()
                    
            epoch_val_loss /= len(val_loader)
            val_losses.append(epoch_val_loss)
            val_accuracies.append(correct_val / total_val)
            all_val_outputs = torch.cat(all_val_outputs)
            all_val_targets = torch.cat(all_val_targets)
            val_metrics = compute_metrics(all_val_outputs, all_val_targets)
            val_precisions.append(val_metrics[1])
            val_recalls.append(val_metrics[2])
            val_f1s.append(val_metrics[3])
            val_roc_aucs.append(val_metrics[4])

            self.scheduler.step(epoch_val_loss)
            logging.info(f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Train Acc: {correct_train/total_train:.4f}, Val Acc: {correct_val/total_val:.4f}, Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.data.to(self.device)).cpu()
            self.best_threshold = find_best_threshold(outputs, self.target)
            logging.info(f"Best threshold found: {self.best_threshold:.4f}")

        return self.model.state_dict(), train_losses, val_losses, train_accuracies, val_accuracies, learning_rates, train_precisions, val_precisions, train_recalls, val_recalls, train_f1s, val_f1s, train_roc_aucs, val_roc_aucs

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.data.to(self.device)).cpu()
            metrics = compute_metrics(outputs, self.target, threshold=self.best_threshold)
        return metrics

# 经典算法比较实验
def compare_classical_algorithms(X_train, y_train, X_test, y_test):
    classifiers = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42)
    }

    results = {}

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(torch.tensor(y_pred), torch.tensor(y_test))
        results[name] = metrics
        logging.info(f"{name} - Accuracy: {metrics[0]:.4f}, Precision: {metrics[1]:.4f}, Recall: {metrics[2]:.4f}, F1 Score: {metrics[3]:.4f}, ROC AUC: {metrics[4]:.4f}")

    return results

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

# 联邦学习训练函数
def federated_training(clients, server, rounds, output_dir='results'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for round in range(rounds):
        logging.info(f"Round {round + 1} started.")
        client_updates = []
        all_train_losses = []
        all_val_losses = []
        all_train_accuracies = []
        all_val_accuracies = []
        all_train_precisions = []
        all_val_precisions = []
        all_train_recalls = []
        all_val_recalls = []
        all_train_f1s = []
        all_val_f1s = []
        all_train_roc_aucs = []
        all_val_roc_aucs = []
        all_learning_rates = []

        for i, client in enumerate(clients):
            if len(torch.unique(client.target)) > 1:
                update, train_losses, val_losses, train_accuracies, val_accuracies, learning_rates, train_precisions, val_precisions, train_recalls, val_recalls, train_f1s, val_f1s, train_roc_aucs, val_roc_aucs = client.train()
                client_updates.append(update)
                all_train_losses.append(train_losses)
                all_val_losses.append(val_losses)
                all_train_accuracies.append(train_accuracies)
                all_val_accuracies.append(val_accuracies)
                all_train_precisions.append(train_precisions)
                all_val_precisions.append(val_precisions)
                all_train_recalls.append(train_recalls)
                all_val_recalls.append(val_recalls)
                all_train_f1s.append(train_f1s)
                all_val_f1s.append(val_f1s)
                all_train_roc_aucs.append(train_roc_aucs)
                all_val_roc_aucs.append(val_roc_aucs)
                all_learning_rates.append(learning_rates)
                
                # 保存每轮训练的学习曲线
                plt.figure()
                plt.plot(train_losses, label=f'Client {i+1} Train Loss')
                plt.plot(val_losses, label=f'Client {i+1} Val Loss', linestyle='--')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title(f'Client {i+1} Training and Validation Loss - Round {round + 1}')
                plt.legend(loc='upper right')
                plt.savefig(os.path.join(output_dir, f'client_{i+1}_train_val_loss_round_{round + 1}.png'))
                plt.close()

                plt.figure()
                plt.plot(train_accuracies, label=f'Client {i+1} Train Accuracy')
                plt.plot(val_accuracies, label=f'Client {i+1} Val Accuracy', linestyle='--')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.title(f'Client {i+1} Training and Validation Accuracy - Round {round + 1}')
                plt.legend(loc='upper right')
                plt.savefig(os.path.join(output_dir, f'client_{i+1}_train_val_accuracy_round_{round + 1}.png'))
                plt.close()

                plt.figure()
                plt.plot(learning_rates, label=f'Client {i+1} Learning Rate')
                plt.xlabel('Epoch')
                plt.ylabel('Learning Rate')
                plt.title(f'Client {i+1} Learning Rate - Round {round + 1}')
                plt.legend(loc='upper right')
                plt.savefig(os.path.join(output_dir, f'client_{i+1}_learning_rate_round_{round + 1}.png'))
                plt.close()

                plt.figure()
                plt.plot(train_precisions, label=f'Client {i+1} Train Precision')
                plt.plot(val_precisions, label=f'Client {i+1} Val Precision', linestyle='--')
                plt.xlabel('Epoch')
                plt.ylabel('Precision')
                plt.title(f'Client {i+1} Training and Validation Precision - Round {round + 1}')
                plt.legend(loc='upper right')
                plt.savefig(os.path.join(output_dir, f'client_{i+1}_train_val_precision_round_{round + 1}.png'))
                plt.close()

                plt.figure()
                plt.plot(train_recalls, label=f'Client {i+1} Train Recall')
                plt.plot(val_recalls, label=f'Client {i+1} Val Recall', linestyle='--')
                plt.xlabel('Epoch')
                plt.ylabel('Recall')
                plt.title(f'Client {i+1} Training and Validation Recall - Round {round + 1}')
                plt.legend(loc='upper right')
                plt.savefig(os.path.join(output_dir, f'client_{i+1}_train_val_recall_round_{round + 1}.png'))
                plt.close()

                plt.figure()
                plt.plot(train_f1s, label=f'Client {i+1} Train F1 Score')
                plt.plot(val_f1s, label=f'Client {i+1} Val F1 Score', linestyle='--')
                plt.xlabel('Epoch')
                plt.ylabel('F1 Score')
                plt.title(f'Client {i+1} Training and Validation F1 Score - Round {round + 1}')
                plt.legend(loc='upper right')
                plt.savefig(os.path.join(output_dir, f'client_{i+1}_train_val_f1_round_{round + 1}.png'))
                plt.close()

                plt.figure()
                plt.plot(train_roc_aucs, label=f'Client {i+1} Train ROC AUC')
                plt.plot(val_roc_aucs, label=f'Client {i+1} Val ROC AUC', linestyle='--')
                plt.xlabel('Epoch')
                plt.ylabel('ROC AUC')
                plt.title(f'Client {i+1} Training and Validation ROC AUC - Round {round + 1}')
                plt.legend(loc='upper right')
                plt.savefig(os.path.join(output_dir, f'client_{i+1}_train_val_roc_auc_round_{round + 1}.png'))
                plt.close()
            else:
                logging.info(f"Skipping training for client {i+1} due to insufficient class variety in targets.")

        if client_updates:
            aggregated_update = server.aggregate_updates(client_updates)
            server.update_model(aggregated_update)
            logging.info(f"Round {round + 1} completed.")

            for i, client in enumerate(clients):
                metrics = client.evaluate()
                logging.info(f"Client {i+1} evaluation metrics - Accuracy: {metrics[0]:.4f}, Precision: {metrics[1]:.4f}, Recall: {metrics[2]:.4f}, F1 Score: {metrics[3]:.4f}, ROC AUC: {metrics[4]:.4f}")

        else:
            logging.info(f'Round {round + 1} skipped due to insufficient class variety in targets.')

# 加载数据并初始化客户端和服务器
def main():
    try:
        # 乳腺癌数据
        feature_columns = ['radius_mean', 'perimeter_mean', 'area_mean', 'concavity_mean',
                           'concave points_mean', 'radius_worst', 'perimeter_worst', 
                           'area_worst', 'concavity_worst', 'concave points_worst']
        
        # 预处理所有数据集
        datasets = [
            ('/content/FedHealthDP_Project/data/split/breast_cancer_X_train_0.csv', '/content/FedHealthDP_Project/data/split/breast_cancer_y_train_0.csv'),
            ('/content/FedHealthDP_Project/data/split/breast_cancer_X_train_1.csv', '/content/FedHealthDP_Project/data/split/breast_cancer_y_train_1.csv'),
            ('/content/FedHealthDP_Project/data/split/breast_cancer_X_train_2.csv', '/content/FedHealthDP_Project/data/split/breast_cancer_y_train_2.csv')
        ]
        
        clients = []
        for i, (X_path, y_path) in enumerate(datasets):
            df_train, X_train, y_train = preprocess_data(X_path, y_path, 'diagnosis', all_feature_columns=feature_columns)
            
            logging.info(f"Dataset {i} target variable distribution after preprocessing: %s", torch.unique(y_train, return_counts=True))
            
            if X_train.shape[1] == 0:
                raise ValueError("No features selected for breast cancer data.")
            
            feature_selection_results = compare_feature_selection_methods(X_train, y_train)
            logging.info(f"Dataset {i} feature selection comparison results: %s", feature_selection_results)
            
            X_train = select_features(X_train, y_train, num_features=10)
            
            if len(torch.unique(y_train)) > 1:
                X_train, y_train = balance_data(X_train, y_train, method='SMOTE')
                model = DeeperNN(X_train.shape[1])
                clients.append(FederatedLearningClient(model, X_train, y_train, epochs=30, batch_size=32, val_split=0.2))
        
        global_model = DeeperNN(X_train.shape[1])
        server = FederatedLearningServer(global_model, dp_epsilon=1.0, dp_sensitivity=0.1, dp_dynamic_factor=1.0)

        federated_training(clients, server, rounds=10, output_dir='breast_cancer_training_results')

        torch.save(global_model.state_dict(), '/content/FedHealthDP_Project/models/global_model.pth')

        X_train_combined, y_train_combined = [], []
        for X_path, y_path in datasets:
            df_train, X_train, y_train = preprocess_data(X_path, y_path, 'diagnosis', all_feature_columns=feature_columns)
            X_train_combined.append(X_train)
            y_train_combined.append(y_train)
        
        X_train_combined = torch.cat(X_train_combined)
        y_train_combined = torch.cat(y_train_combined)
        
        X_train, X_test, y_train, y_test = train_test_split(X_train_combined, y_train_combined, test_size=0.2, random_state=42)
        classical_results = compare_classical_algorithms(X_train.numpy(), y_train.numpy(), X_test.numpy(), y_test.numpy())
        logging.info(f"Classical algorithms comparison results: {classical_results}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
