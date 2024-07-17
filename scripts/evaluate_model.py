import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

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

# 数据加载和预处理函数
def preprocess_data(file_path, y_file_path, feature_columns, y_column_name, categorical_columns=None):
    data = pd.read_csv(file_path)
    y_data = pd.read_csv(y_file_path)

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

    y = y_data[y_column_name].apply(lambda x: 1 if x in ['M', 'YES', 1] else 0).values
    y = torch.tensor(y.astype(np.float32), dtype=torch.float32)

    existing_features = [col for col in feature_columns if col in data.columns]
    if len(existing_features) == 0:
        raise ValueError("No matching feature columns found in the data.")

    data = data[existing_features]
    data = data.apply(pd.to_numeric, errors='coerce').fillna(0)

    return torch.tensor(data.values.astype(np.float32)), y

# 特征选择函数
def select_features(X, y, num_features=10):
    model = RandomForestClassifier(random_state=42)
    model.fit(X.numpy(), y.numpy())
    importances = model.feature_importances_
    selected_features = np.argsort(importances)[-num_features:]
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

# 评估模型
def evaluate_model(model, X_test, y_test, threshold=0.5):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test).cpu().detach()
        predictions = (outputs >= threshold).float()
        
    # 计算各种指标
    accuracy, precision, recall, f1, roc_auc = compute_metrics(outputs, y_test, threshold)
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}")

    # 打印分类报告
    print("Classification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    # 绘制混淆矩阵
    conf_matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # 绘制ROC曲线
    fpr, tpr, _ = roc_curve(y_test, outputs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

# 交叉验证训练和评估
def cross_validate_model(X, y, num_folds=5):
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"Fold {fold + 1}/{num_folds}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = DeeperNN(X_train.shape[1])
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 训练模型
        num_epochs = 100
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train.unsqueeze(1))
            loss.backward()
            optimizer.step()
        
        # 评估模型
        best_threshold = find_best_threshold(model(X_val).cpu().detach(), y_val)
        accuracy, precision, recall, f1, roc_auc = compute_metrics(model(X_val).cpu().detach(), y_val, threshold=best_threshold)
        fold_results.append((accuracy, precision, recall, f1, roc_auc))
    
    fold_results = np.array(fold_results)
    print("Cross-validation results:")
    print(f"Accuracy: {fold_results[:,0].mean():.4f} ± {fold_results[:,0].std():.4f}")
    print(f"Precision: {fold_results[:,1].mean():.4f} ± {fold_results[:,1].std():.4f}")
    print(f"Recall: {fold_results[:,2].mean():.4f} ± {fold_results[:,2].std():.4f}")
    print(f"F1 Score: {fold_results[:,3].mean():.4f} ± {fold_results[:,3].std():.4f}")
    print(f"ROC AUC: {fold_results[:,4].mean():.4f} ± {fold_results[:,4].std():.4f}")

# 加载特征列
with open('/content/FedHealthDP_Project/data/split/breast_features.csv', 'r') as f:
    breast_feature_columns = f.read().splitlines()

# 加载模型
model_path = '/content/FedHealthDP_Project/models/global_model.pth'

try:
    # 加载乳腺癌测试数据
    X_breast_test, y_breast_test = preprocess_data(
        '/content/FedHealthDP_Project/data/split/breast_cancer_X_test.csv',
        '/content/FedHealthDP_Project/data/split/breast_cancer_y_test.csv',
        breast_feature_columns,
        'diagnosis'
    )

    # 加载训练好的模型
    model = DeeperNN(X_breast_test.shape[1])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 评估乳腺癌模型
    print("Evaluating Breast Cancer Model")
    best_threshold_breast = find_best_threshold(model(X_breast_test).cpu().detach(), y_breast_test)
    evaluate_model(model, X_breast_test, y_breast_test, threshold=best_threshold_breast)
    
    # 加载乳腺癌训练数据并进行交叉验证
    print("Cross-validating Breast Cancer Model")
    X_breast_train, y_breast_train = preprocess_data(
        '/content/FedHealthDP_Project/data/split/breast_cancer_X_train.csv',
        '/content/FedHealthDP_Project/data/split/breast_cancer_y_train.csv',
        breast_feature_columns,
        'diagnosis'
    )
    cross_validate_model(X_breast_train, y_breast_train)

except Exception as e:
    print(f"An error occurred: {e}")
