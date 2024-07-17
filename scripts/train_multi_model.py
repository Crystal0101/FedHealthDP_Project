import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin

# 定义深度神经网络模型
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

# 包装深度神经网络模型为符合Scikit-learn API的分类器
class DeepNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_shape, epochs=30, batch_size=32, dp_sensitivity=0.1, dp_epsilon=1.0, dp_dynamic_factor=1.0):
        self.input_shape = input_shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.dp_sensitivity = dp_sensitivity
        self.dp_epsilon = dp_epsilon
        self.dp_dynamic_factor = dp_dynamic_factor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DeeperNN(input_shape).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10, eta_min=1e-6)
        self.best_threshold = 0.5
        self.early_stopping_patience = 5

    def fit(self, X, y):
        self.model.train()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            epoch_loss = 0
            for inputs, targets in loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            self.scheduler.step()
            avg_epoch_loss = epoch_loss / len(loader)
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_epoch_loss:.4f}")

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor).cpu().numpy()
            self.best_threshold = find_best_threshold(outputs, y)
            print(f"Best threshold found: {self.best_threshold:.4f}")

        return self

    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor).cpu().numpy()
        return (outputs >= self.best_threshold).astype(int)

    def predict_proba(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor).cpu().numpy()
        return np.hstack((1 - outputs, outputs))

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

    y = y_data[y_column_name].apply(lambda x: 1 if x in ['M', 'YES', 1] else 0).values

    existing_features = [col for col in all_feature_columns if col in data.columns]
    if len(existing_features) == 0:
        raise ValueError("No matching feature columns found in the data.")

    print("Matching feature columns:", existing_features)

    data = data[existing_features]
    data = data.apply(pd.to_numeric, errors='coerce').fillna(0)  # 填充 NaN 值

    return data.values, y

# 数据增强方法
def balance_data(X, y, method='SMOTE'):
    unique, counts = np.unique(y, return_counts=True)
    print(f"Class distribution before balancing: {dict(zip(unique, counts))}")
    if len(unique) > 1:
        if method == 'ADASYN':
            sampler = ADASYN(random_state=42)
        else:
            sampler = SMOTE(random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)
        return X_res, y_res
    else:
        print("Skipping balancing as there is only one class in target variable.")
        return X, y

# 特征选择
def select_features(X, y, num_features=20):
    mi = mutual_info_classif(X, y)
    selected_features_mi = np.argsort(mi)[-num_features:]
    
    lasso = LogisticRegression(penalty='l1', solver='liblinear')
    lasso.fit(X, y)
    selected_features_lasso = np.argsort(np.abs(lasso.coef_))[0][-num_features:]
    
    xgb = XGBClassifier()
    xgb.fit(X, y)
    selected_features_xgb = np.argsort(xgb.feature_importances_)[-num_features:]
    
    selected_features = np.unique(np.concatenate([selected_features_mi, selected_features_lasso, selected_features_xgb]))
    return X[:, selected_features]

# 选择最佳阈值
def find_best_threshold(outputs, targets):
    precisions, recalls, thresholds = precision_recall_curve(targets, outputs)
    f1_scores = 2 * precisions * recalls / (precisions + recalls)
    best_threshold = thresholds[np.argmax(f1_scores)]
    return best_threshold

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
    print("Target variable distribution after preprocessing:", np.unique(y_breast_train, return_counts=True))

    if X_breast_train.shape[1] == 0:
        raise ValueError("No features selected for breast cancer data.")

    X_breast_train = select_features(X_breast_train, y_breast_train, num_features=20)

    if len(np.unique(y_breast_train)) > 1:
        X_breast_train, y_breast_train = balance_data(X_breast_train, y_breast_train, method='SMOTE')
    
    # 分割训练和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_breast_train, y_breast_train, test_size=0.2, random_state=42)
    
    # 定义模型
    model_logreg = LogisticRegression()
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_xgb = XGBClassifier(n_estimators=100, random_state=42)

    # 训练多个模型
    print("Training Logistic Regression...")
    model_logreg.fit(X_train, y_train)
    print("Training Random Forest...")
    model_rf.fit(X_train, y_train)
    print("Training XGBoost...")
    model_xgb.fit(X_train, y_train)

    # 初始化深度神经网络模型
    deep_model = DeepNNClassifier(input_shape=X_train.shape[1])
    deep_model.fit(X_train, y_train)
    
    # 融合模型
    voting_clf = VotingClassifier(estimators=[
        ('logreg', model_logreg),
        ('rf', model_rf),
        ('xgb', model_xgb),
        ('deep', deep_model)], voting='soft')
    print("Training Voting Classifier...")
    voting_clf.fit(X_train, y_train)
    
    # 评估模型
    y_pred = voting_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'ROC AUC: {roc_auc:.4f}')

except Exception as e:
    print(f"An error occurred: {e}")
