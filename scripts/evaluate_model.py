import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, confusion_matrix, classification_report, roc_curve, auc, matthews_corrcoef, cohen_kappa_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import logging

# 初始化日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 定义更复杂的模型
class DeeperNN(nn.Module):
    def __init__(self, input_shape):
        super(DeeperNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
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

    logging.info("Data columns: %s", data.columns)
    logging.info("Target columns: %s", y_data.columns)
    logging.info("Feature columns: %s", feature_columns)

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

    logging.info("Matching feature columns: %s", existing_features)

    data = data[existing_features]
    
    data = data.apply(pd.to_numeric, errors='coerce').fillna(0)

    logging.info(f"Data loaded from {file_path} with {data.shape[1]} features.")

    return torch.tensor(data.values.astype(np.float32)), y, data.columns.tolist()

# 特征选择函数
def select_features(X, y, feature_names, num_features=10):
    logging.info("Performing feature selection...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X.numpy(), y.numpy())
    importances = model.feature_importances_
    selected_indices = np.argsort(importances)[-num_features:]
    return X[:, selected_indices], importances[selected_indices], np.array(feature_names)[selected_indices]

# 计算模型性能指标
def compute_metrics(outputs, targets, threshold=0.5):
    predictions = (outputs >= threshold).float()
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, zero_division=0)
    recall = recall_score(targets, predictions, zero_division=0)
    f1 = f1_score(targets, predictions, zero_division=0)
    roc_auc = roc_auc_score(targets, outputs)
    mcc = matthews_corrcoef(targets, predictions)
    kappa = cohen_kappa_score(targets, predictions)
    return accuracy, precision, recall, f1, roc_auc, mcc, kappa

# 选择最佳阈值
def find_best_threshold(outputs, targets):
    precisions, recalls, thresholds = precision_recall_curve(targets, outputs)
    f1_scores = 2 * precisions * recalls / (precisions + recalls)
    best_threshold = thresholds[np.argmax(f1_scores)]
    return best_threshold

# 评估模型
def evaluate_model(model, X_test, y_test, threshold=0.5, output_dir='results'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model.eval()
    with torch.no_grad():
        outputs = model(X_test).cpu().detach()
        predictions = (outputs >= threshold).float()
        
    # 计算各种指标
    accuracy, precision, recall, f1, roc_auc, mcc, kappa = compute_metrics(outputs, y_test, threshold)
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")
    logging.info(f"ROC AUC: {roc_auc:.4f}")
    logging.info(f"MCC: {mcc:.4f}")
    logging.info(f"Kappa: {kappa:.4f}")

    # 打印分类报告
    logging.info("Classification Report:")
    logging.info("\n" + classification_report(y_test, predictions, zero_division=0))

    # 打印记录数量
    logging.info(f"Number of records in evaluation: {len(y_test)}")

    # 绘制混淆矩阵并保存到本地文件
    conf_matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    # 绘制ROC曲线并保存到本地文件
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
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

    # 绘制Precision-Recall曲线并保存到本地文件
    precisions, recalls, _ = precision_recall_curve(y_test, outputs)
    plt.figure()
    plt.plot(recalls, precisions, marker='.', color='blue')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
    plt.close()

# 交叉验证训练和评估
def cross_validate_model(X, y, feature_names, num_folds=5, output_dir='results'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_results = []
    train_losses = []
    val_losses = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        logging.info(f"Fold {fold + 1}/{num_folds}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = DeeperNN(X_train.shape[1])
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 训练模型
        num_epochs = 100
        train_loss_per_epoch = []
        val_loss_per_epoch = []
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss_per_epoch.append(loss.item())

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val.unsqueeze(1))
                val_loss_per_epoch.append(val_loss.item())
        
        train_losses.append(train_loss_per_epoch)
        val_losses.append(val_loss_per_epoch)
        
        # 评估模型
        best_threshold = find_best_threshold(model(X_val).cpu().detach(), y_val)
        accuracy, precision, recall, f1, roc_auc, mcc, kappa = compute_metrics(model(X_val).cpu().detach(), y_val, threshold=best_threshold)
        fold_results.append((accuracy, precision, recall, f1, roc_auc, mcc, kappa))
    
    fold_results = np.array(fold_results)
    logging.info("Cross-validation results:")
    logging.info(f"Accuracy: {fold_results[:,0].mean():.4f} ± {fold_results[:,0].std():.4f}")
    logging.info(f"Precision: {fold_results[:,1].mean():.4f} ± {fold_results[:,1].std():.4f}")
    logging.info(f"Recall: {fold_results[:,2].mean():.4f} ± {fold_results[:,2].std():.4f}")
    logging.info(f"F1 Score: {fold_results[:,3].mean():.4f} ± {fold_results[:,3].std():.4f}")
    logging.info(f"ROC AUC: {fold_results[:,4].mean():.4f} ± {fold_results[:,4].std():.4f}")
    logging.info(f"MCC: {fold_results[:,5].mean():.4f} ± {fold_results[:,5].std():.4f}")
    logging.info(f"Kappa: {fold_results[:,6].mean():.4f} ± {fold_results[:,6].std():.4f}")

    # 绘制学习曲线并保存到本地文件
    plt.figure()
    for i in range(num_folds):
        plt.plot(train_losses[i], label=f'Train Fold {i+1}')
        plt.plot(val_losses[i], label=f'Val Fold {i+1}', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curves')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'))
    plt.close()

# 评估经典机器学习模型
def evaluate_classical_models(X_train, y_train, X_test, y_test, output_dir='results'):
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probas = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        roc_auc = roc_auc_score(y_test, probas)
        
        results[name] = (accuracy, precision, recall, f1, roc_auc)
        
        logging.info(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}")

    logging.info(f"Classical algorithms comparison results: {results}")

    return results

# 集成学习预测函数
# 调整权重的集成方法
def ensemble_predict(models, X, weights):
    predictions = [model(X) for model in models]
    weighted_predictions = torch.zeros_like(predictions[0])
    for weight, prediction in zip(weights, predictions):
        weighted_predictions += weight * prediction
    return weighted_predictions / sum(weights)

# 评估加权集成模型
def evaluate_ensemble(models, X_test, y_test, threshold=0.5, output_dir='results', weights=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if weights is None:
        weights = [1.0 / len(models)] * len(models)

    with torch.no_grad():
        outputs = ensemble_predict(models, X_test, weights).cpu().detach()
        predictions = (outputs >= threshold).float()
    
    # 计算各种指标
    accuracy, precision, recall, f1, roc_auc, mcc, kappa = compute_metrics(outputs, y_test, threshold)
    logging.info(f"Ensemble Accuracy: {accuracy:.4f}")
    logging.info(f"Ensemble Precision: {precision:.4f}")
    logging.info(f"Ensemble Recall: {recall:.4f}")
    logging.info(f"Ensemble F1 Score: {f1:.4f}")
    logging.info(f"Ensemble ROC AUC: {roc_auc:.4f}")
    logging.info(f"Ensemble MCC: {mcc:.4f}")
    logging.info(f"Ensemble Kappa: {kappa:.4f}")

    # 打印分类报告
    logging.info("Ensemble Classification Report:")
    logging.info("\n" + classification_report(y_test, predictions, zero_division=0))

    # 绘制混淆矩阵并保存到本地文件
    conf_matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Ensemble Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_dir, 'ensemble_confusion_matrix.png'))
    plt.close()

    # 绘制ROC曲线并保存到本地文件
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
    plt.savefig(os.path.join(output_dir, 'ensemble_roc_curve.png'))
    plt.close()

    # 绘制Precision-Recall曲线并保存到本地文件
    precisions, recalls, _ = precision_recall_curve(y_test, outputs)
    plt.figure()
    plt.plot(recalls, precisions, marker='.', color='blue')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(os.path.join(output_dir, 'ensemble_precision_recall_curve.png'))
    plt.close()

# 加载特征列
with open('/content/FedHealthDP_Project/data/split/breast_features.csv', 'r') as f:
    breast_feature_columns = f.read().splitlines()

try:
    # 加载乳腺癌测试数据
    X_breast_test, y_breast_test, feature_names = preprocess_data(
        '/content/FedHealthDP_Project/data/split/breast_cancer_X_test.csv',
        '/content/FedHealthDP_Project/data/split/breast_cancer_y_test.csv',
        breast_feature_columns,
        'diagnosis'
    )

    # 特征选择
    X_breast_test, selected_importances, selected_feature_names = select_features(X_breast_test, y_breast_test, feature_names, num_features=10)

    # 加载训练好的模型
    # 加载训练好的模型
    models = []
    for i in range(3):  # 假设已经训练了3个模型
        model = DeeperNN(X_breast_test.shape[1])
        model.load_state_dict(torch.load(f'/content/FedHealthDP_Project/models/global_model_{i}.pth'))
        model.eval()
        models.append(model)

    # 定义模型权重
    weights = [0.4, 0.3, 0.3]  # 根据各模型的性能进行调整

    # 评估乳腺癌模型
    logging.info("Evaluating Ensemble Breast Cancer Model")
    best_threshold_breast = find_best_threshold(ensemble_predict(models, X_breast_test, weights).cpu().detach(), y_breast_test)
    evaluate_ensemble(models, X_breast_test, y_breast_test, threshold=best_threshold_breast, output_dir='breast_cancer_results', weights=weights)

    # 加载乳腺癌训练数据并进行交叉验证
    logging.info("Cross-validating Breast Cancer Model")
    X_breast_train, y_breast_train, feature_names = preprocess_data(
        '/content/FedHealthDP_Project/data/split/breast_cancer_X_train.csv',
        '/content/FedHealthDP_Project/data/split/breast_cancer_y_train.csv',
        breast_feature_columns,
        'diagnosis'
    )
    X_breast_train, selected_importances, selected_feature_names = select_features(X_breast_train, y_breast_train, feature_names, num_features=10)
    cross_validate_model(X_breast_train, y_breast_train, feature_names, output_dir='breast_cancer_results')

    # 评估经典机器学习模型
    logging.info("Evaluating Classical Models")
    evaluate_classical_models(X_breast_train.numpy(), y_breast_train.numpy(), X_breast_test.numpy(), y_breast_test.numpy(), output_dir='breast_cancer_results')

    # 绘制特征重要性图并保存到本地文件
    plt.figure()
    plt.barh(selected_feature_names, selected_importances)
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('Top 10 Feature Importances')
    plt.tight_layout()
    plt.savefig(os.path.join('breast_cancer_results', 'feature_importance.png'))
    plt.close()

except Exception as e:
    logging.error(f"An error occurred: {e}")
