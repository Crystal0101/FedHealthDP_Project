from pyexpat import model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score # type: ignore

from tensorflow_federated.python.learning.models.serialization_test import preprocess # type: ignore

def evaluate_model(model, test_data, test_labels):
    y_pred = model.predict(test_data)
    accuracy = accuracy_score(test_labels, y_pred > 0.5)
    precision = precision_score(test_labels, y_pred > 0.5)
    recall = recall_score(test_labels, y_pred > 0.5)
    f1 = f1_score(test_labels, y_pred > 0.5)
    auc = roc_auc_score(test_labels, y_pred)
    
    return accuracy, precision, recall, f1, auc

# 示例调用
test_data, test_labels = preprocess('data/mimic_test_data.csv')
metrics = evaluate_model(model, test_data, test_labels)
print(metrics)
