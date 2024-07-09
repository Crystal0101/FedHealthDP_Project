from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score # type: ignore

def evaluate_model(model, test_data, test_labels):
    y_pred = model.predict(test_data)
    accuracy = accuracy_score(test_labels, y_pred > 0.5)
    precision = precision_score(test_labels, y_pred > 0.5)
    recall = recall_score(test_labels, y_pred > 0.5)
    f1 = f1_score(test_labels, y_pred > 0.5)
    auc = roc_auc_score(test_labels, y_pred)
    
    return accuracy, precision, recall, f1, auc
