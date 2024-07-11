import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import logging
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to load client data
def load_client_data(num_clients):
    client_data = []
    for i in range(num_clients):
        client_features_path = f"/content/client_{i}_features.csv"
        client_labels_path = f"/content/client_{i}_labels.csv"
        
        if os.path.exists(client_features_path) and os.path.exists(client_labels_path):
            client_features = pd.read_csv(client_features_path)
            client_labels = pd.read_csv(client_labels_path)
            client_data.append((client_features, client_labels))
            logging.info(f"Loaded data for client {i}: features shape {client_features.shape}, labels shape {client_labels.shape}")
        else:
            logging.error(f"Files for client {i} not found.")
            raise FileNotFoundError(f"Files for client {i} not found.")
    return client_data

# Combine and split data
def combine_and_split_data(client_data):
    features = pd.concat([data[0] for data in client_data])
    labels = pd.concat([data[1] for data in client_data])
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

# Virtual client class
class VirtualClient:
    def __init__(self, id):
        self.id = id

# Enhanced model definition
class DeepNeuralNetworkModel(nn.Module):
    def __init__(self, input_dim):
        super(DeepNeuralNetworkModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# Differential privacy mechanism
def add_noise(tensor, epsilon, sensitivity):
    noise = torch.randn(tensor.size()) * (sensitivity / epsilon)
    return tensor + noise

# Federated learning training
def train(model, device, client_data, optimizer, criterion, epoch, epsilon, sensitivity):
    model.train()
    epoch_loss = 0.0
    for data, target in client_data:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(data)
        logging.info(f'Train Epoch: {epoch} | Batch Loss: {loss.item():.4f}')
    
    # Add noise
    for param in model.parameters():
        param.data = add_noise(param.data, epsilon, sensitivity)
    
    # Calculate average loss
    epoch_loss /= len(client_data)
    logging.info(f'Train Epoch: {epoch} | Average Loss: {epoch_loss:.4f}')

# Evaluate the model
def evaluate_model(model, device, X_test, y_test, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1).to(device)
        output = model(X_test)
        test_loss = criterion(output, y_test).item()
        pred = output.round()
        correct = pred.eq(y_test.view_as(pred)).sum().item()
        y_true = y_test.cpu().numpy()
        y_pred = output.cpu().numpy()
    
    accuracy = correct / len(X_test)
    roc_auc = roc_auc_score(y_true, y_pred)
    logging.info(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}')
    return accuracy, roc_auc

# Main script
if __name__ == "__main__":
    num_clients = 5
    client_data = load_client_data(num_clients)
    X_train, X_test, y_train, y_test = combine_and_split_data(client_data)
    
    clients = [VirtualClient(id=f"client_{i}") for i in range(num_clients)]
    client_data = []
    for i, client in enumerate(clients):
        client_X_train = torch.tensor(X_train[i::5], dtype=torch.float32)
        client_y_train = torch.tensor(y_train.values[i::5], dtype=torch.float32).reshape(-1, 1)
        client_data.append((client_X_train, client_y_train))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[1]

    # Hyperparameter grid
    learning_rates = [0.01, 0.001, 0.0001]
    batch_sizes = [32, 64, 128]

    best_accuracy = 0
    best_params = {}

    criterion = nn.BCELoss()

    for lr in learning_rates:
        for batch_size in batch_sizes:
            model = DeepNeuralNetworkModel(input_dim).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
            epsilon = 0.1  # Reduced noise for better accuracy
            sensitivity = 1.0

            for epoch in range(1, 21):  # Increased epochs for better training
                # Federated training with batch processing
                for i in range(0, len(client_data), batch_size):
                    batch_data = client_data[i:i+batch_size]
                    if batch_data:
                        train(model, device, batch_data, optimizer, criterion, epoch, epsilon, sensitivity)

            accuracy, roc_auc = evaluate_model(model, device, X_test, y_test, criterion)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {'learning_rate': lr, 'batch_size': batch_size}
                logging.info(f'New best params: {best_params}, Accuracy: {best_accuracy:.4f}, ROC AUC: {roc_auc:.4f}')

    logging.info(f'Best Hyperparameters: {best_params}, Best Accuracy: {best_accuracy:.4f}')
