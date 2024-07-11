import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import syft as sy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 配置日志
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 加载数据
features = pd.read_csv("features.csv")
labels = pd.read_csv("labels.csv")

# 数据集划分与标准化
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 定义虚拟客户端类
class VirtualClient:
    def __init__(self, id):
        self.id = id

# 创建虚拟客户端
clients = [VirtualClient(id=f"client_{i}") for i in range(5)]

# 将数据分配给虚拟客户端
client_data = []
for i, client in enumerate(clients):
    client_X_train = torch.tensor(X_train[i::5], dtype=torch.float32)
    client_y_train = torch.tensor(y_train.values[i::5], dtype=torch.float32).reshape(-1, 1)
    client_data.append((client_X_train, client_y_train))

# 模型定义
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

# 联邦学习训练
def train(model, device, client_data, optimizer, epoch, epsilon, sensitivity):
    model.train()
    epoch_loss = 0.0
    for data, target in client_data:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.BCELoss()(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(data)  # accumulate the loss for this batch
        logging.info(f'Train Epoch: {epoch} | Batch Loss: {loss.item():.4f}')
    
    # 加噪
    for param in model.parameters():
        param.data = add_noise(param.data, epsilon, sensitivity)
    
    # 计算平均损失
    epoch_loss /= len(client_data)
    logging.info(f'Train Epoch: {epoch} | Average Loss: {epoch_loss:.4f}')

# 差分隐私机制
def add_noise(tensor, epsilon, sensitivity):
    noise = torch.randn(tensor.size()) * (sensitivity / epsilon)
    return tensor + noise

# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X_train.shape[1]
model = LogisticRegressionModel(input_dim).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
epsilon = 1.0
sensitivity = 1.0

for epoch in range(1, 11):
    train(model, device, client_data, optimizer, epoch, epsilon, sensitivity)

# 评估模型
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in zip(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)):
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += nn.BCELoss()(output, target).item()
        pred = output.round()
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(X_test)
accuracy = correct / len(X_test)
logging.info(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')
