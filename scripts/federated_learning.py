import tensorflow_federated as tff
from models.model import model_fn
from models.dp_optimizer import create_dp_optimizer
from utils.data_utils import create_tf_dataset, preprocess

# 创建联邦数据集
train_data1, train_labels1 = preprocess('data/mimic_data_part1.csv')
train_data2, train_labels2 = preprocess('data/mimic_data_part2.csv')
train_dataset1 = create_tf_dataset(train_data1, train_labels1)
train_dataset2 = create_tf_dataset(train_data2, train_labels2)

federated_train_data = [tff.simulation.datasets.ClientData.from_clients_and_tf_fn(
    client_ids=['client1', 'client2'],
    serializable_dataset_fn=lambda x: create_tf_dataset(x))]

dp_optimizer_fn = lambda: create_dp_optimizer(1.0, 0.5, 32, 0.01)

# 定义带差分隐私的联邦平均算法
iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=dp_optimizer_fn)

state = iterative_process.initialize()

# 运行联邦训练过程
NUM_ROUNDS = 10
for round_num in range(NUM_ROUNDS):
    state, metrics = iterative_process.next(state, federated_train_data)
    print(f'Round {round_num + 1}, Metrics: {metrics}')

# 评估全局模型
evaluation = tff.learning.build_federated_evaluation(model_fn)
train_metrics = evaluation(state.model, federated_train_data)
print(f'Train Metrics: {train_metrics}')
