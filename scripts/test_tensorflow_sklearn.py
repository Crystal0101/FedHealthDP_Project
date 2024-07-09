# test_tensorflow_sklearn.py

import tensorflow as tf
from keras import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# 加载数据
data = load_iris()
X = data.data
y = data.target

# 数据集拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 创建模型函数
def create_model(optimizer='adam', init_mode='uniform'):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(8, input_shape=(4,), kernel_initializer=init_mode, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 使用KerasClassifier进行包装
model = KerasClassifier(build_fn=create_model, verbose=0)

# 定义超参数网格
param_grid = {
    'batch_size': [10, 20],
    'epochs': [10, 50],
    'optimizer': ['SGD', 'Adam'],
    'init_mode': ['uniform', 'lecun_uniform']
}

# 使用GridSearchCV进行超参数搜索
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X_train, y_train)

# 打印最佳参数和结果
print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")

# 使用最佳参数进行模型训练
best_model = grid_result.best_estimator_
loss, accuracy = best_model.score(X_test, y_test)
print(f'Test Accuracy: {accuracy}')
