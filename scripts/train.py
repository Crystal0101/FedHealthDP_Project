import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import tensorflow as tf
from scikeras.wrappers import KerasClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV, cross_val_score
from scipy.stats import uniform
import logging
from sklearn.metrics import accuracy_score
import yaml
from preprocess import preprocess
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def preprocess_data():
    features, labels = preprocess()
    logging.info(f'Features shape: {features.shape}, Labels shape: {labels.shape}')
    
    # 合并稀有类别
    labels = labels.astype(str)  # 确保所有标签都是字符串类型
    diagnosis_counts = labels.value_counts()
    top_diagnoses = diagnosis_counts.head(10).index
    labels = labels.apply(lambda x: x if x in top_diagnoses else 'Other')
    labels_encoded, class_names, label_encoder = encode_labels(labels)
    logging.info(f"Number of classes after keeping top 10 categories: {len(class_names)}")

    # 确保时间列为datetime类型
    if 'admittime' in features.columns:
        features['admittime'] = pd.to_datetime(features['admittime'], errors='coerce')
    if 'dischtime' in features.columns:
        features['dischtime'] = pd.to_datetime(features['dischtime'], errors='coerce')
    if 'deathtime' in features.columns:
        features['deathtime'] = pd.to_datetime(features['deathtime'], errors='coerce')
    if 'dob' in features.columns:
        features['dob'] = pd.to_datetime(features['dob'], errors='coerce')

    # 特征工程：增加新的特征
    if 'dischtime' in features.columns and 'admittime' in features.columns:
        features['length_of_stay'] = (features['dischtime'] - features['admittime']).dt.days
    if 'admittime' in features.columns and 'dob' in features.columns:
        features['age'] = features['admittime'].dt.year - features['dob'].dt.year
    if 'deathtime' in features.columns:
        features['is_dead'] = features['deathtime'].apply(lambda x: 0 if x == pd.Timestamp('2199-01-01 00:00:00') else 1)

    # 移除常量特征
    selector = VarianceThreshold()
    features = selector.fit_transform(features)
    logging.info(f'Features shape after removing constant features: {features.shape}')
    
    # 再次检查数据有效性，确保无常量特征或无效值
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        raise ValueError("The feature matrix contains NaN or infinite values.")
    
    # 动态选择最佳特征数量
    pipe = Pipeline([
        ('select', SelectKBest(f_classif)),
        ('model', LogisticRegression(max_iter=1000))
    ])
    
    param_grid = {'select__k': range(10, min(features.shape[1], 51), 10)}
    search = GridSearchCV(pipe, param_grid, cv=5, verbose=3)
    search.fit(features, labels_encoded)
    
    best_k = search.best_params_['select__k']
    logging.info(f'Best number of features: {best_k}')

    selector = SelectKBest(f_classif, k=best_k)
    features = selector.fit_transform(features, labels_encoded)
    
    logging.info(f'Features shape after feature selection: {features.shape}')
    return features, labels_encoded, class_names, label_encoder

def encode_labels(labels):
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    class_names = label_encoder.classes_
    return labels_encoded, class_names, label_encoder

def split_data(features, labels_encoded):
    return train_test_split(features, labels_encoded, test_size=0.2, random_state=42)

def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    logging.info(f'After standardization - X_train shape: {X_train.shape}, X_test shape: {X_test.shape}')
    return X_train, X_test, scaler

def one_hot_encode_labels(y_train, y_test, num_classes):
    y_train_categorical = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test_categorical = tf.keras.utils.to_categorical(y_test, num_classes)
    return y_train_categorical, y_test_categorical

def create_multiclass_model(input_shape, num_classes, optimizer='adam', dropout_rate=0.5, init_mode='uniform', learning_rate=0.01):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='relu', kernel_initializer=init_mode, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(16, activation='relu', kernel_initializer=init_mode, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    optimizer_instance = tf.keras.optimizers.get(optimizer)
    optimizer_instance.learning_rate = learning_rate
    model.compile(optimizer=optimizer_instance, loss='categorical_crossentropy', metrics=['accuracy'])
    logging.info(f'Model created with input shape: {input_shape}, num classes: {num_classes}')
    return model

class EpochLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logging.info(f'Epoch {epoch + 1}: loss = {logs["loss"]}, accuracy = {logs["accuracy"]}, val_loss = {logs["val_loss"]}, val_accuracy = {logs["val_accuracy"]}')

class WrappedNN(KerasClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = kwargs.pop('num_classes', None)
        
    def fit(self, X, y, **kwargs):
        y_categorical = tf.keras.utils.to_categorical(y, self.num_classes)
        return super().fit(X, y_categorical, callbacks=[EpochLogger()], **kwargs)
    
    def predict(self, X, **kwargs):
        predictions = super().predict(X, **kwargs)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X, **kwargs):
        proba = super().predict_proba(X, **kwargs)
        if proba.ndim == 1:
            proba = np.expand_dims(proba, axis=1)
        if proba.shape[1] == 1:  # Binary classification case
            proba = np.hstack((1 - proba, proba))
        return proba

def predict_disease(model, input_data, scaler, label_encoder):
    input_data = scaler.transform([input_data])
    logging.info(f'Input data shape for prediction: {input_data.shape}')
    prediction = model.predict(input_data)
    if len(prediction.shape) > 1:
        predicted_class = np.argmax(prediction, axis=1)
    else:
        predicted_class = np.argmax(prediction)
    predicted_disease = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_disease


if __name__ == "__main__":
    config = load_config()

    features, labels_encoded, class_names, label_encoder = preprocess_data()
    logging.info(f"Unique labels: {np.unique(labels_encoded)}")
    logging.info(f"Number of classes: {len(class_names)}")

    X_train, X_test, y_train, y_test = split_data(features, labels_encoded)
    logging.info(f'Training data shape: {X_train.shape}')
    logging.info(f'Test data shape: {X_test.shape}')
    X_train, X_test, scaler = standardize_data(X_train, X_test)
    logging.info(f'Scaled training data shape: {X_train.shape}')
    logging.info(f'Scaled test data shape: {X_test.shape}')

    num_classes = len(class_names)
    y_train_categorical, y_test_categorical = one_hot_encode_labels(y_train, y_test, num_classes)
    logging.info(f'One-hot encoded training labels shape: {y_train_categorical.shape}')
    logging.info(f'One-hot encoded test labels shape: {y_test_categorical.shape}')

    multiclass_model = KerasClassifier(
        model=create_multiclass_model,
        model__input_shape=X_train.shape[1],
        model__num_classes=num_classes,
        verbose=0,  # 设置为0以避免Keras打印冗长输出
        batch_size=32,
        epochs=1,
        optimizer='adam',
        dropout_rate=0.5,
        init_mode='uniform',
        learning_rate=0.001
    )

    param_distributions = {
        'model__optimizer': ['SGD', 'Adam', 'RMSprop', 'Nadam'],
        'model__dropout_rate': uniform(0.1, 0.6),
        'model__init_mode': ['uniform', 'lecun_uniform', 'normal', 'he_normal'],
        'model__learning_rate': uniform(0.0001, 0.1)
    }

    logging.info('Starting RandomizedSearchCV...')
    random_search = RandomizedSearchCV(estimator=multiclass_model, param_distributions=param_distributions, n_iter=100, n_jobs=-1, cv=5, verbose=3, error_score='raise', random_state=42)
    random_search_result = random_search.fit(X_train, y_train_categorical)
    logging.info(f"Best: {random_search_result.best_score_} using {random_search_result.best_params_}")

    best_multiclass_model = random_search_result.best_estimator_
    logging.info(f'Best model input shape: {best_multiclass_model.model_.input_shape}')
    logging.info(f'Best model output shape: {best_multiclass_model.model_.output_shape}')

    multiclass_accuracy = best_multiclass_model.score(X_test, y_test_categorical)
    logging.info(f'Multiclass Model Test Accuracy: {multiclass_accuracy}')

    rf_model = RandomForestClassifier(n_estimators=200, random_state=42, verbose=1)  # 设置为1以打印训练进度
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, verbosity=1)  # 设置为1以打印训练进度
    gb_model = GradientBoostingClassifier(n_estimators=200, random_state=42, verbose=1)  # 设置为1以打印训练进度
    wrapped_nn_model = WrappedNN(
        model=create_multiclass_model,
        model__input_shape=X_train.shape[1],
        model__num_classes=num_classes,
        verbose=1,  # 设置为1以打印训练进度
        batch_size=32,
        epochs=1,
        optimizer='adam',
        dropout_rate=0.5,
        init_mode='uniform',
        learning_rate=0.001
    )

    # Train each model and check output shape
    models = [wrapped_nn_model, rf_model, xgb_model, gb_model]
    for model in models:
        logging.info(f'Starting training for model: {model.__class__.__name__}')
        model.fit(X_train, y_train)
        logging.info(f'Training completed for model: {model.__class__.__name__}')
        logging.info(f'Model score: {model.score(X_test, y_test)}')

    ensemble_model = VotingClassifier(estimators=[
        ('nn', wrapped_nn_model),
        ('rf', rf_model),
        ('xgb', xgb_model),
        ('gb', gb_model)
    ], voting='soft', n_jobs=-1)

    logging.info('Starting training for ensemble model...')
    ensemble_model.fit(X_train, y_train)
    ensemble_accuracy = accuracy_score(y_test, ensemble_model.predict(X_test))
    logging.info(f'Ensemble Model Test Accuracy: {ensemble_accuracy}')