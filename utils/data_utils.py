import pandas as pd # type: ignore
import tensorflow as tf

def preprocess(data_path):
    data = pd.read_csv(data_path)
    features = data[['feature1', 'feature2', 'feature3']]
    labels = data['label']
    return features.values, labels.values

def create_tf_dataset(features, labels):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    return dataset.batch(32)
