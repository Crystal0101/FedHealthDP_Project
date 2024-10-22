import tensorflow as tf
import tensorflow_federated as tff

def create_keras_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def model_fn():
    keras_model = create_keras_model(8)  # Adjust the input_shape based on the number of features
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=spec,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()])
import tensorflow as tf
import tensorflow_federated as tff

def create_keras_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def model_fn():
    keras_model = create_keras_model(8)  # Adjust the input_shape based on the number of features
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=spec,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()])
import tensorflow as tf
import tensorflow_federated as tff

def create_keras_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def model_fn():
    keras_model = create_keras_model(8)  # Adjust the input_shape based on the number of features
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=spec,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()])
