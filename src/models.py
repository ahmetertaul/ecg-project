"""Keras model definitions."""

from tensorflow import keras
from tensorflow.keras import layers

INPUT_LEN = 187
N_CLASSES = 5


def build_mlp(input_len=INPUT_LEN, n_classes=N_CLASSES, dropout=0.3):
    """Fully connected network on the raw 187-sample signal."""
    inputs = keras.Input(shape=(input_len,))
    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="mlp")


def build_simple_cnn(input_len=INPUT_LEN, n_classes=N_CLASSES, dropout=0.3):
    """Three Conv1D + MaxPool blocks followed by a small dense classifier."""
    inputs = keras.Input(shape=(input_len,))
    x = layers.Reshape((input_len, 1))(inputs)  # Conv1D expects (time, channels)
    for filters in [32, 64, 128]:
        x = layers.Conv1D(filters, kernel_size=5, padding="same", activation="relu")(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="simple_cnn")


def residual_block(x, filters=32, kernel_size=5):
    """Conv -> ReLU -> Conv, add the block input (skip connection), ReLU, then max pooling."""
    shortcut = x
    y = layers.Conv1D(filters, kernel_size, padding="same", activation="relu")(x)
    y = layers.Conv1D(filters, kernel_size, padding="same")(y)
    y = layers.Add()([y, shortcut])
    y = layers.Activation("relu")(y)
    return layers.MaxPooling1D(pool_size=5, strides=2)(y)


def build_resnet_cnn(input_len=INPUT_LEN, n_classes=N_CLASSES, n_blocks=5, filters=32):
    """Residual 1D CNN based on Kachuee et al. (2018)."""
    inputs = keras.Input(shape=(input_len,))
    x = layers.Reshape((input_len, 1))(inputs)
    x = layers.Conv1D(filters, kernel_size=5, padding="same")(x)
    for _ in range(n_blocks):
        x = residual_block(x, filters)
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="resnet_cnn")


def build_lstm(input_len=INPUT_LEN, n_classes=N_CLASSES, units=64):
    """Two stacked LSTM layers reading the beat one sample at a time."""
    inputs = keras.Input(shape=(input_len,))
    x = layers.Reshape((input_len, 1))(inputs)
    x = layers.LSTM(units, return_sequences=True)(x)
    x = layers.LSTM(units)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="lstm")
