from tensorflow import keras
from tensorflow.keras import layers


def build_cnn_lstm(input_shape: tuple[int, int, int], num_classes: int) -> keras.Model:
    """Compact CNN-LSTM for fixed-length MFCC feature sequences."""
    inputs = keras.Input(shape=input_shape, name="mfcc_sequence")
    x = layers.Reshape((input_shape[0], input_shape[1]))(inputs)
    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(96, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.LSTM(64, dropout=0.2)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="emotion_probabilities")(x)
    model = keras.Model(inputs, outputs, name="EmotionSense_CNN_LSTM")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
