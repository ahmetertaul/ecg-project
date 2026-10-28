"""Training loop for the Keras models."""

import time

from sklearn.metrics import f1_score
from tensorflow import keras

from src.augment import AugmentedBatches


def set_seed(seed):
    """Seed Python, NumPy and TensorFlow random generators."""
    keras.utils.set_random_seed(seed)


class MacroF1Callback(keras.callbacks.Callback):
    """Computes macro F1 on the validation set after each epoch and adds it to the logs as val_macro_f1."""

    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_val, batch_size=1024, verbose=0).argmax(axis=1)
        logs["val_macro_f1"] = f1_score(self.y_val, y_pred, average="macro")


def train_model(model, X_tr, y_tr, X_val, y_val, epochs=50, batch_size=128, lr=1e-3,
                class_weight=None, patience=8, verbose=2, augment=None):
    """Compile and train a model. Early stopping keeps the weights with the best validation macro F1.

    augment: optional dict of AugmentedBatches arguments (e.g. seed, noise_std). When given, the training
    batches are augmented on the fly.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        # must come first so that val_macro_f1 is in the logs for the other callbacks
        MacroF1Callback(X_val, y_val),
        keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=patience,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4),
    ]

    if augment is None:
        train_data = {"x": X_tr, "y": y_tr, "batch_size": batch_size}
    else:
        train_data = {"x": AugmentedBatches(X_tr, y_tr, batch_size, **augment)}

    start = time.time()
    history = model.fit(
        **train_data,
        validation_data=(X_val, y_val),
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=verbose,
    )
    train_time = time.time() - start
    return history, train_time
