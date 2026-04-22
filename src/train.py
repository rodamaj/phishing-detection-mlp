import random

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from src.evaluate import compute_metrics
from src.model import create_model


def train_model(X, y, epochs=20, learning_rate=0.001, batch_size=32, patience=3, random_state=8):
    """Entrena el modelo, elige el mejor umbral en validación y devuelve los datos clave."""

    # Fija semillas para que el entrenamiento sea reproducible entre ejecuciones.
    random.seed(random_state)
    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    # Reserva una parte del conjunto de entrenamiento para validación.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Construye y configura el modelo con pérdida y optimizador estándar.
    model = create_model(X_train.shape[1])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # Early stopping detiene el entrenamiento cuando la validación deja de mejorar.
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
            )
        ],
        verbose=1,
    )

    # Recupera la mejor época observada en validación para reportarla al final.
    val_losses = history.history["val_loss"]
    best_val_loss = min(val_losses)
    best_epoch = val_losses.index(best_val_loss) + 1

    # Obtiene probabilidades en validación para buscar el umbral más conveniente.
    y_val_true = y_val.reshape(-1, 1)
    y_val_prob = model.predict(X_val, verbose=0).reshape(-1, 1)
    best_metrics = None

    # Prueba varios umbrales y se queda con el que mejor balancea precisión y recall.
    for threshold in np.arange(0.1, 0.91, 0.05):
        metrics = compute_metrics(
            y_val_true,
            (y_val_prob >= threshold).astype(float),
            float(threshold),
        )

        if best_metrics is None or (
            metrics["f1"],
            metrics["precision"],
            metrics["recall"],
        ) > (
            best_metrics["f1"],
            best_metrics["precision"],
            best_metrics["recall"],
        ):
            best_metrics = metrics

    return model, best_epoch, best_val_loss, best_metrics
