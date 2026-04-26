import tensorflow as tf


def create_model(input_dim: int):
    """Construye una red neuronal multicapa para clasificación binaria."""

    # Sequential permite apilar las capas en el mismo orden en que se ejecutan.
    model = tf.keras.Sequential(
        [
            # Define la forma esperada de cada fila de entrada.
            tf.keras.layers.Input(shape=(input_dim,)),
            # Primera capa densa para aprender patrones a partir de las features.
            tf.keras.layers.Dense(64, activation="relu"),
            # Dropout ayuda a reducir el sobreajuste durante entrenamiento.
            tf.keras.layers.Dropout(0.2),
            # Segunda capa oculta para refinar la representación aprendida.
            tf.keras.layers.Dense(32, activation="relu"),
            # Segunda regularización por dropout.
            tf.keras.layers.Dropout(0.2),
            # La salida sigmoide produce una probabilidad de phishing.
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    return model
