from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential


def create_model(input_dim: int):
    """Construye una red neuronal multicapa para clasificación binaria."""

    # Sequential permite apilar las capas en el mismo orden en que se ejecutan.
    model = Sequential(
        [
            # Define la forma esperada de cada fila de entrada.
            Input(shape=(input_dim,)),
            # Primera capa densa para aprender patrones a partir de las features.
            Dense(64, activation="relu"),
            # Dropout ayuda a reducir el sobreajuste durante entrenamiento.
            Dropout(0.2),
            # Segunda capa oculta para refinar la representación aprendida.
            Dense(32, activation="relu"),
            # Segunda regularización por dropout.
            Dropout(0.2),
            # La salida sigmoide produce una probabilidad de phishing.
            Dense(1, activation="sigmoid"),
        ]
    )

    return model
