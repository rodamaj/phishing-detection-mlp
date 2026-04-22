from typing import Tuple

import pandas as pd
import random


def split_train_test(
    df: pd.DataFrame, train_ratio: float = 0.5, random_state: int = 8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Divide un DataFrame en conjuntos de entrenamiento y prueba."""

    # Lista de índices y mezcla aleatoria con semilla para reproducibilidad
    indices = list(df.index)
    random.seed(random_state)
    random.shuffle(indices)

    # Punto de corte
    cut = int(len(indices) * train_ratio)

    # Separar índices
    train_idx = indices[:cut]
    test_idx = indices[cut:]

    # Crear datasets
    train_df = df.loc[train_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)

    # train_df.to_csv("data/dataset.train.csv", index=False)
    # test_df.to_csv("data/dataset.test.csv", index=False)

    return train_df, test_df
