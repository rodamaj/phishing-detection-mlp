from pathlib import Path
from typing import Tuple, Union

import pandas as pd

from .folds import split_train_test
from .normalization import normalize_data


def load_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Carga un dataset desde un archivo CSV.
    """

    return pd.read_csv(file_path)


def split_features_target(
    df: pd.DataFrame, target_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separa el dataset en variables de entrada (X) y variable objetivo (y).
    """

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y


def prepare_data(
    file_path: Union[str, Path],
    cols_to_normalize: list,
    target_column: str,
    train_ratio: float = 0.5,
    random_state: int = 8,
):
    """
    Pipeline completo de preparación de datos:
    1. Carga
    2. Normalización
    3. División en conjuntos de entrenamiento y prueba
    4. Separación X / y
    """

    df = load_data(file_path)
    df, scaler = normalize_data(df, cols_to_normalize)
    train_data, test_data = split_train_test(
        df, train_ratio=train_ratio, random_state=random_state
    )

    X_train, y_train = split_features_target(train_data, target_column)
    X_test, y_test = split_features_target(test_data, target_column)

    return X_train, X_test, y_train, y_test, scaler
