from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd

from src.constants import FEATURE_COLUMNS

from .folds import split_train_test
from .normalization import apply_scaler, normalize_data


def load_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """Carga un dataset desde un archivo CSV."""

    return pd.read_csv(file_path)


def split_features_target(
    df: pd.DataFrame, target_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separa el dataset en variables de entrada y variable objetivo."""

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y


def select_feature_columns(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_columns: list,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Conserva las features del modelo en el orden esperado."""

    missing_columns = [
        column
        for column in feature_columns
        if column not in X_train.columns or column not in X_test.columns
    ]
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas: {missing_columns}")

    X_train = X_train[feature_columns].copy()
    X_test = X_test[feature_columns].copy()

    return X_train, X_test


def prepare_data(
    file_path: Union[str, Path],
    cols_to_normalize: list,
    target_column: str,
    feature_columns: list = FEATURE_COLUMNS,
    train_ratio: float = 0.5,
    random_state: int = 8,
):
    """Prepara los datos cargando, dividiendo, separando variables, normalizando y convirtiendo a arreglos."""

    df = load_data(file_path)
    train_data, test_data = split_train_test(
        df, train_ratio=train_ratio, random_state=random_state
    )

    X_train, y_train = split_features_target(train_data, target_column)
    X_test, y_test = split_features_target(test_data, target_column)
    X_train, X_test = select_feature_columns(X_train, X_test, feature_columns)

    X_train, scaler = normalize_data(X_train, cols_to_normalize)
    X_test = apply_scaler(X_test, scaler, cols_to_normalize)

    X_train = X_train.to_numpy(dtype=np.float32)
    X_test = X_test.to_numpy(dtype=np.float32)
    y_train = y_train.to_numpy(dtype=np.float32)
    y_test = y_test.to_numpy(dtype=np.float32)

    return X_train, X_test, y_train, y_test, scaler
