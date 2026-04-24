from typing import Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler


def normalize_data(
    df: pd.DataFrame, cols_to_normalize: list
) -> Tuple[pd.DataFrame, StandardScaler]:
    """Normaliza columnas numéricas utilizando StandardScaler."""

    df = df.copy()
    scaler = StandardScaler()
    df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

    return df, scaler


def apply_scaler(
    df: pd.DataFrame, scaler: StandardScaler, cols_to_normalize: list
) -> pd.DataFrame:
    """Aplica un escalador ya entrenado sobre las columnas indicadas."""

    df = df.copy()
    df[cols_to_normalize] = scaler.transform(df[cols_to_normalize])

    return df
