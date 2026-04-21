from typing import Tuple

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def normalize_data(
    df: pd.DataFrame, cols_to_normalize: list
) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    Normaliza las columnas numéricas dadas del dataset utilizando Min-Max Scaling.
    """

    scaler = MinMaxScaler()
    df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

    # df.to_csv("data/dataset.normalized.csv", index=False)

    return df, scaler
