# src/predictor.py

import numpy as np
import pandas as pd
from src.data.normalization import apply_scaler
from src.constants import FEATURE_COLUMNS, COLS_TO_NORMALIZE
from src.features.url_features import extract_url_features


class Predictor:
    """Prepara URLs y usa el modelo entrenado para clasificarlas."""

    def __init__(self, model, scaler, threshold: float = 0.5):
        """Inicializa el predictor con el modelo, scaler y umbral de decisión."""

        self.model = model
        self.scaler = scaler
        self.threshold = threshold

    def _prepare_input(self, url: str):
        """Prepara una URL como arreglo numérico escalado para el modelo."""

        features = extract_url_features(url)
        X = pd.DataFrame([features], columns=FEATURE_COLUMNS)
        X = apply_scaler(X, self.scaler, COLS_TO_NORMALIZE)

        return X.to_numpy(dtype=np.float32)

    def predict(self, url: str, threshold: float | None = None):
        """Predice la probabilidad y clase para una URL."""

        X = self._prepare_input(url)
        threshold = self.threshold if threshold is None else threshold

        prob = float(self.model.predict(X, verbose=0)[0][0])
        pred = 1 if prob >= threshold else 0

        return prob, pred
