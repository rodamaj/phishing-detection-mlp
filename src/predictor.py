# src/predictor.py

from ast import parse
from cProfile import label
import numpy as np
import pandas as pd
import re
import math
from collections import Counter
from urllib.parse import urlparse
from src.data.normalization import apply_scaler
from src.constants import COLS_TO_NORMALIZE


class Predictor:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    # Extrae el TLD y su longitud
    def _get_tld_len(self, parsed):
        domain_parts = parsed.netloc.split(".")
        if len(domain_parts) > 1:
            return len(domain_parts[-1])
        return 0

    # Cuenta subdominios (si hay más de 2 partes en el dominio)
    def _get_subdomain_count(self, parsed):
        domain_parts = parsed.netloc.split(".")
        if len(domain_parts) <= 2:
            return 0
        return len(domain_parts) - 2

    # Calcula la entropía de la URL
    def _get_entropy(self, url: str):
        if not url:
            return 0

        counter = Counter(url)
        length = len(url)

        entropy = 0
        for count in counter.values():
            p = count / length
            entropy -= p * math.log2(p)

        return entropy

    # Extractor de las demás features y preparación del input para el modelo
    def _extract_features(self, url: str) -> dict:

        parsed = urlparse(url)

        # Variables base
        total_len = len(url)
        dom_len = len(parsed.netloc)
        tld_len = self._get_tld_len(parsed)
        subdom_cnt = self._get_subdomain_count(parsed)
        is_ip = 1 if re.match(r"\d+\.\d+\.\d+\.\d+", parsed.netloc) else 0
        is_https = 1 if parsed.scheme == "https" else 0
        letter_cnt = sum(c.isalpha() for c in url)
        digit_cnt = sum(c.isdigit() for c in url)
        special_cnt = sum(not c.isalnum() for c in url)
        eq_cnt = url.count("=")
        qm_cnt = url.count("?")
        amp_cnt = url.count("&")
        dot_cnt = url.count(".")
        dash_cnt = url.count("-")
        under_cnt = url.count("_")
        slash_cnt = url.count("/")
        path_len = len(parsed.path)
        query_len = len(parsed.query)
        entropy = self._get_entropy(url)

        # Ratios (evitar división por cero)
        if total_len > 0:
            letter_ratio = letter_cnt / total_len
            digit_ratio = digit_cnt / total_len
            spec_ratio = special_cnt / total_len
        else:
            letter_ratio = digit_ratio = spec_ratio = 0

        # Diccionario final
        features = {
            "url_len": total_len,
            "dom_len": dom_len,
            "is_ip": is_ip,
            "tld_len": tld_len,
            "subdom_cnt": subdom_cnt,
            "letter_cnt": letter_cnt,
            "digit_cnt": digit_cnt,
            "special_cnt": special_cnt,
            "eq_cnt": eq_cnt,
            "qm_cnt": qm_cnt,
            "amp_cnt": amp_cnt,
            "dot_cnt": dot_cnt,
            "dash_cnt": dash_cnt,
            "under_cnt": under_cnt,
            "letter_ratio": letter_ratio,
            "digit_ratio": digit_ratio,
            "spec_ratio": spec_ratio,
            "is_https": is_https,
            "slash_cnt": slash_cnt,
            "entropy": entropy,
            "path_len": path_len,
            "query_len": query_len,
        }

        return features

    # Preparación de datos
    def _prepare_input(self, url: str):
        f = self._extract_features(url)

        X = np.array([[
            f["url_len"],
            f["dom_len"],
            f["tld_len"],
            f["subdom_cnt"],
            f["letter_cnt"],
            f["digit_cnt"],
            f["special_cnt"],
            f["eq_cnt"],
            f["qm_cnt"],
            f["amp_cnt"],
            f["dot_cnt"],
            f["dash_cnt"],
            f["under_cnt"],
            f["letter_ratio"],
            f["digit_ratio"],
            f["spec_ratio"],
            f["is_https"],
            f["slash_cnt"],
            f["entropy"],
            f["path_len"],
            f["query_len"],
            f["is_ip"],
        ]], dtype=np.float32)

        # Aplicar scaler a columnas que corresponden a las features numéricas
        X[:, :len(COLS_TO_NORMALIZE)] = self.scaler.transform(X[:, :len(COLS_TO_NORMALIZE)])

        return X

    # Predicción
    def predict(self, url: str, threshold: float = 0.5):
        X = self._prepare_input(url)

        prob = self.model.predict(X, verbose=0)[0][0]
        pred = 1 if prob >= threshold else 0

        return prob, pred

    # Interacción de usuario
    def interaction(self):
        while True:
            url = input("Ingrese URL (o 'salir'): ")

            if url.lower() == "salir":
                break

            prob, pred = self.predict(url)

            print(f"\nProbabilidad no phishing: {prob:.6f}")
            print(f"Probabilidad phishing: {1 - prob:.6f}")
            print()