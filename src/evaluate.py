from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true, y_pred, threshold: float) -> dict:
    """Calcula métricas de clasificación binarias para un umbral dado."""

    # La matriz de confusión resume aciertos y errores por clase.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0.0, 1.0]).ravel()

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "predicted_positives": int(y_pred.sum()),
    }


def evaluate_model(model, X, y, threshold: float) -> dict:
    """Evalúa el modelo usando un umbral fijo sobre el conjunto indicado."""

    # Ajusta la forma de y para que coincida con la salida del modelo.
    y_true = y.reshape(-1, 1)
    # El modelo devuelve probabilidades entre 0 y 1 para cada URL.
    y_prob = model.predict(X, verbose=0).reshape(-1, 1)
    # El umbral convierte esas probabilidades en clases binarias.
    y_pred = (y_prob >= threshold).astype(float)

    return compute_metrics(y_true, y_pred, float(threshold))
