def run_prediction_cli(predictor):
    """Ejecuta el ciclo interactivo de clasificación de URLs."""

    while True:
        url = input("Ingrese URL (o 'salir'): ")

        if url.lower() == "salir":
            break

        prob, pred = predictor.predict(url)

        print(f"\nProbabilidad phishing: {prob:.6f}")
        print(f"Probabilidad no phishing: {1 - prob:.6f}")
        print(f"Umbral de decisión usado: {predictor.threshold:.4f}")
        print(f"Predicción: {'Phishing' if pred == 1 else 'No Phishing'}")
        print()
