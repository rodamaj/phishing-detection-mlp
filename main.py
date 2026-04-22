from src.data.data import prepare_data
from src.evaluate import evaluate_model
from src.train import train_model
from src.constants import COLS_TO_NORMALIZE


def main():
    """Ejecuta el flujo principal de preparación, entrenamiento y evaluación."""

    # Parámetros e hiperparámetros
    file_path = "data/dataset.csv"
    target_column = "label"
    random_state = 8
    train_ratio = 0.8
    epochs = 20
    learning_rate = 0.001

    X_train, X_test, y_train, y_test, _ = prepare_data(
        file_path=file_path,
        cols_to_normalize=COLS_TO_NORMALIZE,
        target_column=target_column,
        train_ratio=train_ratio,
        random_state=random_state,
    )

    print("Datos preparados:")
    print("Forma de X_train:", X_train.shape)
    print("Forma de X_test:", X_test.shape)
    print("Forma de y_train:", y_train.shape)
    print("Forma de y_test:", y_test.shape)

    model, best_epoch, best_val_loss, validation_metrics = train_model(
        X=X_train,
        y=y_train,
        epochs=epochs,
        learning_rate=learning_rate,
        random_state=random_state,
    )

    selected_threshold = validation_metrics["threshold"]
    test_metrics = evaluate_model(model, X_test, y_test, threshold=selected_threshold)

    print()
    print("Resultados de entrenamiento:")
    print(f"Mejor época: {best_epoch:.4f}")
    print(f"Mejor pérdida de validación: {best_val_loss:.4f}")
    print(f"Umbral elegido en validación: {selected_threshold:.4f}")
    print()

    print("Resultados de validación (mejor umbral por F1):")
    for metric_name, value in validation_metrics.items():
        print(f"{metric_name}: {value:.4f}")
    print()

    print("Resultados de evaluación (usando el umbral elegido en validación):")
    for metric_name, value in test_metrics.items():
        print(f"{metric_name}: {value:.4f}")
    print()


if __name__ == "__main__":
    main()
