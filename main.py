import argparse

from src.cli import run_prediction_cli
from src.artifacts import (
    artifacts_are_compatible,
    load_training_artifacts,
    load_training_metadata,
    save_training_artifacts,
    training_artifacts_exist,
)
from src.data.data import prepare_data
from src.evaluate import evaluate_model
from src.train import train_model
from src.constants import COLS_TO_NORMALIZE, FEATURE_COLUMNS
from src.predictor import Predictor


def parse_args():
    """Lee opciones de ejecución desde la consola."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Fuerza reentrenar el modelo aunque existan artefactos guardados.",
    )
    parser.add_argument(
        "--artifact-dir",
        default="artifacts",
        help="Directorio donde se guardan o cargan los artefactos.",
    )

    return parser.parse_args()


def main():
    """Ejecuta el flujo principal de preparación, entrenamiento y evaluación."""

    args = parse_args()

    # Parámetros e hiperparámetros
    file_path = "data/dataset.csv"
    target_column = "label"
    random_state = 8
    train_ratio = 0.8
    epochs = 20
    learning_rate = 0.001

    artifact_dir = args.artifact_dir
    force_retrain = args.retrain
    training_config = {
        "target_column": target_column,
        "train_ratio": train_ratio,
        "random_state": random_state,
        "epochs": epochs,
        "learning_rate": learning_rate,
    }

    should_train = force_retrain
    if training_artifacts_exist(artifact_dir) and not force_retrain:
        metadata = load_training_metadata(artifact_dir)
        should_train = not artifacts_are_compatible(
            metadata,
            FEATURE_COLUMNS,
            COLS_TO_NORMALIZE,
            training_config,
        )
    else:
        should_train = True

    if should_train:
        X_train, X_test, y_train, y_test, scaler = prepare_data(
            file_path=file_path,
            cols_to_normalize=COLS_TO_NORMALIZE,
            target_column=target_column,
            feature_columns=FEATURE_COLUMNS,
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

        metadata = {
            "threshold": selected_threshold,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "feature_columns": FEATURE_COLUMNS,
            "cols_to_normalize": COLS_TO_NORMALIZE,
            "training_config": training_config,
        }
        save_training_artifacts(model, scaler, metadata, artifact_dir)
        print(f"Artefactos guardados en: {artifact_dir}")
    else:
        model, scaler, metadata = load_training_artifacts(artifact_dir)
        selected_threshold = metadata["threshold"]

        print(f"Modelo cargado desde caché: {artifact_dir}")
        print(f"Umbral cargado: {selected_threshold:.4f}")
        print()

    predictor = Predictor(model, scaler, threshold=selected_threshold)
    run_prediction_cli(predictor)


if __name__ == "__main__":
    main()
