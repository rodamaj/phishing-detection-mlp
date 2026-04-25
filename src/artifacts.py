import json
import pickle
from pathlib import Path

from tensorflow.keras.models import load_model


MODEL_FILENAME = "model.keras"
SCALER_FILENAME = "scaler.pkl"
METADATA_FILENAME = "metadata.json"


def get_artifact_paths(artifact_dir: str | Path) -> dict:
    """Construye las rutas de los artefactos de entrenamiento."""

    artifact_dir = Path(artifact_dir)

    return {
        "dir": artifact_dir,
        "model": artifact_dir / MODEL_FILENAME,
        "scaler": artifact_dir / SCALER_FILENAME,
        "metadata": artifact_dir / METADATA_FILENAME,
    }


def _to_jsonable(value):
    """Convierte valores de numpy y colecciones anidadas a tipos serializables."""

    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]

    if hasattr(value, "item"):
        return value.item()

    return value


def training_artifacts_exist(artifact_dir: str | Path) -> bool:
    """Indica si existen todos los artefactos necesarios para cargar el modelo."""

    paths = get_artifact_paths(artifact_dir)

    return (
        paths["model"].exists()
        and paths["scaler"].exists()
        and paths["metadata"].exists()
    )


def load_training_metadata(artifact_dir: str | Path) -> dict:
    """Carga los metadatos guardados del entrenamiento."""

    paths = get_artifact_paths(artifact_dir)

    with paths["metadata"].open("r", encoding="utf-8") as file:
        return json.load(file)


def artifacts_are_compatible(
    metadata: dict,
    feature_columns: list,
    cols_to_normalize: list,
    training_config: dict | None = None,
) -> bool:
    """Verifica si los artefactos guardados coinciden con las columnas actuales."""

    columns_match = (
        metadata.get("feature_columns") == feature_columns
        and metadata.get("cols_to_normalize") == cols_to_normalize
    )

    if training_config is None:
        return columns_match

    return columns_match and metadata.get("training_config") == training_config


def save_training_artifacts(
    model,
    scaler,
    metadata: dict,
    artifact_dir: str | Path,
) -> None:
    """Guarda el modelo, scaler y metadatos para reutilizarlos."""

    paths = get_artifact_paths(artifact_dir)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    model.save(paths["model"])

    with paths["scaler"].open("wb") as file:
        pickle.dump(scaler, file)

    with paths["metadata"].open("w", encoding="utf-8") as file:
        json.dump(_to_jsonable(metadata), file, indent=2)


def load_training_artifacts(artifact_dir: str | Path):
    """Carga el modelo, scaler y metadatos guardados."""

    paths = get_artifact_paths(artifact_dir)
    model = load_model(paths["model"])

    with paths["scaler"].open("rb") as file:
        scaler = pickle.load(file)

    metadata = load_training_metadata(artifact_dir)

    return model, scaler, metadata
