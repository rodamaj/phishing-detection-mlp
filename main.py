from src.data.data import prepare_data
from src.constants import COLS_TO_NORMALIZE


def main():
    file_path = "data/dataset.csv"
    target_column = "label"

    X_train, X_test, y_train, y_test, scaler = prepare_data(
        file_path=file_path,
        cols_to_normalize=COLS_TO_NORMALIZE,
        target_column=target_column,
        train_ratio=0.5,
        random_state=8,
    )

    print("Datos preparados:")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)


if __name__ == "__main__":
    main()
