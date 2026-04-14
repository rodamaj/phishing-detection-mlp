import pandas as pd
import random

# Cargar datos
df = pd.read_csv("../data/Dataset_normalizado.csv")

TRAIN_RATIO = 0.5
RANDOM_SEED = 8

# Lista de índices y mezcla aleatoria
indices = list(df.index)
random.seed(RANDOM_SEED)  # para reproducibilidad
random.shuffle(indices)

# Punto de corte
cut = int(len(indices) * TRAIN_RATIO)

# Separar índices
train_idx = indices[:cut]
test_idx = indices[cut:]

# Crear datasets
train_df = df.loc[train_idx].reset_index(drop=True)
test_df = df.loc[test_idx].reset_index(drop=True)

# Guardar en otro archivo CSV
train_df.to_csv("../data/Dataset_train.csv", index=False)
test_df.to_csv("../data/Dataset_test.csv", index=False)
