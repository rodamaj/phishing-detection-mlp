import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Cargar datos
df = pd.read_csv("../data/Dataset.csv")

# Columnas a normalizar
cols_normalizar = [
    'url_len', 'dom_len', 'tld_len', 'subdom_cnt',
    'letter_cnt', 'digit_cnt', 'special_cnt',
    'eq_cnt', 'qm_cnt', 'amp_cnt',
    'dot_cnt', 'dash_cnt', 'under_cnt', 'slash_cnt',
    'entropy', 'path_len', 'query_len'
]

# No se normalizan las que son ratios porque ya están en el rango [0, 1]

scaler = MinMaxScaler()
df[cols_normalizar] = scaler.fit_transform(df[cols_normalizar])

# Guardar en otro archivo CSV
df.to_csv("../data/Dataset_normalizado.csv", index=False)

print(df.head())