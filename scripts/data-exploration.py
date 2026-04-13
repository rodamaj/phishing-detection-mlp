# Generates histograms, correlation heatmap, and class distribution for the phishing dataset.

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar dataset
df = pd.read_csv("../data/Dataset.csv")

# Make plots directory if it doesn't exist
os.makedirs("../img", exist_ok=True)

# Structure and summary
print(df.info())
print(df.describe())

# Null values
print(df.isnull().sum())

# Histograms

# Group 1: lengths
cols1 = ['url_len', 'dom_len', 'tld_len', 'path_len', 'query_len']
df[cols1].hist(figsize=(10,6))
plt.tight_layout()
plt.savefig("../img/hist_longitudes.png")
plt.close()

# Group 2: counts
cols2 = ['letter_cnt','digit_cnt','special_cnt','dot_cnt','dash_cnt','under_cnt']
df[cols2].hist(figsize=(10,6))
plt.tight_layout()
plt.savefig("../img/hist_conteos.png")
plt.close()

# Group 3: ratios
cols3 = ['letter_ratio','digit_ratio','spec_ratio']
df[cols3].hist(figsize=(8,4))
plt.tight_layout()
plt.savefig("../img/hist_ratios.png")
plt.close()

# Group 4: binaries + label
cols4 = ['is_ip','is_https','label']
df[cols4].hist(figsize=(6,4))
plt.tight_layout()
plt.savefig("../img/hist_binarios.png")
plt.close()

# Correlation heatmap
df_num = df.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(10,8))
sns.heatmap(df_num.corr(), annot=False, cmap='coolwarm')
plt.title("Matriz de correlación")
plt.tight_layout()
plt.savefig("../img/heatmap.png")
plt.close()

# Phishing count
conteo = df['label'].value_counts().sort_index()
total = conteo.sum()

# Bar graph for class distribution
plt.figure(figsize=(6,4))
ax = conteo.plot(kind='bar')

plt.xticks([0,1], ['No Phishing (0)', 'Phishing (1)'])
plt.xlabel('Clase')
plt.ylabel('Cantidad')
plt.title('Distribución de clases')
y_max = conteo.max()
plt.ylim(0, y_max * 1.15)

# Labels
for i, v in enumerate(conteo):
    porcentaje = (v / total) * 100
    ax.text(i, v + (y_max * 0.03), f"{v} ({porcentaje:.1f}%)",
            ha='center', fontsize=10)

plt.tight_layout()
plt.savefig("../img/class_distribution.png")
plt.close()