import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuración de estilo
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300

# Cargar dataset
df = pd.read_excel('BaseUBER.xlsx')

# Parsear variables necesarias
df['wait_time_num'] = df['wait_time'].str.extract('(\d+)').astype(float)
df['treat_num'] = df['treat'].astype(int)
df['commute_num'] = df['commute'].astype(int)

# Info basica
print("--- Info Básica ---")
print(df.info())
print("\n--- Descripción de variables ---")
print(df[['trips_pool', 'wait_time_num', 'commute_num']].describe())

# 1. Distribución de trips_pool
plt.figure(figsize=(8, 5))
sns.histplot(df['trips_pool'], bins=20, kde=True, color='blue', alpha=0.6)
plt.title('Distribución de Viajes Compartidos (trips_pool)', fontsize=14)
plt.xlabel('Cantidad de Viajes (trips_pool)', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.tight_layout()
plt.savefig('eda_trips_pool.png')
plt.close()

# 2. trips_pool vs wait_time (Boxplot o scatter)
plt.figure(figsize=(8, 5))
sns.boxplot(x='wait_time_num', y='trips_pool', data=df, palette='Set2')
plt.title('Viajes Compartidos vs Tiempo de Espera', fontsize=14)
plt.xlabel('Tiempo de Espera (minutos)', fontsize=12)
plt.ylabel('Viajes Compartidos (trips_pool)', fontsize=12)
plt.tight_layout()
plt.savefig('eda_wait_time.png')
plt.close()

# 3. trips_pool vs commute
plt.figure(figsize=(7, 5))
sns.violinplot(x='commute', y='trips_pool', data=df, palette='muted', inner="quartile")
plt.title('Distribución de Viajes Compartidos por Horario', fontsize=14)
plt.xlabel('Es hora pico (Commute)', fontsize=12)
plt.ylabel('Viajes Compartidos (trips_pool)', fontsize=12)
plt.tight_layout()
plt.savefig('eda_commute.png')
plt.close()

print("\nEDA generado correctamente. Imágenes guardadas.")
