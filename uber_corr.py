import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar los datos
try:
    df = pd.read_excel('BaseUBER.xlsx')
except Exception as e:
    print(f"Error loading BaseUBER.xlsx: {e}")
    exit(1)

# Preparar las variables de interés
# trips_pool es int64
# wait_time es str ('2 mins', '5 mins') -> lo convertiremos a int
# treat es bool
# commute es bool

df['wait_time_num'] = df['wait_time'].str.extract('(\d+)').astype(float)
df['treat_num'] = df['treat'].astype(int)
df['commute_num'] = df['commute'].astype(int)

vars_of_interest = ['trips_pool', 'wait_time_num', 'treat_num', 'commute_num']
df_subset = df[vars_of_interest]

# Renombrar para el gráfico si es necesario
df_subset.columns = ['trips_pool', 'wait_time', 'treat', 'commute']

# Calcular correlación
corr = df_subset.corr()
print("Correlation Matrix:")
print(corr)

# Generar gráfico
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Matriz de Correlación: Variables Uber Pool')
plt.tight_layout()
plt.savefig('uber_correlation.png', dpi=300)
print("Plot saved as uber_correlation.png")
