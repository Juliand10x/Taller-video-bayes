import pandas as pd
import matplotlib.pyplot as plt

# Cargar el CSV recién generado con 10,000 draws
df = pd.read_csv('uber_summary.csv', index_col=0)

params = ['beta_wait', 'beta_commute']
labels = ['Tiempo de Espera\n(beta_wait)', 'Efecto Hora Pico\n(beta_commute)']

means = df.loc[params, 'mean'].values
hdi_lower = df.loc[params, 'hdi_3%'].values
hdi_upper = df.loc[params, 'hdi_97%'].values

# Errores asimétricos para errorbar manual
errors = [means - hdi_lower, hdi_upper - means]

# Estilizado
plt.figure(figsize=(9, 4.5))
plt.style.use('seaborn-v0_8-whitegrid')

# Graficar coeficientes
plt.errorbar(means, range(len(params)), xerr=errors, 
             fmt='o', color='#0f4c81', ecolor='#5c92c8', 
             elinewidth=5, capsize=8, capthick=2, markersize=12)

# Configuración del eje Y
plt.yticks(range(len(params)), labels, fontsize=12, fontweight='bold', color='#333333')
plt.gca().invert_yaxis() # Para que wait quede arriba

# Línea base en ZERO
plt.axvline(x=0, color='#d9534f', linestyle='--', linewidth=2.5, label='Efecto Nulo (0)')

plt.title('Intervalos de Credibilidad al 95% (H.D.I.)\n¿Qué variables impactan los viajes en Uber Pool?', 
          pad=15, fontsize=14, fontweight='bold', color='#111111')
plt.xlabel('Efecto calculado en cantidad de viajes (Magnitud)', fontsize=12)

# Anotaciones explícitas de los valores para mayor profesionalismo
for i, (m, l, u) in enumerate(zip(means, hdi_lower, hdi_upper)):
    plt.text(m, i - 0.2, f'M: {m:.1f}\n[{l:.1f}, {u:.1f}]', 
             ha='center', va='bottom', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

plt.legend(loc='lower left', fontsize=11)
plt.tight_layout()

# Sobreescribimos la gráfica que no le gusta al usuario
plt.savefig('uber_forest_plot.png', dpi=300)
plt.close()
print("Gráfica forest salvada!")
