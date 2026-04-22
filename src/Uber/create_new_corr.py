import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Create folder if it doesn't exist
os.makedirs('img/uber', exist_ok=True)

# Load data (using CSV if available for speed, else XLSX)
csv_path = 'data/BaseUBER.csv'
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    df = pd.read_excel('data/BaseUBER.xlsx')

# Preprocessing
if 'wait_time_num' not in df.columns:
    df['wait_time_num'] = df['wait_time'].str.extract('(\d+)').astype(float)

# Numeric conversion for correlation
cols_original = [
    'wait_time_num', 
    'treat', 
    'commute', 
    'trips_pool', 
    'trips_express', 
    'rider_cancellations', 
    'total_driver_payout', 
    'total_matches', 
    'total_double_matches'
]

df_corr = df[cols_original].copy()

# Map internal names to "Original Names" as seen in PDF
# The PDF uses the exact column names: wait_time, treat, commute, trips_pool, etc.
# We will use the exact column names minus the '_num' suffix I added.
df_corr.columns = [
    'wait_time', 
    'treat', 
    'commute', 
    'trips_pool', 
    'trips_express', 
    'rider_cancellations', 
    'total_driver_payout', 
    'total_matches', 
    'total_double_matches'
]

# Calculate Correlation
corr = df_corr.corr()

# Plot
plt.figure(figsize=(12, 10))
sns.set_theme(style="white")
heatmap = sns.heatmap(
    corr, 
    annot=True, 
    fmt=".2f", 
    cmap='coolwarm', 
    center=0,
    vmin=-1, 
    vmax=1,
    linewidths=.5, 
    cbar_kws={"shrink": .8}
)

plt.title('Matriz de Correlación: Variables Uber Pool', fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Save as 'uber_correlation.png' which is the name used in LaTeX
output_path = 'img/uber/uber_correlation.png'
plt.savefig(output_path, dpi=300)
print(f"Matrix saved to {output_path}")
