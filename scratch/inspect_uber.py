import pandas as pd
import os

file_path = 'data/BaseUBER.xlsx'
if os.path.exists(file_path):
    df = pd.read_excel(file_path)
    print("Columns:", df.columns.tolist())
    print("\nTypes:\n", df.dtypes)
    print("\nFirst 5 rows:\n", df.head())
else:
    print(f"File {file_path} not found.")
