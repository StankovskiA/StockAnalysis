from constants import DATA_DIR, ORIGINAL_FILE
import pandas as pd
import os

df = pd.read_csv(os.path.join(DATA_DIR, ORIGINAL_FILE))

print('Ensure data is loaded correctly')
print(df.head())
print()
print()

print('Examine nulls')
print(df.isnull().sum())
print()
print()

print('Examine outliers, range of values, etc.')
print(df.describe())
print()
print()

print('Examine data types')
print(df.dtypes)
print()
print()

print('Examine date range of date column')
df['Date'] = pd.to_datetime(df['date'])
print(df['date'].min(), df['date'].max())
print()
print()
