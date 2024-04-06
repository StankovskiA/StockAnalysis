import itertools
from constants import DATA_DIR, ORIGINAL_FILE
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from dtw import *

df = pd.read_csv(os.path.join(DATA_DIR, ORIGINAL_FILE))

print('Ensure data is loaded correctly')
print(df.head())
print("\n\n")

print('Examine nulls')
print(df.isnull().sum())
print("\n\n")

print('Examine outliers, range of values, etc.')
print(df.describe())
print("\n\n")

print('Examine data types')
print(df.dtypes)
print("\n\n")

print('Examine date range of date column')
df['date'] = pd.to_datetime(df['date'])
print(df['date'].min(), df['date'].max())
print("\n\n")

valores_distintos = df['Name'].unique()
print(valores_distintos)

df = df.drop(['Name'], axis=1)
print(df)

# Crear un objeto StandardScaler
scaler = StandardScaler()

# Normalizar el DataFrame
# Copiar el DataFrame original para preservar los datos originales
df_normalized = df.copy()
df_normalized[df_normalized.columns[1:]] = scaler.fit_transform(
    df_normalized[df_normalized.columns[1:]])

plt.plot(df_normalized['date'], df_normalized['open'],
         label='Open', color='blue')
plt.plot(df_normalized['date'], df_normalized['high'],
         label='High', color='green')
plt.plot(df_normalized['date'], df_normalized['low'], label='Low', color='red')
plt.plot(df_normalized['date'], df_normalized['close'],
         label='Close', color='orange')
plt.plot(df_normalized['date'], df_normalized['volume'],
         label='Volume', color='black')

plt.title('Stock market price analysis')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.legend()
plt.show()

correlation_matrix = df_normalized.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlación')
plt.show()


def maximum_shearing_correlation_distance(dataframe, column1, column2):
    series1 = dataframe[column1].values
    series2 = dataframe[column2].values

    # Inicialization
    max_distance = np.inf
    max_shift = 0

    max_date = None

    # Calcular la distancia para cada desplazamiento
    for shift, date in enumerate(dataframe['date']):
        distance = np.sum(np.abs(series1 - np.roll(series2, shift)))
        if distance < max_distance:
            max_distance = distance
            max_shift = shift
            max_date = date

    return max_distance, max_shift, max_date


# Create a list with all possible combinations of columns
column_combinations = list(itertools.combinations(
    df_normalized.columns[1:], 2))  # Exclude the 'date' column

# Calculate the maximum shearing correlation distance for each column combination
for column_pair in column_combinations:
    max_dist, shift, max_date = maximum_shearing_correlation_distance(
        df_normalized, column_pair[0], column_pair[1])
    print("Maximum shearing correlation distance between '{}' and '{}':".format(
        column_pair[0], column_pair[1]), max_dist)
    print("Shifting:", shift)
    print("Date:", max_date)
    print()


print("We stay with the time series of open and volume")
series1 = df_normalized['open'].values
series2 = df_normalized['volume'].values

alignment = dtw(series1, series2, keep_internals=True)
distance = alignment.distance

# Obtener la alineación entre las dos series
alignment.plot(type="twoway")
print(distance)
