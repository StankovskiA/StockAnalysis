from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from dtw import *
import itertools
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def data_exploration(df: pd.DataFrame) -> None:
    logger.info('Ensure data is loaded correctly')
    logger.info(df.head())

    logger.info('Examine nulls')
    logger.info(df.isnull().sum())

    logger.info('Examine outliers, range of values, etc.')
    logger.info(df.describe())

    logger.info('Examine data types')
    logger.info(df.dtypes)

    logger.info('Examine date range of date column')
    df['date'] = pd.to_datetime(df['date'])
    logger.info(f"{df['date'].min()} - {df['date'].max()}")

    valores_distintos = df['Name'].unique()
    logger.info(valores_distintos)

    df = df.drop(['Name'], axis=1)
    logger.info(df)

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
    plt.plot(df_normalized['date'], df_normalized['low'],
             label='Low', color='red')
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

    fig = go.Figure(data=[go.Candlestick(x=df['date'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'])])

    fig.update_layout(height=800)
    fig.show()

    correlation_matrix = df_normalized.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()

    # Create a list with all possible combinations of columns
    column_combinations = list(itertools.combinations(
        df_normalized.columns[1:], 2))  # Exclude the 'date' column

    # Calculate the maximum shearing correlation distance for each column combination
    for column_pair in column_combinations:
        max_dist, shift, max_date = maximum_shearing_correlation_distance(
            df_normalized, column_pair[0], column_pair[1])
        logger.info(f"Maximum shearing correlation distance between '{column_pair[0]}' and '{column_pair[1]}': {max_dist}")
        logger.info(f"Shifting: {shift}")
        logger.info(f"Date: {max_date}")

    logger.info("We stay with the time series of open and volume")
    series1 = df_normalized['open'].values
    series2 = df_normalized['volume'].values

    alignment = dtw(series1, series2, keep_internals=True)
    distance = alignment.distance

    # Obtener la alineación entre las dos series
    alignment.plot(type="twoway")
    logger.info(distance)

    # Closing prices line chart
    plt.plot(df['date'], df['close'], label='Close', zorder=1)

    for year in df['date'].dt.year.unique():
        if year != 2018:
            last_close = df[df['date'].dt.year == year]['close'].iloc[-1]
            plt.scatter(df['date'][df['date'].dt.year ==
                        year].iloc[-1], last_close, color='red', s=30)

    plt.title('Closing Price of Microsoft')
    plt.ylabel('Close')
    plt.xlabel('Date')
    plt.show()

    # Histogram of Closing Prices
    plt.figure(figsize=(10, 6))
    sns.histplot(df_normalized['close'], kde=True, bins=30)
    plt.title('Distribution of Closing Prices')
    plt.xlabel('Close Price')
    plt.ylabel('Frequency')
    plt.show()

    # Line chart of Stock Prices and Volume Traded
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Closing prices on the first y-axis (left)
    ax1.plot(df_normalized['date'], df_normalized['close'],
             color='blue', label='Closing Price')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Closing Price')
    ax1.tick_params('y')
    ax1.lines[0].set_linewidth(1.5)

    # Volume on the second y-axis (right)
    ax2 = ax1.twinx()
    ax2.plot(df_normalized['date'], df_normalized['volume'],
             color='orange', label='Volume')
    ax2.set_ylabel('Volume')
    ax2.tick_params('y')
    ax2.lines[0].set_linewidth(1.5)

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.title('Stock Prices and Volume Traded')
    plt.show()
