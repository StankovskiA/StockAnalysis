# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from constants import DATA_DIR, ORIGINAL_FILE
import matplotlib.pyplot as plt
import pandas as pd
import os

# Load the preprocessed dataset
def load_data(data_dir, file_name):
    df = pd.read_csv(os.path.join(data_dir, file_name))
    # Convert 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'])
    return df

# Split data into train and test sets
def train_test_split_data(df, test_size=0.2):
    train_df, test_df = train_test_split(df, test_size=test_size, shuffle=False)
    return train_df, test_df

# Feature Engineering
def feature_engineering(df):
    # Perform any feature engineering if necessary
    return df

# Model Training
def train_model(train_df):
    # Example: Train a simple linear regression model
    X_train = train_df[['open']]  # Input features
    y_train = train_df['close']    # Target variable

    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Model Evaluation
def evaluate_model(model, test_df):
    X_test = test_df[['open']]  # Input features
    y_test = test_df['close']    # Target variable

    # Predictions
    y_pred = model.predict(X_test)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error (MSE):", mse)

    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(test_df['date'], y_test, label='Actual', color='blue')
    plt.plot(test_df['date'], y_pred, label='Predicted', color='red')
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def main():
    # Load data
    df = load_data(DATA_DIR, ORIGINAL_FILE)

    # Split data into train and test sets
    train_df, test_df = train_test_split_data(df)

    # Feature Engineering (if needed)
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)

    # Train model
    model = train_model(train_df)

    # Evaluate model
    evaluate_model(model, test_df)

if __name__ == "__main__":
    main()
