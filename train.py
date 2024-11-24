import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load dataset
data = pd.read_csv('data/housing.csv')

# Preprocess dataset
data = data.select_dtypes(include=['number']).dropna()  # Use numeric columns and drop NaN

# Debug: Inspect dataset structure
print("Dataset Info:")
print(data.info())
print("Dataset Description:")
print(data.describe())

# Ensure the target column exists
if 'SalePrice' in data.columns:
    y = data['SalePrice']
    X = data.drop('SalePrice', axis=1)
else:
    print("Warning: 'SalePrice' column not found. Using the last column as target.")
    y = data.iloc[:, -1]  # Use the last column as the target if 'SalePrice' is not present
    X = data.iloc[:, :-1]

# Debug: Inspect target variable and features
print("Features (X) Sample:")
print(X.head())
print("Target (y) Sample:")
print(y.head())

# Shuffle data to avoid ordering bias
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Debug: Check dataset sizes
print(f"Training Data Size: {X_train.shape[0]}")
print(f"Testing Data Size: {X_test.shape[0]}")

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

# Debug: Print predictions and actual values
print(f"Predictions: {predictions[:5]}")
print(f"Actual Values: {y_test.values[:5]}")
print(f"Mean Absolute Error: {mae}")

# Save model
joblib.dump(model, 'house_price_model.pkl')
print("Model saved as 'house_price_model.pkl'.")
