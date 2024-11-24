import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load dataset
data = pd.read_csv('data/housing.csv')

# Preprocess dataset
data = data.select_dtypes(include=['number']).dropna()  # Use numeric columns and drop NaN

# Shuffle dataset to avoid ordering bias
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Ensure the target column exists
if 'SalePrice' in data.columns:
    y = data['SalePrice']
    X = data.drop('SalePrice', axis=1)
else:
    print("Warning: 'SalePrice' column not found. Using the last column as target.")
    y = data.iloc[:, -1]
    X = data.iloc[:, :-1]

# Debug: Inspect features and target
print("Sample Features (X):")
print(X.head())
print("Sample Target (y):")
print(y.head())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Debug: Check for overlapping rows
overlap = pd.merge(X_train, X_test, how='inner')
print(f"Number of overlapping rows between train and test: {len(overlap)}")

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

# Debug: Inspect predictions vs actual values
print("Sample Predictions vs Actual Values:")
print(pd.DataFrame({'Predicted': predictions[:5], 'Actual': y_test.values[:5]}))

print(f"Mean Absolute Error: {mae}")

# Save model
joblib.dump(model, 'house_price_model.pkl')
print("Model saved as 'house_price_model.pkl'.")
