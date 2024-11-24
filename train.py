import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load dataset
data = pd.read_csv('data/housing.csv')

# Preprocess dataset
data = data.select_dtypes(include=['number']).dropna()  # Use numeric columns and drop NaN
X = data.drop('SalePrice', axis=1, errors='ignore')  # Replace 'SalePrice' if your target column has a different name
y = data['SalePrice'] if 'SalePrice' in data.columns else data.iloc[:, -1]  # Ensure target column is correct

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {mae}")

# Save model
joblib.dump(model, 'house_price_model.pkl')
