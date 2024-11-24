import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load dataset
try:
    data = pd.read_csv('data/Housing.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: Dataset not found at 'data/Housing.csv'.")
    exit()

# Check dataset content
print("Columns in Dataset:", data.columns)
if 'price' not in data.columns:
    print("Error: 'price' column not found in dataset.")
    exit()

print("Target Column Summary:")
print(data['price'].describe())

# Preprocess dataset
# Convert categorical columns to numeric using one-hot encoding
data = pd.get_dummies(data, drop_first=True)

# Split features and target
X = data.drop('price', axis=1)  # Features
y = data['price']  # Target

# Check dataset size
print(f"Dataset size: {data.shape}, Features size: {X.shape}, Target size: {y.shape}")

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape}, Testing set size: {X_test.shape}")

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print("Predictions vs Actuals:")
print(pd.DataFrame({'Predicted': predictions[:5], 'Actual': y_test.values[:5]}))

print(f"Mean Absolute Error: {mae}")

# Save model
joblib.dump(model, 'house_price_model.pkl')
print("Model saved as 'house_price_model.pkl'.")
