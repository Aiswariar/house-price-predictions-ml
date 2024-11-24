import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset path
dataset_path = 'data/Housing.csv'  # Relative path

# Check if the dataset exists
if os.path.exists(dataset_path):
    print(f"Dataset found at: {os.path.abspath(dataset_path)}")
else:
    print(f"Error: Dataset not found at {os.path.abspath(dataset_path)}")
    exit()  # Exit the script if the file is not found

# Load dataset
data = pd.read_csv(dataset_path)

# Check basic info about the dataset
print(f"Dataset shape: {data.shape}")
print(f"Columns: {data.columns}")
print(f"Descriptive statistics of target variable:")
print(data['price'].describe())

# Check for missing values and duplicate rows
print("Missing values:")
print(data.isnull().sum())

print("Duplicate rows:")
print(data.duplicated().sum())

# Preprocess dataset: select only numeric columns and drop rows with missing values
data = data.select_dtypes(include=['number']).dropna()  # Use numeric columns and drop NaN

# Check if 'price' column exists
if 'price' not in data.columns:
    print("Error: 'price' column not found in the dataset.")
    exit()

# Define features and target
X = data.drop('price', axis=1)
y = data['price']

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Visualize the relationship between 'area' and 'price'
plt.figure(figsize=(8,6))
sns.scatterplot(x=data['area'], y=data['price'])
plt.title('Area vs Price')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()

# Train a simpler model (Linear Regression) first to check if it's a data/model issue
print("\nTraining with Linear Regression...")
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Evaluate Linear Regression model
predictions = linear_model.predict(X_test)
mae_linear = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error (Linear Regression): {mae_linear}")

# Train the RandomForestRegressor model
print("\nTraining with RandomForestRegressor...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate RandomForestRegressor model
rf_predictions = rf_model.predict(X_test)
mae_rf = mean_absolute_error(y_test, rf_predictions)
print(f"Mean Absolute Error (RandomForestRegressor): {mae_rf}")

# Save the best model (RandomForestRegressor in this case)
joblib.dump(rf_model, 'house_price_model.pkl')
print(f"Model saved as 'house_price_model.pkl'")

# Check predictions vs true values for RandomForestRegressor
print("\nPredictions (first 10 values):", rf_predictions[:10])
print("True values (first 10):", y_test[:10].values)   
