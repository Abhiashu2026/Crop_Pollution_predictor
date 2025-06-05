import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Data Loading
df = pd.read_csv('crop_pollution_dataset.csv')

# Step 2: Data Preprocessing
# Handle missing values, if any
df = df.dropna()

# Step 3: Feature Selection and Splitting Data into X and y
X = df.drop(['pollution_level'], axis=1)  # Features (All except pollution level)
y = df['pollution_level']  # Target variable (pollution level)

# Step 4: Splitting Data into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Training (using Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predictions
y_pred = model.predict(X_test)

# Step 7: Model Evaluation (R-squared and MSE)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'R-squared: {r2}')
print(f'Mean Squared Error: {mse}')

# Step 8: Save the trained model
with open('crop_pollution_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Visualizing the Predictions vs Actual
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=2)
plt.xlabel("Actual Pollution Level")
plt.ylabel("Predicted Pollution Level")
plt.title("Actual vs Predicted Pollution Level")
plt.show()
