# train_model.py

# 1. Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

# 2. Load Dataset
DATA_PATH = 'crop_pollution_dataset.csv'
MODEL_PATH = 'crop_pollution_model.pkl'

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Dataset not found at path: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# 3. Data Preprocessing
X = df[['production_tonnes', 'fertilizer_use_kg_per_hectare', 'pesticide_use_kg_per_hectare']]
y = df['pollution_index']

# 4. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Save the trained model
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully to 'crop_pollution_model.pkl'")
