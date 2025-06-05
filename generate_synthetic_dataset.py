import pandas as pd
import numpy as np

# Number of rows in the dataset
num_samples = 1000

# Random data generation for different crops
crops = ['Rice', 'Wheat', 'Maize', 'Barley', 'Soybean']
temperature = np.random.uniform(20, 40, num_samples)  # Random temperatures (20-40°C)
ph = np.random.uniform(5.5, 7.5, num_samples)  # Random pH values (5.5 - 7.5)
substrate_concentration = np.random.uniform(0.1, 2.0, num_samples)  # Random substrate concentration (mol/L)
inhibitor_concentration = np.random.uniform(0.0, 0.5, num_samples)  # Random inhibitor concentration (mol/L)
crop_type = np.random.choice(crops, num_samples)  # Random selection of crops

# Pollution level (this can be calculated using some formula or rule, here using a random scale)
pollution_level = np.random.uniform(10, 100, num_samples)  # Pollution level (1-100 scale)

# Create the DataFrame
data = {
    'crop': crop_type,
    'temperature': temperature,
    'pH': ph,
    'substrate_concentration': substrate_concentration,
    'inhibitor_concentration': inhibitor_concentration,
    'pollution_level': pollution_level
}

df = pd.DataFrame(data)

# Save the DataFrame to CSV
df.to_csv('synthetic_dataset.csv', index=False)
print("Dataset saved as 'synthetic_dataset.csv'")
