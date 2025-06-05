# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Configure the Streamlit page
st.set_page_config(page_title="Crop Pollution Predictor", layout="centered")

# Load the trained model
try:
    model = pickle.load(open('crop_pollution_model.pkl', 'rb'))
except FileNotFoundError:
    st.error("❌ Model file 'crop_pollution_model.pkl' not found!")
    st.stop()

# Load the dataset
try:
    df = pd.read_csv('crop_pollution_dataset_enhanced.csv')
except FileNotFoundError:
    st.error("❌ Dataset file 'crop_pollution_dataset_enhanced.csv' not found!")
    st.stop()

# Title of the app
st.title('🌾 Crop Production Pollution Prediction App')

st.markdown("""
Welcome to the **Crop Pollution Prediction App**!  
This tool uses a machine learning model to estimate the *Pollution Index* and calculate **CO₂ emissions** based on crop production inputs.

👨‍🌾 Let’s work towards sustainable agriculture!
""")

# ℹ️ Pollution Index Explanation
with st.expander("ℹ️ What is Pollution Index & CO₂ Emissions?"):
    st.markdown("""
    ### Pollution Index
    The **Pollution Index (PI)** reflects how much pollution is potentially caused by growing a crop, based on:
    - **Fertilizer use** (kg per hectare)
    - **Pesticide use** (kg per hectare)

    ### CO₂ Emissions (Estimated)
    CO₂ emissions are calculated based on known emission factors:
    - Fertilizers emit ~**1.3 kg CO₂** per **1 kg** used  
    - Pesticides emit ~**5.1 kg CO₂** per **1 kg** used  

    This gives an estimate of greenhouse gas emissions from farming practices.

    #### 🚦 Risk Levels:
    - 🟢 **Low Risk**: Pollution Index < 40  
    - 🟡 **Medium Risk**: 40 ≤ PI < 70  
    - 🔴 **High Risk**: PI ≥ 70  
    """)

# ------------------------------
# 🚜 User Input Section
# ------------------------------
st.header("🔧 Input Parameters")

state = st.selectbox(
    '📍 Select State',
    df['state'].unique(),
    help="Choose the state where the crop is grown."
)

crop = st.selectbox(
    '🌿 Select Crop',
    df['crop'].unique(),
    help="Select the crop type you want to analyze."
)

year = st.selectbox(
    '📅 Select Year',
    sorted(df['year'].unique()),
    help="Choose the year of production for analysis."
)

production = st.number_input(
    '📦 Production (in tonnes)',
    min_value=100,
    max_value=10000,
    step=100,
    help="Total crop production quantity (in tonnes)."
)

fertilizer_use = st.number_input(
    '🌿 Fertilizer Use (kg/hectare)',
    min_value=50,
    max_value=300,
    step=10,
    help="Amount of fertilizer used per hectare. Higher values may lead to more pollution and emissions."
)

pesticide_use = st.number_input(
    '🧪 Pesticide Use (kg/hectare)',
    min_value=5,
    max_value=50,
    step=1,
    help="Amount of pesticide used per hectare. Impacts pollution index and CO₂ emissions."
)

# Create input dataframe
input_data = pd.DataFrame({
    'production_tonnes': [production],
    'fertilizer_use_kg_per_hectare': [fertilizer_use],
    'pesticide_use_kg_per_hectare': [pesticide_use]
})

# ------------------------------
# 🔮 Prediction Section
# ------------------------------
if st.button('🔍 Predict Pollution Index & CO₂ Emissions'):
    try:
        prediction = model.predict(input_data)
        pollution_index = prediction[0]

        # CO2 Emission Calculation
        CO2_FERTILIZER_FACTOR = 1.3  # kg CO2 per kg fertilizer
        CO2_PESTICIDE_FACTOR = 5.1   # kg CO2 per kg pesticide
        estimated_emissions = (fertilizer_use * CO2_FERTILIZER_FACTOR) + (pesticide_use * CO2_PESTICIDE_FACTOR)

        # Risk Levels
        if pollution_index < 40:
            risk_level = "🟢 Low Risk"
            color = "green"
        elif pollution_index < 70:
            risk_level = "🟡 Medium Risk"
            color = "orange"
        else:
            risk_level = "🔴 High Risk"
            color = "red"

        # Display results
        st.markdown(f"""
        ### 🌍 Estimated Pollution Index: **{pollution_index:.2f}**  
        <span style="color:{color}; font-weight:bold; font-size:18px;">{risk_level}</span>

        ### 💨 Estimated CO₂ Emissions: **{estimated_emissions:.2f} kg CO₂/hectare**
        """, unsafe_allow_html=True)

        st.subheader("📄 Summary of Input & Output")
        result_df = input_data.copy()
        result_df['state'] = state
        result_df['crop'] = crop
        result_df['year'] = year
        result_df['Predicted_Pollution_Index'] = pollution_index
        result_df['Estimated_CO2_Emissions_kg_per_hectare'] = estimated_emissions
        st.dataframe(result_df)

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# ------------------------------
# 📊 Dataset Exploration
# ------------------------------
st.markdown("---")
st.header("📊 Explore the Crop Pollution Dataset")

if st.checkbox('📂 Show Raw Dataset'):
    st.dataframe(df)

if st.checkbox('🌍 State-wise Mean Pollution Index'):
    st.bar_chart(df.groupby('state')['pollution_index'].mean())

if st.checkbox('🌿 Crop-wise Pollution Index (Boxplot)'):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='crop', y='pollution_index')
    st.pyplot(fig)

if st.checkbox('📈 Yearly Trend of Pollution Index'):
    year_plot = df.groupby('year')['pollution_index'].mean().reset_index()
    fig, ax = plt.subplots()
    sns.lineplot(data=year_plot, x='year', y='pollution_index', marker='o')
    st.pyplot(fig)

if st.checkbox('🧠 Show Correlation Heatmap'):
    st.subheader("🔗 Feature Correlation")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[['production_tonnes', 'fertilizer_use_kg_per_hectare', 'pesticide_use_kg_per_hectare', 'pollution_index']].corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# ------------------------------
# Footer
# ------------------------------
st.markdown("""
--- 
Minor Project | NIT Jalandhar  
""")
