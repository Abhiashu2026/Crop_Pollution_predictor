# Crop Pollution & Sustainable Crop Recommendation System

Major Project | NIT Jalandhar

This project is a Streamlit-based decision-support system for sustainable agriculture. It combines:

- crop pollution prediction
- state and district level dashboarding
- Punjab-first crop recommendation
- MSP-backed financial indicators
- Hindi/English farmer advisory interface
- model comparison and accuracy reporting
- data reliability documentation

The app is now presented as a major-project dashboard. Punjab is the default state, with all Punjab districts included, while the dataset also covers multiple Indian states for scalable comparison.

## Key Features

- Pollution Index prediction from crop production, fertilizer use and pesticide use.
- Estimated CO2 emissions per hectare.
- Sustainable crop recommendation for selected state, district and year.
- Punjab-first district crop advisory with all Punjab districts.
- Hindi and English language support.
- State-wise and district-wise sustainability dashboard.
- Model comparison using R2, MAE and RMSE.
- Data provenance documentation for report and PPT.

## App Sections

1. Overview
2. Pollution Predictor
3. Crop Recommendation
4. State/District Dashboard
5. Model Performance
6. Data Reliability

## Dataset

The expanded dataset is generated at:

```text
data/state_district_crop_environment_dataset.csv
```

Generate it with:

```bash
python scripts/build_major_dataset.py
```

The dataset contains:

- state
- district
- crop
- year
- data status
- production tonnes
- fertilizer use kg/hectare
- pesticide use kg/hectare
- soil nutrient fields
- MSP
- estimated yield
- expected net return
- water stress
- CO2 estimate
- pollution index
- sustainability score

Rows are marked as:

- Official-aligned historical
- Provisional estimate
- Model-estimated scenario

This distinction is important for scientific honesty in the report and viva.

## Data Reliability

See:

```text
DATA_SOURCES.md
```

Main reliability sources include:

- Ministry of Agriculture and Farmers Welfare / PIB crop estimates
- data.gov.in fertilizer consumption datasets
- Government MSP releases
- Soil Health Card references
- IPCC greenhouse gas methodology

## Run Locally

```bash
pip install -r requirements.txt
python scripts/build_major_dataset.py
streamlit run app.py
```

## Deployment

This repository can be deployed directly on Streamlit Cloud. The app entry point is:

```text
app.py
```

## Scientific Limitation

The project is a decision-support prototype. State-level government data is used where available, while district/crop expansion and 2025/2026 values are labelled as model-estimated scenarios. Final field deployment should integrate real farm-level yield, input cost, soil health card and mandi price data.
