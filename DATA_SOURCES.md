# Data Sources and Reliability

This project separates **observed/official-aligned data**, **rule-derived indicators**, and **projected scenarios**. This is important because the public datasets for crop production, fertilizer use, pesticide use, and district-level soil nutrients are not always released in the same format or for the same year.

## Data Provenance Table

| Data Area | Source / Authority | Used For | Year Coverage in App | Reliability Level | Notes |
|---|---|---|---|---|---|
| Crop production | Ministry of Agriculture & Farmers Welfare / Directorate of Economics and Statistics / PIB final estimates | State-level crop production trend and production baseline | 2019-20 to 2023-24 official-aligned; 2024-25 onward projected | High for official years | Final estimates are government-published. App projections are labelled separately. |
| Fertilizer use | data.gov.in, Department of Fertilizers state/UT-wise demand, supply and consumption | Fertilizer use feature and pollution load | 2019-20 to 2023-24 official-aligned; 2024-25 onward projected | High for official years | Used as state-level baseline; district/crop values are derived from state baseline and crop multipliers. |
| Pesticide use | Directorate of Plant Protection, Quarantine & Storage / data.gov.in / Ministry agriculture references | Pesticide use feature and pollution load | Up to latest public official/provisional records; projections after latest release | Medium to high | Public pesticide data is often state-level. District/crop values are model-derived and labelled. |
| MSP | Government of India MSP releases / PIB / CACP-linked public releases | MSP, gross value and financial stability | 2026-27 MSP values where available | High | MSP is policy support price, not guaranteed realized mandi price for every farmer. |
| Soil nutrient parameters | User-provided Punjab soil nutrient dataset plus agronomic nutrient suitability ranges | Punjab district crop recommendation | District rows from provided CSV; other states use state-level synthetic baselines | Medium | Punjab district values are direct project data. Other state district values are placeholders until official district soil cards are integrated. |
| Emission factors | IPCC methodology and agricultural carbon footprint literature | CO2-equivalent estimate from fertilizer and pesticide use | Model formula | Medium | Used for educational decision-support, not regulatory carbon accounting. |

## Key Official Links for Report / PPT

- PIB final estimates of major agricultural crops 2023-24: https://pib.gov.in/PressReleasePage.aspx?PRID=2058534
- data.gov.in fertilizer consumption state/UT-wise 2019-20 to 2023-24: https://www.data.gov.in/resource/stateut-wise-details-demand-supply-and-consumption-all-fertilizer-2019-20-2023-24
- PIB MSP for Kharif crops, Marketing Season 2026-27: https://www.pib.gov.in/PressReleasePage.aspx?PRID=2260617
- IPCC 2006 Guidelines for National Greenhouse Gas Inventories: https://www.ipcc-nggip.iges.or.jp/public/2006gl/
- Soil Health Card scheme reference: https://soilhealth.dac.gov.in/

## Reliability Statement for Viva

The system is a **major project decision-support prototype**. It uses government-published sources for state-level agricultural context where available, then derives crop- and district-level scenario rows using transparent multipliers and labels them as derived/projected. Therefore, model outputs should be interpreted as advisory estimates rather than official prescriptions.

## Why Projection Is Used for 2025/2026

Final official agricultural statistics are generally released after the crop year. For 2025 and 2026, the app marks values as **Projected scenario** wherever final government datasets are not yet available. This avoids presenting forecasts as official data.
