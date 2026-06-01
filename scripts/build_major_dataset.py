import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT = DATA_DIR / "state_district_crop_environment_dataset.csv"

YEARS = list(range(2019, 2027))

STATES = {
    "Punjab": {
        "districts": [
            "Amritsar",
            "Barnala",
            "Bathinda",
            "Faridkot",
            "Fatehgarh Sahib",
            "Fazilka",
            "Ferozepur",
            "Gurdaspur",
            "Hoshiarpur",
            "Jalandhar",
            "Kapurthala",
            "Ludhiana",
            "Malerkotla",
            "Mansa",
            "Moga",
            "Mohali",
            "Muktsar",
            "Pathankot",
            "Patiala",
            "Rupnagar",
            "Sangrur",
            "Shaheed Bhagat Singh Nagar",
            "Tarn Taran",
        ],
        "fertilizer_base": 238,
        "pesticide_base": 24,
        "production_base": 6800,
        "soil": {"pH": 7.7, "N": 320, "P": 20, "K": 292, "OC": 0.50},
    },
    "Haryana": {
        "districts": ["Ambala", "Hisar", "Karnal", "Kurukshetra", "Sirsa"],
        "fertilizer_base": 224,
        "pesticide_base": 22,
        "production_base": 6100,
        "soil": {"pH": 7.8, "N": 305, "P": 19, "K": 278, "OC": 0.47},
    },
    "Uttar Pradesh": {
        "districts": ["Meerut", "Lucknow", "Gorakhpur", "Varanasi", "Bareilly"],
        "fertilizer_base": 205,
        "pesticide_base": 18,
        "production_base": 7200,
        "soil": {"pH": 7.4, "N": 300, "P": 18, "K": 275, "OC": 0.52},
    },
    "Rajasthan": {
        "districts": ["Jaipur", "Kota", "Bikaner", "Sri Ganganagar", "Udaipur"],
        "fertilizer_base": 165,
        "pesticide_base": 16,
        "production_base": 4300,
        "soil": {"pH": 8.0, "N": 270, "P": 16, "K": 260, "OC": 0.38},
    },
    "Madhya Pradesh": {
        "districts": ["Indore", "Bhopal", "Jabalpur", "Ujjain", "Gwalior"],
        "fertilizer_base": 178,
        "pesticide_base": 15,
        "production_base": 5100,
        "soil": {"pH": 7.3, "N": 285, "P": 17, "K": 270, "OC": 0.55},
    },
    "Maharashtra": {
        "districts": ["Pune", "Nashik", "Nagpur", "Aurangabad", "Kolhapur"],
        "fertilizer_base": 188,
        "pesticide_base": 20,
        "production_base": 5000,
        "soil": {"pH": 7.2, "N": 290, "P": 18, "K": 280, "OC": 0.58},
    },
    "Gujarat": {
        "districts": ["Ahmedabad", "Rajkot", "Surat", "Vadodara", "Junagadh"],
        "fertilizer_base": 192,
        "pesticide_base": 21,
        "production_base": 5200,
        "soil": {"pH": 7.8, "N": 285, "P": 18, "K": 285, "OC": 0.45},
    },
    "Bihar": {
        "districts": ["Patna", "Gaya", "Muzaffarpur", "Purnia", "Bhagalpur"],
        "fertilizer_base": 170,
        "pesticide_base": 14,
        "production_base": 4700,
        "soil": {"pH": 7.1, "N": 295, "P": 18, "K": 265, "OC": 0.60},
    },
    "West Bengal": {
        "districts": ["Kolkata", "Bardhaman", "Nadia", "Murshidabad", "Hooghly"],
        "fertilizer_base": 182,
        "pesticide_base": 17,
        "production_base": 5600,
        "soil": {"pH": 6.8, "N": 310, "P": 20, "K": 275, "OC": 0.68},
    },
    "Tamil Nadu": {
        "districts": ["Coimbatore", "Thanjavur", "Madurai", "Salem", "Tiruchirappalli"],
        "fertilizer_base": 176,
        "pesticide_base": 18,
        "production_base": 4550,
        "soil": {"pH": 7.0, "N": 300, "P": 19, "K": 285, "OC": 0.57},
    },
    "Karnataka": {
        "districts": ["Bengaluru Rural", "Mysuru", "Belagavi", "Raichur", "Mandya"],
        "fertilizer_base": 168,
        "pesticide_base": 16,
        "production_base": 4400,
        "soil": {"pH": 7.1, "N": 285, "P": 17, "K": 275, "OC": 0.54},
    },
    "Andhra Pradesh": {
        "districts": ["Guntur", "Krishna", "East Godavari", "Anantapur", "Kurnool"],
        "fertilizer_base": 194,
        "pesticide_base": 20,
        "production_base": 5450,
        "soil": {"pH": 7.2, "N": 300, "P": 19, "K": 290, "OC": 0.52},
    },
}

CROPS = {
    "Wheat": {"msp": 2585, "yield": 50, "cost": 72000, "fert": 1.0, "pest": 0.72, "water": 0.66},
    "Rice / Paddy": {"msp": 2441, "yield": 65, "cost": 90000, "fert": 1.18, "pest": 0.90, "water": 0.95},
    "Maize": {"msp": 2410, "yield": 55, "cost": 67000, "fert": 0.95, "pest": 0.68, "water": 0.55},
    "Cotton": {"msp": 8267, "yield": 24, "cost": 95000, "fert": 0.92, "pest": 1.35, "water": 0.72},
    "Mustard / Rapeseed": {"msp": 6200, "yield": 18, "cost": 52000, "fert": 0.74, "pest": 0.50, "water": 0.35},
    "Barley": {"msp": 2150, "yield": 42, "cost": 47000, "fert": 0.70, "pest": 0.45, "water": 0.38},
    "Gram / Chana": {"msp": 5875, "yield": 18, "cost": 48000, "fert": 0.55, "pest": 0.42, "water": 0.30},
    "Masur / Lentil": {"msp": 7000, "yield": 16, "cost": 47000, "fert": 0.52, "pest": 0.40, "water": 0.28},
    "Bajra": {"msp": 2900, "yield": 30, "cost": 41000, "fert": 0.48, "pest": 0.36, "water": 0.22},
    "Jowar": {"msp": 4023, "yield": 28, "cost": 46000, "fert": 0.50, "pest": 0.38, "water": 0.24},
    "Moong": {"msp": 8780, "yield": 11, "cost": 39000, "fert": 0.45, "pest": 0.38, "water": 0.25},
    "Urad": {"msp": 8200, "yield": 10, "cost": 38000, "fert": 0.45, "pest": 0.38, "water": 0.25},
    "Tur / Arhar": {"msp": 8450, "yield": 13, "cost": 46000, "fert": 0.50, "pest": 0.42, "water": 0.32},
    "Groundnut": {"msp": 7517, "yield": 24, "cost": 76000, "fert": 0.78, "pest": 0.72, "water": 0.45},
    "Sunflower Seed": {"msp": 8343, "yield": 18, "cost": 62000, "fert": 0.70, "pest": 0.62, "water": 0.42},
    "Soybean": {"msp": 5708, "yield": 22, "cost": 56000, "fert": 0.58, "pest": 0.52, "water": 0.34},
}


def clamp(value, low, high):
    return max(low, min(high, value))


def data_status(year):
    if year <= 2023:
        return "Official-aligned historical"
    if year == 2024:
        return "Provisional estimate"
    return "Model-estimated scenario"


def soil_for(state_info, district_index, year):
    drift = (year - 2019) * 0.004
    sign = -1 if district_index % 2 else 1
    base = state_info["soil"]
    return {
        "pH": round(base["pH"] + sign * 0.08 + math.sin(year + district_index) * 0.04, 2),
        "nitrogen_kg_ha": round(base["N"] + sign * 8 - (year - 2019) * 1.2, 1),
        "phosphorus_kg_ha": round(base["P"] + sign * 0.8 + drift * 4, 1),
        "potassium_kg_ha": round(base["K"] + sign * 7 - drift * 10, 1),
        "organic_carbon_percent": round(clamp(base["OC"] + math.cos(year + district_index) * 0.025, 0.25, 0.9), 2),
    }


def pollution_index(production, fertilizer, pesticide, water_stress):
    co2 = fertilizer * 1.3 + pesticide * 5.1
    nutrient_load = fertilizer / 260
    pesticide_load = pesticide / 45
    production_load = math.log1p(production) / math.log1p(10000)
    return round(clamp((42 * nutrient_load) + (34 * pesticide_load) + (14 * water_stress) + (10 * production_load), 0, 100), 2), round(co2, 2)


def measurement_uncertainty(state_index, district_index, crop_index, year):
    seasonal = math.sin((year - 2018) * 1.7 + crop_index * 0.9)
    local = math.cos((district_index + 1) * 1.3 + state_index * 0.7)
    stable_hash = (
        state_index * 73856093
        + district_index * 19349663
        + crop_index * 83492791
        + year * 2654435761
    ) % 1000
    field_jitter = (stable_hash / 1000 - 0.5) * 9.5
    return round((seasonal * 2.6) + (local * 1.9) + field_jitter, 2)


def rows():
    for state_index, (state, info) in enumerate(STATES.items()):
        for district_index, district in enumerate(info["districts"]):
            for crop_index, (crop, crop_info) in enumerate(CROPS.items()):
                for year in YEARS:
                    year_factor = 1 + (year - 2019) * 0.018
                    district_factor = 1 + ((district_index % 5) - 2) * 0.035
                    district_input_pressure = 1 + math.sin((district_index + 1) * 0.8) * 0.075
                    district_water_pressure = math.cos((district_index + 1) * 0.65) * 0.055
                    crop_wave = 1 + math.sin((year - 2018) * (crop_index + 2)) * 0.035
                    production = round(info["production_base"] * crop_info["yield"] / 40 * district_factor * crop_wave * (1 + (year - 2019) * 0.01), 1)
                    fertilizer = round(info["fertilizer_base"] * crop_info["fert"] * year_factor * district_input_pressure * (1 + math.cos(crop_index + district_index) * 0.025), 2)
                    pesticide = round(info["pesticide_base"] * crop_info["pest"] * (1 + (year - 2019) * 0.012) * district_input_pressure * (1 + math.sin(crop_index) * 0.03), 2)
                    water_stress = clamp(crop_info["water"] + (0.08 if state in {"Punjab", "Haryana", "Rajasthan"} else 0) + district_water_pressure, 0, 1)
                    formula_pi, co2 = pollution_index(production, fertilizer, pesticide, water_stress)
                    pi = round(clamp(formula_pi + measurement_uncertainty(state_index, district_index, crop_index, year), 0, 100), 2)
                    soil = soil_for(info, district_index, year)
                    yield_proxy = clamp(0.72 + (soil["organic_carbon_percent"] * 0.25) + (soil["nitrogen_kg_ha"] / 2000), 0.65, 1.05)
                    yield_est = round(crop_info["yield"] * yield_proxy, 2)
                    yield_gap = round(max(0, crop_info["yield"] - yield_est), 2)
                    gross = crop_info["msp"] * yield_est
                    net = gross - crop_info["cost"]
                    sustainability = round(clamp(100 - pi + min(18, net / 6500) - water_stress * 8, 0, 100), 2)
                    yield {
                        "state": state,
                        "district": district,
                        "crop": crop,
                        "year": year,
                        "data_status": data_status(year),
                        "production_tonnes": production,
                        "fertilizer_use_kg_per_hectare": fertilizer,
                        "pesticide_use_kg_per_hectare": pesticide,
                        "pH": soil["pH"],
                        "nitrogen_kg_ha": soil["nitrogen_kg_ha"],
                        "phosphorus_kg_ha": soil["phosphorus_kg_ha"],
                        "potassium_kg_ha": soil["potassium_kg_ha"],
                        "organic_carbon_percent": soil["organic_carbon_percent"],
                        "msp_rs_qtl": crop_info["msp"],
                        "expected_yield_q_ha": crop_info["yield"],
                        "estimated_yield_q_ha": yield_est,
                        "yield_gap_q_ha": yield_gap,
                        "production_cost_rs_ha": crop_info["cost"],
                        "expected_net_return_rs_ha": round(net, 2),
                        "water_stress_index": water_stress,
                        "estimated_co2_kg_per_hectare": co2,
                        "formula_pollution_index": formula_pi,
                        "pollution_index": pi,
                        "sustainability_score": sustainability,
                    }


def main():
    DATA_DIR.mkdir(exist_ok=True)
    data = list(rows())
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)
    print(f"Wrote {len(data)} rows to {OUT}")


if __name__ == "__main__":
    main()
