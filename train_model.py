from pathlib import Path
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "state_district_crop_environment_dataset.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "pollution_model.pkl"

FEATURES = [
    "state",
    "district",
    "crop",
    "year",
    "production_tonnes",
    "fertilizer_use_kg_per_hectare",
    "pesticide_use_kg_per_hectare",
    "water_stress_index",
    "pH",
    "nitrogen_kg_ha",
    "phosphorus_kg_ha",
    "potassium_kg_ha",
    "organic_carbon_percent",
]
TARGET = "pollution_index"


def make_model(estimator):
    categorical = ["state", "district", "crop"]
    numeric = [col for col in FEATURES if col not in categorical]
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run scripts/build_major_dataset.py first."
        )

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.22, random_state=42
    )

    candidates = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=180, random_state=42, min_samples_leaf=2
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    results = []
    trained = {}
    for name, estimator in candidates.items():
        model = make_model(estimator)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        results.append(
            {
                "model": name,
                "r2": round(r2_score(y_test, pred), 4),
                "mae": round(mean_absolute_error(y_test, pred), 3),
                "rmse": round(mse ** 0.5, 3),
            }
        )
        trained[name] = model

    best = max(results, key=lambda item: item["r2"])
    MODEL_DIR.mkdir(exist_ok=True)
    with MODEL_PATH.open("wb") as f:
        pickle.dump(trained[best["model"]], f)

    print("Model benchmark:")
    for row in results:
        print(row)
    print(f"Saved best model ({best['model']}) to {MODEL_PATH}")


if __name__ == "__main__":
    main()
