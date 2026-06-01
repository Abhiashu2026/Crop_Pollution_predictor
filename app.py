from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "state_district_crop_environment_dataset.csv"

st.set_page_config(
    page_title="Major Project | Crop Pollution & Punjab Crop Advisor",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.3rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }
    div[data-testid="stMetric"] {
        background: #f6f8f5;
        border: 1px solid #d9e2dc;
        border-radius: 8px;
        padding: 0.85rem;
    }
    div[data-testid="stHorizontalBlock"] {
        gap: 1rem;
    }
    @media (max-width: 760px) {
        .block-container {
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }
        h1 {
            font-size: 1.65rem !important;
            line-height: 1.15 !important;
        }
        h2, h3 {
            line-height: 1.2 !important;
        }
        div[data-testid="stMetric"] {
            padding: 0.65rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            overflow-x: auto;
            flex-wrap: nowrap;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

TEXT = {
    "English": {
        "title": "Crop Pollution & Sustainable Crop Recommendation System",
        "subtitle": "Major Project | NIT Jalandhar | Punjab-first, India-scalable farmer advisory prototype",
        "language": "Language",
        "state": "State",
        "district": "District",
        "crop": "Crop",
        "year": "Year",
        "production": "Production (tonnes)",
        "fertilizer": "Fertilizer use (kg/hectare)",
        "pesticide": "Pesticide use (kg/hectare)",
        "predict": "Predict pollution and sustainability",
        "pollution": "Pollution Index",
        "co2": "Estimated CO2 kg/ha",
        "sustainability": "Sustainability Score",
        "recommendation": "Recommended crop",
        "data_status": "Data status",
        "major_footer": "Major Project | NIT Jalandhar | Decision-support prototype for sustainable agriculture",
    },
    "Hindi": {
        "title": "फसल प्रदूषण और सतत फसल अनुशंसा प्रणाली",
        "subtitle": "मेजर प्रोजेक्ट | NIT जालंधर | पंजाब-प्रथम, भारत-स्तर किसान सलाह प्रणाली",
        "language": "भाषा",
        "state": "राज्य",
        "district": "जिला",
        "crop": "फसल",
        "year": "वर्ष",
        "production": "उत्पादन (टन)",
        "fertilizer": "उर्वरक उपयोग (किग्रा/हेक्टेयर)",
        "pesticide": "कीटनाशक उपयोग (किग्रा/हेक्टेयर)",
        "predict": "प्रदूषण और स्थिरता का अनुमान लगाएं",
        "pollution": "प्रदूषण सूचकांक",
        "co2": "अनुमानित CO2 किग्रा/हेक्टेयर",
        "sustainability": "सततता स्कोर",
        "recommendation": "अनुशंसित फसल",
        "data_status": "डेटा स्थिति",
        "major_footer": "मेजर प्रोजेक्ट | NIT जालंधर | सतत कृषि के लिए निर्णय-सहायता प्रोटोटाइप",
    },
}


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error(
            "Dataset not found. Run scripts/build_major_dataset.py to generate "
            "data/state_district_crop_environment_dataset.csv."
        )
        st.stop()
    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_resource
def train_models(df):
    features = [
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
    target = "pollution_index"
    X = df[features]
    y = df[target]

    categorical = ["state", "district", "crop"]
    numeric = [col for col in features if col not in categorical]
    def make_preprocess():
        return ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
                ("num", StandardScaler(), numeric),
            ]
        )

    candidates = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=90, random_state=42, min_samples_leaf=2, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.22, random_state=42
    )
    trained = {}
    metrics = []
    for name, estimator in candidates.items():
        model = Pipeline([("preprocess", make_preprocess()), ("model", estimator)])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        metrics.append(
            {
                "Model": name,
                "R2": round(r2_score(y_test, pred), 4),
                "MAE": round(mean_absolute_error(y_test, pred), 3),
                "RMSE": round(mse ** 0.5, 3),
            }
        )
        trained[name] = model

    metrics_df = pd.DataFrame(metrics).sort_values("R2", ascending=False)
    best_name = metrics_df.iloc[0]["Model"]
    return trained[best_name], metrics_df, features


def risk_label(pi):
    if pi < 35:
        return "Low", "green"
    if pi < 65:
        return "Medium", "orange"
    return "High", "red"


def farmer_advice(row, language):
    if language == "Hindi":
        advice = []
        if row["pollution_index"] >= 65:
            advice.append("उर्वरक और कीटनाशक उपयोग घटाने के लिए मिट्टी परीक्षण आधारित डोज अपनाएं।")
        if row["water_stress_index"] > 0.7:
            advice.append("पानी की मांग अधिक है; ड्रिप/लेजर लेवलिंग या वैकल्पिक फसल पर विचार करें।")
        if row["sustainability_score"] > 70:
            advice.append("यह फसल वर्तमान इनपुट और आर्थिक मानकों पर बेहतर विकल्प दिखती है।")
        return advice or ["फसल विकल्प मध्यम स्तर का है; स्थानीय कृषि विशेषज्ञ से पुष्टि करें।"]
    advice = []
    if row["pollution_index"] >= 65:
        advice.append("Reduce fertilizer and pesticide dose using soil-test based recommendation.")
    if row["water_stress_index"] > 0.7:
        advice.append("Water demand is high; consider drip irrigation, laser leveling, or an alternative crop.")
    if row["sustainability_score"] > 70:
        advice.append("This crop is comparatively strong under the current sustainability and economics criteria.")
    return advice or ["This is a moderate option; validate with local agronomy support before field adoption."]


def plot_bar(df, x, y, title, color="#16734C"):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(df[x], df[y], color=color)
    ax.set_title(title)
    ax.set_ylabel(y)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    return fig


df = load_data()
model, metrics_df, model_features = train_models(df)

language = st.sidebar.selectbox("Language / भाषा", ["English", "Hindi"])
T = TEXT[language]

st.title(T["title"])
st.caption(T["subtitle"])

st.sidebar.markdown("### Filters")
states = sorted(df["state"].unique())
default_state_index = states.index("Punjab") if "Punjab" in states else 0
state = st.sidebar.selectbox(T["state"], states, index=default_state_index)
districts = sorted(df.loc[df["state"] == state, "district"].unique())
district = st.sidebar.selectbox(T["district"], districts)
years = sorted(df["year"].unique())
year = st.sidebar.selectbox(T["year"], years, index=years.index(2026) if 2026 in years else len(years) - 1)

state_district_df = df[(df["state"] == state) & (df["district"] == district) & (df["year"] == year)]

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Overview",
        "Pollution Predictor",
        "Crop Recommendation",
        "State/District Dashboard",
        "Model Performance",
        "Data Reliability",
    ]
)

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("States", df["state"].nunique())
    c2.metric("District profiles", df[["state", "district"]].drop_duplicates().shape[0])
    c3.metric("Crops", df["crop"].nunique())
    c4.metric("Rows", f"{len(df):,}")

    st.markdown(
        """
        This upgraded major-project dashboard merges crop pollution prediction with
        Punjab-first crop recommendation. Punjab is the default state, while the
        dataset also covers multiple Indian states and district profiles to make
        the project scalable beyond one region.
        """
    )
    top = (
        state_district_df.sort_values("sustainability_score", ascending=False)
        .head(6)[
            [
                "crop",
                "pollution_index",
                "sustainability_score",
                "estimated_co2_kg_per_hectare",
                "expected_net_return_rs_ha",
                "data_status",
            ]
        ]
    )
    st.subheader(f"Best crop options for {district}, {state} ({year})")
    st.dataframe(top, use_container_width=True)

with tab2:
    st.subheader("Pollution prediction from production inputs")
    left, right = st.columns([0.42, 0.58])
    with left:
        crop = st.selectbox(T["crop"], sorted(state_district_df["crop"].unique()))
        sample = state_district_df[state_district_df["crop"] == crop].iloc[0]
        production = st.number_input(T["production"], min_value=100.0, max_value=25000.0, value=float(sample["production_tonnes"]), step=100.0)
        fertilizer = st.number_input(T["fertilizer"], min_value=10.0, max_value=450.0, value=float(sample["fertilizer_use_kg_per_hectare"]), step=5.0)
        pesticide = st.number_input(T["pesticide"], min_value=0.0, max_value=80.0, value=float(sample["pesticide_use_kg_per_hectare"]), step=1.0)
        predict = st.button(T["predict"], type="primary")

    with right:
        input_row = sample.copy()
        input_row["production_tonnes"] = production
        input_row["fertilizer_use_kg_per_hectare"] = fertilizer
        input_row["pesticide_use_kg_per_hectare"] = pesticide
        input_df = pd.DataFrame([input_row[model_features]])
        predicted_pi = float(model.predict(input_df)[0])
        co2 = fertilizer * 1.3 + pesticide * 5.1
        risk, risk_color = risk_label(predicted_pi)
        if predict:
            st.metric(T["pollution"], f"{predicted_pi:.2f}", f"{risk} risk")
            st.metric(T["co2"], f"{co2:.2f}")
            st.markdown(f"<b style='color:{risk_color}'>Risk category: {risk}</b>", unsafe_allow_html=True)
            result = input_row.to_frame().T[
                [
                    "state",
                    "district",
                    "crop",
                    "year",
                    "production_tonnes",
                    "fertilizer_use_kg_per_hectare",
                    "pesticide_use_kg_per_hectare",
                    "data_status",
                ]
            ]
            result["predicted_pollution_index"] = round(predicted_pi, 2)
            result["estimated_co2_kg_per_hectare"] = round(co2, 2)
            st.dataframe(result, use_container_width=True)

with tab3:
    st.subheader("Sustainable crop recommendation")
    ranked = state_district_df.sort_values(
        ["sustainability_score", "expected_net_return_rs_ha"], ascending=False
    ).copy()
    best = ranked.iloc[0]
    a, b, c, d = st.columns(4)
    a.metric(T["recommendation"], best["crop"])
    b.metric(T["sustainability"], f"{best['sustainability_score']:.2f}")
    c.metric(T["pollution"], f"{best['pollution_index']:.2f}")
    d.metric("Net return Rs/ha", f"{best['expected_net_return_rs_ha']:,.0f}")

    show_cols = [
        "crop",
        "msp_rs_qtl",
        "estimated_yield_q_ha",
        "pollution_index",
        "estimated_co2_kg_per_hectare",
        "expected_net_return_rs_ha",
        "sustainability_score",
        "data_status",
    ]
    st.dataframe(ranked[show_cols].head(12), use_container_width=True)
    st.markdown("#### Farmer advisory")
    for item in farmer_advice(best, language):
        st.write(f"- {item}")

    st.info(
        "Sustainability Score = low pollution + expected return + water stress penalty. "
        "It is an advisory score, not an official agronomic prescription."
    )

with tab4:
    st.subheader("State and district dashboard")
    summary = (
        df[df["year"] == year]
        .groupby("state", as_index=False)
        .agg(
            mean_pollution=("pollution_index", "mean"),
            mean_sustainability=("sustainability_score", "mean"),
            mean_co2=("estimated_co2_kg_per_hectare", "mean"),
        )
        .sort_values("mean_sustainability", ascending=False)
    )
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_bar(summary.head(10), "state", "mean_sustainability", f"Top states by sustainability ({year})"))
    with col2:
        st.pyplot(plot_bar(summary.sort_values("mean_pollution", ascending=False).head(10), "state", "mean_pollution", f"Highest pollution index states ({year})", "#A44536"))

    district_summary = (
        df[(df["state"] == state) & (df["year"] == year)]
        .groupby("district", as_index=False)
        .agg(mean_pollution=("pollution_index", "mean"), mean_sustainability=("sustainability_score", "mean"))
    )
    st.dataframe(district_summary.sort_values("mean_sustainability", ascending=False), use_container_width=True)

with tab5:
    st.subheader("Model comparison and accuracy")
    st.dataframe(metrics_df, use_container_width=True)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(metrics_df["Model"], metrics_df["R2"], color=["#16734C", "#2C6E91", "#B87914"][: len(metrics_df)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("R2 score")
    ax.set_title("Pollution model R2 comparison")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.25)
    st.pyplot(fig)
    st.warning(
        "Accuracy is evaluated on the project dataset, which combines official-aligned, derived, and projected scenario rows. "
        "For publication-grade validation, replace derived labels with observed farm-level measurements."
    )

with tab6:
    st.subheader("Data reliability and source traceability")
    status_counts = df["data_status"].value_counts().rename_axis("Data status").reset_index(name="Rows")
    st.dataframe(status_counts, use_container_width=True)
    st.markdown(
        """
        **Source-backed reliability strategy**

        - Government crop-production and fertilizer datasets are used as the primary reliability base.
        - District and crop expansion is marked as derived when direct official district crop-pollution records are unavailable.
        - 2025 and 2026 rows are marked as projected scenarios unless final government data exists.
        - The app is a decision-support prototype for farmers, researchers, and students.

        See `DATA_SOURCES.md` in the repository for full source links and limitations.
        """
    )
    st.markdown(
        """
        **Future farmer-facing scope**

        - Mobile-first Hindi/Punjabi advisory interface.
        - Soil Health Card integration.
        - Live mandi price and MSP comparison.
        - District-wise crop rotation and low-pollution recommendations.
        - Direct extension support for farmers through local agriculture departments.
        """
    )

st.markdown("---")
st.caption(T["major_footer"])
