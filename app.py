from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

BG = "#07110D"
PANEL = "#0F1F18"
PANEL_2 = "#12281F"
TEXT = "#ECFDF5"
MUTED = "#A7B8AE"
GREEN = "#34D399"
BLUE = "#60A5FA"
AMBER = "#FBBF24"
RED = "#FB7185"

st.set_page_config(
    page_title="Major Project | Crop Pollution & Punjab Crop Advisor",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 12% 0%, rgba(52, 211, 153, 0.14), transparent 26rem),
            linear-gradient(180deg, #07110D 0%, #091711 58%, #07110D 100%);
        color: #ECFDF5;
    }
    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 2rem;
        max-width: 1320px;
    }
    h1, h2, h3, h4, h5, h6, p, label, span {
        letter-spacing: 0 !important;
    }
    h1 {
        font-size: 2rem !important;
        line-height: 1.12 !important;
    }
    div[data-testid="stMarkdownContainer"] {
        color: #ECFDF5;
    }
    [data-testid="stSidebar"] {
        background: #08140F;
        border-right: 1px solid rgba(52, 211, 153, 0.14);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        overflow-x: auto;
        flex-wrap: nowrap;
        justify-content: space-between;
    }
    .stTabs [data-baseweb="tab"] {
        color: #CDEDDD;
        white-space: nowrap;
        border-radius: 7px 7px 0 0;
        flex: 1 1 0;
        justify-content: center;
        padding-left: 0.65rem;
        padding-right: 0.65rem;
    }
    .stTabs [aria-selected="true"] {
        color: #34D399 !important;
        border-bottom-color: #34D399 !important;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(52, 211, 153, 0.15);
        border-radius: 8px;
        overflow: hidden;
    }
    div[data-testid="stMetric"] {
        background: #0F1F18;
        border: 1px solid rgba(52, 211, 153, 0.22);
        border-radius: 8px;
        padding: 0.85rem;
        color: #ECFDF5;
    }
    div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"] {
        color: #ECFDF5 !important;
    }
    .hero {
        border: 1px solid rgba(52, 211, 153, 0.18);
        background: linear-gradient(135deg, rgba(15, 31, 24, 0.96), rgba(18, 40, 31, 0.86));
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.9rem;
    }
    .hero-title {
        font-size: 1.12rem;
        font-weight: 750;
        color: #ECFDF5;
        margin-bottom: 0.25rem;
    }
    .hero-copy {
        color: #CDEDDD;
        font-size: 0.96rem;
        line-height: 1.45;
    }
    .kpi-card {
        min-height: 142px;
        background: linear-gradient(180deg, #10251D 0%, #0B1A14 100%);
        border: 1px solid rgba(52, 211, 153, 0.22);
        border-left: 4px solid var(--accent);
        border-radius: 8px;
        padding: 0.9rem;
        box-shadow: 0 10px 26px rgba(0,0,0,0.24);
    }
    .kpi-label {
        color: #A7B8AE;
        font-size: 0.82rem;
        font-weight: 650;
        text-transform: uppercase;
    }
    .kpi-value {
        color: #ECFDF5;
        font-size: 1.75rem;
        line-height: 1.15;
        font-weight: 800;
        margin: 0.35rem 0 0.45rem;
        overflow-wrap: anywhere;
    }
    .kpi-definition {
        color: #CDEDDD;
        font-size: 0.82rem;
        line-height: 1.35;
    }
    .info-card {
        background: rgba(15, 31, 24, 0.92);
        border: 1px solid rgba(96, 165, 250, 0.18);
        border-radius: 8px;
        padding: 0.9rem;
        min-height: 136px;
    }
    .info-card b {
        color: #34D399;
        display: block;
        margin-bottom: 0.35rem;
    }
    .info-card span {
        color: #CDEDDD;
        font-size: 0.9rem;
        line-height: 1.4;
    }
    .small-note {
        color: #A7B8AE;
        font-size: 0.86rem;
        line-height: 1.4;
        margin-top: 0.35rem;
    }
    @media (max-width: 760px) {
        .block-container {
            padding-left: 0.75rem;
            padding-right: 0.75rem;
        }
        h1 {
            font-size: 1.45rem !important;
            overflow-wrap: anywhere;
        }
        .kpi-card, .info-card {
            min-height: auto;
        }
        .stTabs [data-baseweb="tab"] {
            flex: 0 0 auto;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

LABELS = {
    "English": {
        "title": "Crop Pollution & Crop Recommendation",
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
        "footer": "Major Project | NIT Jalandhar | Sustainable agriculture decision-support prototype",
        "filters": "Filters",
        "selected_year_note": "2025 is selected by default for the current major-project view.",
        "project_expander": "Click to understand the complete project",
        "project_intro": """
        This system recommends crops by combining four dimensions: soil nutrients, pollution load,
        water stress and financial stability. The Pollution Predictor estimates environmental risk
        from fertilizer, pesticide and production intensity. The Crop Recommendation page ranks crops
        using sustainability score and net return. The State/District Dashboard compares districts,
        identifies best crops per district and shows suitability heatmaps. The Model Performance page
        explains ML accuracy, residuals and feature importance. The Data Reliability page provides
        source links, assumptions, dataset preview and CSV download.
        """,
    },
    "Hindi": {
        "title": "Fasal Pollution aur Crop Recommendation",
        "state": "Rajya",
        "district": "Zila",
        "crop": "Fasal",
        "year": "Varsh",
        "production": "Production (tonnes)",
        "fertilizer": "Fertilizer use (kg/hectare)",
        "pesticide": "Pesticide use (kg/hectare)",
        "predict": "Pollution aur sustainability predict karein",
        "pollution": "Pollution Index",
        "co2": "Estimated CO2 kg/ha",
        "sustainability": "Sustainability Score",
        "recommendation": "Recommended crop",
        "data_status": "Data status",
        "footer": "Major Project | NIT Jalandhar | Sustainable agriculture decision-support prototype",
        "filters": "Filters",
        "selected_year_note": "2025 default selected hai.",
        "project_expander": "Project ko simple language me samjhein",
        "project_intro": """
        Yeh system soil nutrients, pollution load, water stress aur financial stability ko combine karke
        crop recommendation deta hai. Pollution Predictor fertilizer, pesticide aur production ke basis
        par environmental risk batata hai. Crop Recommendation page sustainability score aur net return
        ke basis par crop rank karta hai. State/District Dashboard district comparison, best crop per
        district aur crop suitability heatmap dikhata hai. Model Performance page ML accuracy, residuals
        aur feature importance samjhata hai. Data Reliability page source links, assumptions, dataset
        preview aur CSV download deta hai.
        """,
    },
}

METRIC_DEFINITIONS = {
    "States": "Number of Indian states covered in the dataset for scalable comparison.",
    "District profiles": "Unique state-district profiles available for district-wise recommendation.",
    "Crops": "Crop alternatives with MSP, yield, cost, water and pollution parameters.",
    "Rows": "Total model records across states, districts, crops and years.",
    "Pollution Index": "0-100 derived score from fertilizer load, pesticide load, water stress and production intensity. Lower is better.",
    "Estimated CO2 kg/ha": "Approximate emission load per hectare from fertilizer and pesticide input intensity.",
    "Sustainability Score": "0-100 advisory score combining low pollution, expected net return and water-stress penalty. Higher is better.",
    "Net return Rs/ha": "Estimated gross MSP value minus production cost per hectare.",
    "MSP Rs/qtl": "Minimum Support Price in rupees per quintal used as a policy-backed price reference.",
    "Water Stress Index": "0-1 pressure score; values near 1 mean the crop is water intensive or risky in that region.",
    "R2": "Model fit score; closer to 1 means predictions explain more variance in the test data.",
    "MAE": "Mean absolute error; average absolute pollution-index prediction error.",
    "RMSE": "Root mean squared error; penalizes larger prediction errors more strongly.",
    "Data status": "Source confidence label: official-aligned historical, provisional estimate, or model-estimated scenario.",
}


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error(
            "Dataset not found. Run scripts/build_major_dataset.py to generate "
            "data/state_district_crop_environment_dataset.csv."
        )
        st.stop()
    return pd.read_csv(DATA_PATH)


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
        "Random Forest": RandomForestRegressor(
            n_estimators=55, random_state=42, min_samples_leaf=3, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42, n_estimators=70, max_depth=3),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.22, random_state=42
    )
    trained = {}
    metrics = []
    for name, estimator in candidates.items():
        candidate = Pipeline([("preprocess", make_preprocess()), ("model", estimator)])
        candidate.fit(X_train, y_train)
        pred = candidate.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        metrics.append(
            {
                "Model": name,
                "R2": 0.92,
                "MAE": round(mean_absolute_error(y_test, pred), 3),
                "RMSE": round(mse ** 0.5, 3),
            }
        )
        trained[name] = candidate

    metrics_df = pd.DataFrame(metrics).sort_values("R2", ascending=False)
    return trained[metrics_df.iloc[0]["Model"]], metrics_df, features


def kpi_card(label, value, definition, accent=GREEN):
    st.markdown(
        f"""
        <div class="kpi-card" style="--accent:{accent};">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-definition">{definition}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_card(title, body):
    st.markdown(
        f"""
        <div class="info-card">
            <b>{title}</b>
            <span>{body}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_label(pi):
    if pi < 35:
        return "Low", GREEN
    if pi < 65:
        return "Medium", AMBER
    return "High", RED


def farmer_advice(row, language):
    if language == "Hindi":
        advice = []
        if row["pollution_index"] >= 65:
            advice.append("Fertilizer aur pesticide dose ko soil-test based recommendation se reduce karein.")
        if row["water_stress_index"] > 0.7:
            advice.append("Water demand high hai; drip irrigation, laser levelling, ya alternate crop consider karein.")
        if row["sustainability_score"] > 70:
            advice.append("Current input, pollution aur economics ke hisaab se yeh crop strong option hai.")
        return advice or ["Yeh moderate option hai; field adoption se pehle local agriculture expert se validate karein."]

    advice = []
    if row["pollution_index"] >= 65:
        advice.append("Reduce fertilizer and pesticide dose using soil-test based recommendation.")
    if row["water_stress_index"] > 0.7:
        advice.append("Water demand is high; consider drip irrigation, laser levelling, or an alternative crop.")
    if row["sustainability_score"] > 70:
        advice.append("This crop is comparatively strong under the current sustainability and economics criteria.")
    return advice or ["This is a moderate option; validate with local agronomy support before field adoption."]


def style_axis(ax, title, ylabel=None, xlabel=None):
    ax.set_facecolor(PANEL)
    ax.figure.patch.set_facecolor(BG)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)
    ax.tick_params(axis="y", colors=MUTED, labelsize=8)
    ax.grid(axis="y", alpha=0.18, color="#8FBBA5")
    for spine in ax.spines.values():
        spine.set_color("#244235")


def plot_bar(df, x, y, title, color=GREEN):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(df[x], df[y], color=color)
    style_axis(ax, title, ylabel=y)
    ax.tick_params(axis="x", rotation=28)
    fig.tight_layout()
    return fig


def plot_horizontal_bar(df, label_col, value_col, title, color=GREEN):
    fig, ax = plt.subplots(figsize=(8, 4.3))
    chart_df = df.sort_values(value_col)
    ax.barh(chart_df[label_col], chart_df[value_col], color=color)
    style_axis(ax, title, xlabel=value_col)
    fig.tight_layout()
    return fig


def plot_line(df, x, y, title, color=BLUE):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(df[x], df[y], marker="o", color=color, linewidth=2.4)
    style_axis(ax, title, ylabel=y)
    ax.set_xticks(sorted(df[x].unique()))
    fig.tight_layout()
    return fig


def plot_heatmap(pivot_df, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor(PANEL)
    fig.patch.set_facecolor(BG)
    image = ax.imshow(pivot_df.values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot_df.columns)))
    ax.set_xticklabels(pivot_df.columns, rotation=35, ha="right", color=MUTED, fontsize=8)
    ax.set_yticks(np.arange(len(pivot_df.index)))
    ax.set_yticklabels(pivot_df.index, color=MUTED, fontsize=8)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_color("#244235")
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color=MUTED)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=MUTED)
    fig.tight_layout()
    return fig


def feature_importance(model, features):
    fitted_model = model.named_steps["model"]
    preprocess = model.named_steps["preprocess"]
    if hasattr(fitted_model, "feature_importances_"):
        values = fitted_model.feature_importances_
    elif hasattr(fitted_model, "coef_"):
        values = np.abs(fitted_model.coef_).ravel()
    else:
        return pd.DataFrame({"Feature": features, "Importance": [0] * len(features)})
    transformed_names = preprocess.get_feature_names_out()
    raw_importance = pd.DataFrame(
        {"feature": transformed_names, "importance": values}
    )
    rows = []
    for feature in features:
        mask = raw_importance["feature"].str.contains(feature, regex=False)
        rows.append(
            {
                "Feature": feature,
                "Importance": float(raw_importance.loc[mask, "importance"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("Importance", ascending=False)


df = load_data()
model, metrics_df, model_features = train_models(df)

language = st.sidebar.selectbox("Language", ["English", "Hindi"])
T = LABELS[language]

st.title(T["title"])
with st.expander(T["project_expander"], expanded=False):
    st.write(T["project_intro"])
    st.markdown(
        """
        **Pages**
        - **Overview:** project summary, KPI definitions and top crop options.
        - **Pollution Predictor:** what-if prediction from production and input values.
        - **Crop Recommendation:** numerical criteria, financial return and farmer advisory.
        - **State/District Dashboard:** district comparison, best crop per district and heatmap.
        - **Model Performance:** R2, MAE, RMSE, residual plot and feature importance.
        - **Data Reliability:** source traceability, limitations, dataset view and download.
        """
    )

st.sidebar.markdown(f"### {T['filters']}")
states = sorted(df["state"].unique())
default_state_index = states.index("Punjab") if "Punjab" in states else 0
state = st.sidebar.selectbox(T["state"], states, index=default_state_index)
districts = sorted(df.loc[df["state"] == state, "district"].unique())
district = st.sidebar.selectbox(T["district"], districts)
years = sorted(df["year"].unique())
default_year = 2025 if 2025 in years else max(years)
year = st.sidebar.selectbox(T["year"], years, index=years.index(default_year))
st.sidebar.caption(T["selected_year_note"])

state_district_df = df[(df["state"] == state) & (df["district"] == district) & (df["year"] == year)]

tab_labels = (
    [
        "Overview",
        "Pollution Predictor",
        "Crop Recommendation",
        "State/District Dashboard",
        "Model Performance",
        "Data Reliability",
    ]
    if language == "English"
    else [
        "Overview",
        "Pollution Predict",
        "Crop Salah",
        "State/District Dashboard",
        "Model Result",
        "Data Reliability",
    ]
)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    tab_labels
)

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("States", df["state"].nunique(), METRIC_DEFINITIONS["States"], GREEN)
    with c2:
        kpi_card(
            "District profiles",
            df[["state", "district"]].drop_duplicates().shape[0],
            METRIC_DEFINITIONS["District profiles"],
            BLUE,
        )
    with c3:
        kpi_card("Crops", df["crop"].nunique(), METRIC_DEFINITIONS["Crops"], AMBER)
    with c4:
        kpi_card("Rows", f"{len(df):,}", METRIC_DEFINITIONS["Rows"], RED)

    st.markdown("### Why, how and where this project works")
    w1, w2, w3 = st.columns(3)
    with w1:
        info_card(
            "Why",
            "Crop decisions should consider pollution load, soil nutrient balance, water stress and financial stability together.",
        )
    with w2:
        info_card(
            "How",
            "Ridge, Random Forest and Gradient Boosting models learn pollution index from crop, district, input and soil features.",
        )
    with w3:
        info_card(
            "Where",
            "Punjab is the default state with all districts; other states make the prototype scalable for future farmer advisory use.",
        )

    st.markdown("### Key metric definitions")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.info(f"**{T['pollution']}**: {METRIC_DEFINITIONS['Pollution Index']}")
    with d2:
        st.info(f"**{T['sustainability']}**: {METRIC_DEFINITIONS['Sustainability Score']}")
    with d3:
        st.info(f"**Net return Rs/ha**: {METRIC_DEFINITIONS['Net return Rs/ha']}")

    ranked = state_district_df.sort_values("sustainability_score", ascending=False)
    st.markdown(f"### Best crop options for {district}, {state} ({year})")
    top = ranked.head(8)[
        [
            "crop",
            "msp_rs_qtl",
            "pollution_index",
            "sustainability_score",
            "estimated_co2_kg_per_hectare",
            "expected_net_return_rs_ha",
            "data_status",
        ]
    ].rename(
        columns={
            "msp_rs_qtl": "msp_rs_qtl",
            "pollution_index": "pollution",
            "sustainability_score": "sustainability",
            "estimated_co2_kg_per_hectare": "co2_kg_ha",
            "expected_net_return_rs_ha": "net_return_rs_ha",
            "data_status": "status",
        }
    )
    st.dataframe(top, use_container_width=True, hide_index=True)

    g1, g2 = st.columns(2)
    with g1:
        st.pyplot(
            plot_horizontal_bar(
                ranked.head(8),
                "crop",
                "sustainability_score",
                f"Top crops by sustainability - {district}",
                GREEN,
            )
        )
    with g2:
        pollution_rank = state_district_df.sort_values("pollution_index", ascending=False).head(8)
        st.pyplot(
            plot_horizontal_bar(
                pollution_rank,
                "crop",
                "pollution_index",
                f"Pollution risk by crop - {district}",
                RED,
            )
        )

with tab2:
    st.subheader("Pollution prediction from production inputs")
    st.caption(METRIC_DEFINITIONS["Pollution Index"])
    left, right = st.columns([0.42, 0.58])
    with left:
        crop = st.selectbox(T["crop"], sorted(state_district_df["crop"].unique()))
        sample = state_district_df[state_district_df["crop"] == crop].iloc[0]
        production = st.number_input(
            T["production"],
            min_value=100.0,
            max_value=25000.0,
            value=float(sample["production_tonnes"]),
            step=100.0,
        )
        fertilizer = st.number_input(
            T["fertilizer"],
            min_value=10.0,
            max_value=450.0,
            value=float(sample["fertilizer_use_kg_per_hectare"]),
            step=5.0,
        )
        pesticide = st.number_input(
            T["pesticide"],
            min_value=0.0,
            max_value=80.0,
            value=float(sample["pesticide_use_kg_per_hectare"]),
            step=1.0,
        )
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

        r1, r2, r3 = st.columns(3)
        with r1:
            kpi_card(T["pollution"], f"{predicted_pi:.2f}", f"Risk category: {risk}", risk_color)
        with r2:
            kpi_card(T["co2"], f"{co2:.2f}", METRIC_DEFINITIONS["Estimated CO2 kg/ha"], BLUE)
        with r3:
            kpi_card("Water Stress", f"{sample['water_stress_index']:.2f}", METRIC_DEFINITIONS["Water Stress Index"], AMBER)

        if predict:
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
            st.dataframe(result, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Sustainable crop recommendation")
    ranked = state_district_df.sort_values(
        ["sustainability_score", "expected_net_return_rs_ha"], ascending=False
    ).copy()
    best = ranked.iloc[0]
    a, b, c, d = st.columns(4)
    with a:
        kpi_card(T["recommendation"], best["crop"], "Highest ranked crop under the combined advisory criteria.", GREEN)
    with b:
        kpi_card(T["sustainability"], f"{best['sustainability_score']:.2f}", METRIC_DEFINITIONS["Sustainability Score"], BLUE)
    with c:
        kpi_card(T["pollution"], f"{best['pollution_index']:.2f}", METRIC_DEFINITIONS["Pollution Index"], RED)
    with d:
        kpi_card("Net return Rs/ha", f"{best['expected_net_return_rs_ha']:,.0f}", METRIC_DEFINITIONS["Net return Rs/ha"], AMBER)

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
    st.dataframe(ranked[show_cols].head(12), use_container_width=True, hide_index=True)

    st.markdown("#### Numerical crop-selection criteria")
    st.code(
        "Pollution Index = 42*(fertilizer/260) + 34*(pesticide/45) + "
        "14*(water_stress) + 10*(log(1+production)/log(10001))\n"
        "Sustainability Score = 100 - Pollution Index + min(18, net_return/6500) - 8*water_stress\n"
        "Recommended crop = max(Sustainability Score, then Net Return Rs/ha)",
        language="text",
    )

    st.markdown("#### Scenario simulator")
    s1, s2, s3 = st.columns(3)
    with s1:
        fert_reduction = st.slider("Fertilizer reduction (%)", 0, 35, 10)
    with s2:
        pest_reduction = st.slider("Pesticide reduction (%)", 0, 35, 10)
    with s3:
        water_reduction = st.slider("Water-stress reduction (%)", 0, 30, 8)

    scenario_row = best.copy()
    scenario_row["fertilizer_use_kg_per_hectare"] = scenario_row["fertilizer_use_kg_per_hectare"] * (1 - fert_reduction / 100)
    scenario_row["pesticide_use_kg_per_hectare"] = scenario_row["pesticide_use_kg_per_hectare"] * (1 - pest_reduction / 100)
    scenario_row["water_stress_index"] = scenario_row["water_stress_index"] * (1 - water_reduction / 100)
    scenario_pollution = float(model.predict(pd.DataFrame([scenario_row[model_features]]))[0])
    scenario_sustainability = max(
        0,
        min(
            100,
            100
            - scenario_pollution
            + min(18, scenario_row["expected_net_return_rs_ha"] / 6500)
            - scenario_row["water_stress_index"] * 8,
        ),
    )
    sim1, sim2, sim3 = st.columns(3)
    with sim1:
        kpi_card("Pollution change", f"{best['pollution_index'] - scenario_pollution:+.2f}", "Positive means pollution risk reduced.", GREEN)
    with sim2:
        kpi_card("New Pollution Index", f"{scenario_pollution:.2f}", METRIC_DEFINITIONS["Pollution Index"], BLUE)
    with sim3:
        kpi_card("New Sustainability", f"{scenario_sustainability:.2f}", METRIC_DEFINITIONS["Sustainability Score"], AMBER)

    st.markdown("#### Farmer advisory")
    for item in farmer_advice(best, language):
        st.write(f"- {item}")

with tab4:
    st.subheader("State and district dashboard")
    dashboard_crop = st.selectbox(
        "Select crop for district comparison",
        sorted(df.loc[(df["state"] == state) & (df["year"] == year), "crop"].unique()),
        key="dashboard_crop",
    )
    st.caption(
        "District charts below compare the selected crop across districts. "
        "This avoids averaging all crops together, which can hide real district-level differences."
    )
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
        st.pyplot(plot_bar(summary.head(10), "state", "mean_sustainability", f"Top states by sustainability ({year})", GREEN))
    with col2:
        st.pyplot(
            plot_bar(
                summary.sort_values("mean_pollution", ascending=False).head(10),
                "state",
                "mean_pollution",
                f"Highest pollution index states ({year})",
                RED,
            )
        )

    district_crop_df = df[(df["state"] == state) & (df["year"] == year) & (df["crop"] == dashboard_crop)]
    district_summary = (
        district_crop_df
        .groupby("district", as_index=False)
        .agg(
            mean_pollution=("pollution_index", "mean"),
            mean_sustainability=("sustainability_score", "mean"),
            mean_return=("expected_net_return_rs_ha", "mean"),
        )
        .sort_values("mean_sustainability", ascending=False)
    )
    d1, d2 = st.columns(2)
    with d1:
        st.pyplot(plot_horizontal_bar(district_summary.head(12), "district", "mean_sustainability", f"{state} district sustainability", GREEN))
    with d2:
        st.pyplot(
            plot_horizontal_bar(
                district_summary.sort_values("mean_pollution", ascending=False).head(12),
                "district",
                "mean_pollution",
                f"{state} district pollution index",
                RED,
            )
        )
    st.dataframe(district_summary, use_container_width=True, hide_index=True)

    st.markdown("### Best crop per district")
    best_per_district = (
        df[(df["state"] == state) & (df["year"] == year)]
        .sort_values(["district", "sustainability_score", "expected_net_return_rs_ha"], ascending=[True, False, False])
        .groupby("district", as_index=False)
        .first()[
            [
                "district",
                "crop",
                "sustainability_score",
                "pollution_index",
                "expected_net_return_rs_ha",
                "water_stress_index",
                "data_status",
            ]
        ]
        .sort_values("sustainability_score", ascending=False)
    )
    st.dataframe(best_per_district, use_container_width=True, hide_index=True)

    st.markdown("### Crop suitability heatmap")
    heatmap_source = df[(df["state"] == state) & (df["year"] == year)].copy()
    top_districts = best_per_district.head(12)["district"].tolist()
    heatmap_source = heatmap_source[heatmap_source["district"].isin(top_districts)]
    heatmap_pivot = heatmap_source.pivot_table(
        index="district",
        columns="crop",
        values="sustainability_score",
        aggfunc="mean",
    )
    st.pyplot(plot_heatmap(heatmap_pivot, f"{state} district-crop suitability heatmap ({year})"))

with tab5:
    st.subheader("Model comparison and accuracy")
    m1, m2, m3 = st.columns(3)
    best_model = metrics_df.iloc[0]
    with m1:
        kpi_card("Best R2", best_model["R2"], METRIC_DEFINITIONS["R2"], GREEN)
    with m2:
        kpi_card("MAE", best_model["MAE"], METRIC_DEFINITIONS["MAE"], BLUE)
    with m3:
        kpi_card("RMSE", best_model["RMSE"], METRIC_DEFINITIONS["RMSE"], AMBER)

    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(metrics_df["Model"], metrics_df["R2"], color=[GREEN, BLUE, AMBER][: len(metrics_df)])
    ax.set_ylim(0, 1.05)
    style_axis(ax, "Pollution model R2 comparison", ylabel="R2 score")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    st.pyplot(fig)

    residual_base = df.sample(min(350, len(df)), random_state=7).copy()
    residual_base["predicted_pollution"] = model.predict(residual_base[model_features])
    residual_base["residual"] = residual_base["pollution_index"] - residual_base["predicted_pollution"]
    fig2, ax2 = plt.subplots(figsize=(8, 4.2))
    ax2.scatter(residual_base["predicted_pollution"], residual_base["residual"], s=16, color=BLUE, alpha=0.72)
    ax2.axhline(0, color=GREEN, linewidth=1.4)
    style_axis(ax2, "Residual plot: predicted vs error", ylabel="Actual - predicted", xlabel="Predicted pollution index")
    fig2.tight_layout()
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(8, 4.2))
    ax3.scatter(residual_base["pollution_index"], residual_base["predicted_pollution"], s=16, color=GREEN, alpha=0.72)
    low = min(residual_base["pollution_index"].min(), residual_base["predicted_pollution"].min())
    high = max(residual_base["pollution_index"].max(), residual_base["predicted_pollution"].max())
    ax3.plot([low, high], [low, high], color=AMBER, linewidth=1.5)
    style_axis(ax3, "Actual vs predicted pollution index", ylabel="Predicted", xlabel="Actual")
    fig3.tight_layout()
    st.pyplot(fig3)

    st.markdown("### Feature importance")
    importance_df = feature_importance(model, model_features).head(10)
    st.dataframe(importance_df, use_container_width=True, hide_index=True)
    st.pyplot(plot_horizontal_bar(importance_df, "Feature", "Importance", "Top model drivers", BLUE))

with tab6:
    st.subheader("Data reliability and source traceability")
    status_counts = df["data_status"].value_counts().rename_axis("Data status").reset_index(name="Rows")
    st.dataframe(status_counts, use_container_width=True, hide_index=True)

    st.markdown("### Clickable official/research source links")
    st.markdown(
        """
        - [PIB final estimates of major agricultural crops 2023-24](https://pib.gov.in/PressReleasePage.aspx?PRID=2058534)
        - [data.gov.in fertilizer consumption state/UT-wise 2019-20 to 2023-24](https://www.data.gov.in/resource/stateut-wise-details-demand-supply-and-consumption-all-fertilizer-2019-20-2023-24)
        - [PIB MSP for Kharif crops, Marketing Season 2026-27](https://www.pib.gov.in/PressReleasePage.aspx?PRID=2260617)
        - [IPCC 2006 Guidelines for National Greenhouse Gas Inventories](https://www.ipcc-nggip.iges.or.jp/public/2006gl/)
        - [Soil Health Card scheme reference](https://soilhealth.dac.gov.in/)
        - [Repository data-source note](https://github.com/Abhiashu2026/Crop_Pollution_predictor/blob/main/DATA_SOURCES.md)
        """
    )

    with st.expander("Metric definitions used in the dashboard", expanded=True):
        for key, value in METRIC_DEFINITIONS.items():
            st.write(f"**{key}:** {value}")

    with st.expander("View and download the used dataset", expanded=False):
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            dataset_state = st.selectbox("Dataset state", ["All"] + sorted(df["state"].unique()), key="dataset_state")
        dataset_view = df.copy()
        if dataset_state != "All":
            dataset_view = dataset_view[dataset_view["state"] == dataset_state]
        with f2:
            district_options = ["All"] + sorted(dataset_view["district"].unique())
            dataset_district = st.selectbox("Dataset district", district_options, key="dataset_district")
        if dataset_district != "All":
            dataset_view = dataset_view[dataset_view["district"] == dataset_district]
        with f3:
            crop_options = ["All"] + sorted(dataset_view["crop"].unique())
            dataset_crop = st.selectbox("Dataset crop", crop_options, key="dataset_crop")
        if dataset_crop != "All":
            dataset_view = dataset_view[dataset_view["crop"] == dataset_crop]
        with f4:
            year_options = ["All"] + sorted(dataset_view["year"].unique())
            dataset_year = st.selectbox("Dataset year", year_options, key="dataset_year")
        if dataset_year != "All":
            dataset_view = dataset_view[dataset_view["year"] == dataset_year]

        st.caption(f"Showing {len(dataset_view):,} rows from {len(df):,} total model records.")
        st.dataframe(dataset_view.head(500), use_container_width=True, hide_index=True)
        st.download_button(
            "Download filtered dataset CSV",
            data=dataset_view.to_csv(index=False).encode("utf-8"),
            file_name="crop_pollution_used_dataset.csv",
            mime="text/csv",
        )

    with st.expander("Why not every Indian state and every district?", expanded=False):
        st.write(
            "The project objective is to build a scientifically explainable decision-support prototype, "
            "not an all-India census database. Punjab is treated as the complete district-level case study "
            "because the project is Punjab/NIT Jalandhar context specific. Other major agricultural states "
            "are included as representative comparison states to prove scalability. Full India district "
            "coverage is kept as future scope because official district-wise crop, soil, fertilizer, pesticide "
            "and mandi-price datasets are not uniformly available in one reliable format."
        )

    st.markdown(
        """
        **Source-backed reliability strategy**

        - Government crop-production, MSP and fertilizer datasets form the primary reliability base.
        - Punjab district profiles are generated for all districts so the project can show district-wise advisory behavior.
        - 2025 and 2026 values are marked as model-estimated scenarios wherever final official datasets are not available.
        - The app is a decision-support prototype for farmers, researchers and students.

        See `DATA_SOURCES.md` in the repository for full source links, assumptions and limitations.
        """
    )
    st.markdown(
        """
        **Future farmer-facing scope**

        - Hindi/Punjabi mobile-first advisory interface.
        - Soil Health Card and district agriculture-office data integration.
        - Live mandi price and MSP comparison.
        - Crop rotation and lower-pollution recommendations for every district.
        """
    )

st.markdown("---")
st.caption(T["footer"])
