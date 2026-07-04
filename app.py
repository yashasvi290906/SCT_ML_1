import os
import streamlit as st
import base64
import time
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from utils import (
    get_price_badge,
    compute_quality_score,
    generate_explanations,
    generate_premium_summary,
    create_feature_importance_chart,
    create_input_summary_chart,
    create_price_comparison_chart,
    compute_investment_scores,
    get_score_color,
    get_score_label,
    get_market_position,
    generate_recommendations,
    get_neighborhood_insights,
    compute_comparison,
    add_to_history,
    get_history,
    NEIGHBORHOODS,
    NEIGHBORHOOD_LABELS,
    compute_prediction_interval,
    compute_confidence,
    find_similar_properties,
    compute_appreciation_forecast,
    compute_investment_risk,
    get_model_reliability_metrics,
    compute_property_strengths_weaknesses,
    compute_improvement_simulator,
    compute_property_report_scorecard,
    compute_benchmark_percentiles,
    generate_prediction_transparency,
    get_dataset_insights,
    generate_ai_verdict,
    generate_value_drivers,
    generate_investment_interpretation,
    generate_next_actions,
    generate_property_profile,
)

st.set_page_config(
    page_title="HomeSense AI - Premium Real Estate Valuation",
    page_icon="house",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_model():
    model = joblib.load("model/house_price_model.pkl")
    columns = joblib.load("model/model_columns.pkl")
    return model, columns


@st.cache_data
def get_cached_dataset_insights():
    return get_dataset_insights()


model, model_columns = load_model()

with open("assets/style.css") as f:
    css = f.read()

def _bg64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""

hero_b64 = _bg64("assets/hero_bg.jpg")

st.markdown(
    f"""<style>{css}
[data-testid="stApp"]{{
  background:{"url(data:image/jpeg;base64,{hero_b64}) center/cover no-repeat fixed," if hero_b64 else ""}
  #0D0705;
}}
[data-testid="stApp"]::before{{
  content:'';position:fixed;inset:0;z-index:0;
  background:linear-gradient(180deg,rgba(13,7,5,0.45) 0%,rgba(13,7,5,0.65) 35%,rgba(13,7,5,0.88) 70%,#0D0705 100%);
  pointer-events:none;
}}
[data-testid="stApp"] > *{{position:relative;z-index:1;}}
[data-testid="stHeader"]{{background:transparent;}}
[data-testid="stToolbar"]{{display:none;}}
#MainMenu{{visibility:hidden;}}
footer{{visibility:hidden;}}
[data-testid="stSidebar"]{{background:#0D0705;}}
[data-testid="stMarkdownContainer"] p{{color:rgba(248,242,235,0.7);}}
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4{{color:#FAF2EB;font-family:'Playfair Display',serif;}}
.stHeader h1{{color:#FAF2EB;}}
.stSubheader{{color:#FAF2EB;}}
[data-testid="stMetric"]{{
  background:linear-gradient(135deg,rgba(30,18,12,0.7),rgba(42,28,18,0.55));
  border:1px solid rgba(212,168,78,0.18);
  border-radius:18px;padding:20px 24px;
  backdrop-filter:blur(24px) saturate(140%);
  box-shadow:0 8px 32px rgba(0,0,0,0.35),inset 0 1px 0 rgba(212,168,78,0.06);
  transition:all 0.4s cubic-bezier(0.4,0,0.2,1);
}}
[data-testid="stMetric"]:hover{{
  transform:translateY(-4px);
  border-color:rgba(212,168,78,0.35);
  box-shadow:0 16px 48px rgba(0,0,0,0.4),0 0 24px rgba(212,168,78,0.08),inset 0 1px 0 rgba(212,168,78,0.1);
}}
[data-testid="stMetric"] label{{color:rgba(248,242,235,0.5) !important;font-size:11px !important;text-transform:uppercase;letter-spacing:1.4px;font-weight:600;}}
[data-testid="stMetric"] [data-testid="stMetricValue"]{{color:#D4A84E !important;font-weight:700;text-shadow:0 0 20px rgba(212,168,78,0.15);}}
[data-testid="stMetric"] [data-testid="stMetricDelta"]{{font-weight:600;}}
div[data-testid="stHorizontalBlock"]{{gap:14px;}}
.stExpander{{
  background:linear-gradient(135deg,rgba(30,18,12,0.6),rgba(42,28,18,0.4));
  border:1px solid rgba(212,168,78,0.1);
  border-radius:16px;
  backdrop-filter:blur(20px);
}}
.stExpander summary{{color:#FAF2EB;}}
.stAlert{{
  background:linear-gradient(135deg,rgba(30,18,12,0.6),rgba(42,28,18,0.4));
  border:1px solid rgba(212,168,78,0.15);
  border-radius:14px;
  backdrop-filter:blur(16px);
}}
.stButton > button, .stDownloadButton > button{{
  background:linear-gradient(135deg,#D4A84E,#C4942A,#B8862A) !important;
  color:#0D0705 !important;
  border:none !important;
  border-radius:14px !important;
  font-weight:700 !important;
  box-shadow:0 4px 24px rgba(212,168,78,0.4),
             0 0 0 1px rgba(212,168,78,0.35) inset,
             0 1px 0 rgba(255,255,255,0.1) inset !important;
  transition:all 0.3s cubic-bezier(0.4,0,0.2,1) !important;
  letter-spacing:0.3px;
}}
.stButton > button:hover, .stDownloadButton > button:hover{{
  background:linear-gradient(135deg,#E8C060,#D4A84E,#C4942A) !important;
  transform:translateY(-2px) !important;
  box-shadow:0 8px 36px rgba(212,168,78,0.55),
             0 0 0 1px rgba(212,168,78,0.5) inset,
             0 0 60px rgba(212,168,78,0.12) !important;
}}
.stButton > button:active{{transform:translateY(0) !important;}}
.stProgress > div > div > div{{background:linear-gradient(90deg,#C4942A,#D4A84E,#E8C060);}}
.section-divider{{max-width:1200px;margin:0 auto;height:1px;background:linear-gradient(90deg,transparent,rgba(212,168,78,0.15),transparent);}}
.premium-card{{
  background:linear-gradient(135deg,rgba(30,18,12,0.7),rgba(42,28,18,0.5));
  backdrop-filter:blur(24px) saturate(140%);
  border:1px solid rgba(212,168,78,0.15);
  border-radius:22px;padding:30px 34px;
  box-shadow:0 10px 36px rgba(0,0,0,0.3),inset 0 1px 0 rgba(212,168,78,0.06);
  transition:all 0.4s cubic-bezier(0.4,0,0.2,1);
}}
.premium-card:hover{{
  transform:translateY(-4px);
  border-color:rgba(212,168,78,0.3);
  box-shadow:0 20px 56px rgba(0,0,0,0.4),0 0 28px rgba(212,168,78,0.08),inset 0 1px 0 rgba(212,168,78,0.1);
}}
.gold-text{{color:#D4A84E;}}
.green-text{{color:#4ECB8D;}}
.red-text{{color:#FF6B6B;}}
.amber-text{{color:#FFD060;}}
.muted-text{{color:rgba(248,242,235,0.45);}}
.primary-text{{color:#FAF2EB;}}
</style>""",
    unsafe_allow_html=True,
)

# ============================================================
# NAVBAR
# ============================================================
st.markdown(
    '<div style="position:fixed;top:16px;left:50%;transform:translateX(-50%);width:min(92%,1200px);display:flex;justify-content:space-between;align-items:center;padding:14px 36px;background:rgba(13,7,5,0.75);backdrop-filter:blur(28px) saturate(180%);border:1px solid rgba(212,168,78,0.15);border-radius:20px;z-index:9999;box-shadow:0 8px 36px rgba(0,0,0,0.5),inset 0 1px 0 rgba(212,168,78,0.06);">'
    '<span style="font-family:Playfair Display,serif;font-size:24px;color:#D4A84E;font-weight:700;">HomeSense AI</span>'
    '<span style="display:flex;gap:32px;color:rgba(250,242,235,0.65);font-size:14px;font-weight:500;">'
    '<a href="#home" style="color:inherit;text-decoration:none;">Home</a> '
    '<a href="#predict" style="color:inherit;text-decoration:none;">Predict</a> '
    '<a href="#analytics" style="color:inherit;text-decoration:none;">Analytics</a> '
    '<a href="#about" style="color:inherit;text-decoration:none;">About</a>'
    '</span></div>',
    unsafe_allow_html=True,
)

# ============================================================
# HERO
# ============================================================
st.markdown('<div id="home"></div>', unsafe_allow_html=True)

hero_l, hero_r = st.columns([1, 1], gap="large")
with hero_l:
    st.markdown(
        '<div style="padding:60px 0 0 0;">'
        '<span style="display:inline-flex;align-items:center;gap:8px;padding:8px 18px;background:rgba(212,168,78,0.1);border:1px solid rgba(212,168,78,0.3);border-radius:100px;color:#D4A84E;font-size:12px;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;">'
        '<span style="width:7px;height:7px;background:#D4A84E;border-radius:50%;animation:pulse 2s ease-in-out infinite;"></span> AI-Powered Valuation Engine</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<h1 style="font-family:Playfair Display,serif;font-size:clamp(40px,5.5vw,70px);line-height:1.08;color:#FAF2EB;font-weight:700;letter-spacing:-1.5px;margin:24px 0;">'
        'Predict Real Estate<br>Value With <span style="color:#D4A84E;">Precision</span></h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-size:18px;line-height:1.75;color:rgba(248,242,235,0.55);max-width:500px;">'
        'Leverage machine learning trained on 1,460 property records to deliver accurate, explainable house price predictions in seconds.</p>',
        unsafe_allow_html=True,
    )
    st.markdown("#")
    st.link_button("Predict Now", "#predict")
with hero_r:
    st.markdown("")
    st.markdown("")
    st.markdown("")
    with st.container():
        st.metric(label="Median Home Value", value="$163,000")
        st.caption("1,460 Properties Analyzed")

# ============================================================
# TRUST METRICS
# ============================================================
st.markdown('<div class="section-divider" style="max-width:1200px;margin:40px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(212,168,78,0.12),transparent);"></div>', unsafe_allow_html=True)

st.markdown(
    '<div style="text-align:center;"><span style="display:inline-flex;align-items:center;gap:8px;padding:6px 16px;background:rgba(212,168,78,0.08);border:1px solid rgba(212,168,78,0.18);border-radius:100px;color:#D4A84E;font-size:11px;font-weight:700;letter-spacing:2px;text-transform:uppercase;">Why HomeSense</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<h2 style="font-family:Playfair Display,serif;font-size:clamp(30px,4vw,48px);color:#FAF2EB;font-weight:700;text-align:center;letter-spacing:-1px;">Trusted by Data</h2>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="font-size:17px;color:rgba(248,242,235,0.45);text-align:center;max-width:540px;margin:0 auto 32px;">Built on rigorous machine learning research with full transparency in how every prediction is made.</p>',
    unsafe_allow_html=True,
)

t1, t2, t3, t4 = st.columns(4)
with t1:
    st.metric(label="Within 10% Accuracy", value="68%")
with t2:
    st.metric(label="Houses Trained", value="1,460")
with t3:
    st.metric(label="Explainable Insights", value="AI")
with t4:
    st.metric(label="Fast Prediction", value="<1s")

# ============================================================
# FEATURES
# ============================================================
st.markdown('<div class="section-divider" style="max-width:1200px;margin:40px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(212,168,78,0.12),transparent);"></div>', unsafe_allow_html=True)

st.markdown(
    '<div style="text-align:center;"><span style="display:inline-flex;align-items:center;gap:8px;padding:6px 16px;background:rgba(212,168,78,0.08);border:1px solid rgba(212,168,78,0.18);border-radius:100px;color:#D4A84E;font-size:11px;font-weight:700;letter-spacing:2px;text-transform:uppercase;">Capabilities</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<h2 style="font-family:Playfair Display,serif;font-size:clamp(30px,4vw,48px);color:#FAF2EB;font-weight:700;text-align:center;letter-spacing:-1px;">Built for Intelligence</h2>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="font-size:17px;color:rgba(248,242,235,0.45);text-align:center;max-width:540px;margin:0 auto 32px;">Every feature is engineered to deliver insights that matter.</p>',
    unsafe_allow_html=True,
)

f1, f2, f3, f4 = st.columns(4)
with f1:
    st.markdown(
        '<div class="premium-card" style="text-align:center;">'
        '<div style="font-size:36px;margin-bottom:12px;">target</div>'
        '<div style="color:#FAF2EB;font-weight:700;font-size:16px;margin-bottom:8px;">AI Price Prediction</div>'
        '<div style="color:rgba(248,242,235,0.45);font-size:14px;line-height:1.6;">Linear regression model trained on 245 property features to deliver accurate market valuations.</div></div>',
        unsafe_allow_html=True,
    )
with f2:
    st.markdown(
        '<div class="premium-card" style="text-align:center;">'
        '<div style="font-size:36px;margin-bottom:12px;">search</div>'
        '<div style="color:#FAF2EB;font-weight:700;font-size:16px;margin-bottom:8px;">Explainable AI</div>'
        '<div style="color:rgba(248,242,235,0.45);font-size:14px;line-height:1.6;">Understand exactly why the model predicts a price - every factor is transparent.</div></div>',
        unsafe_allow_html=True,
    )
with f3:
    st.markdown(
        '<div class="premium-card" style="text-align:center;">'
        '<div style="font-size:36px;margin-bottom:12px;">trending_up</div>'
        '<div style="color:#FAF2EB;font-weight:700;font-size:16px;margin-bottom:8px;">Feature Importance</div>'
        '<div style="color:rgba(248,242,235,0.45);font-size:14px;line-height:1.6;">Visual breakdown of which characteristics contribute most to the estimated value.</div></div>',
        unsafe_allow_html=True,
    )
with f4:
    st.markdown(
        '<div class="premium-card" style="text-align:center;">'
        '<div style="font-size:36px;margin-bottom:12px;">account_balance</div>'
        '<div style="color:#FAF2EB;font-weight:700;font-size:16px;margin-bottom:8px;">Market Insights</div>'
        '<div style="color:rgba(248,242,235,0.45);font-size:14px;line-height:1.6;">Compare your property against market tiers with visual analytics.</div></div>',
        unsafe_allow_html=True,
    )

# ============================================================
# PREDICTION FORM
# ============================================================
st.markdown('<div class="section-divider" style="max-width:1200px;margin:40px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(212,168,78,0.12),transparent);"></div>', unsafe_allow_html=True)

st.markdown('<div id="predict"></div>', unsafe_allow_html=True)
st.markdown(
    '<div style="padding:0 60px;"><span style="display:inline-flex;align-items:center;gap:8px;padding:6px 16px;background:rgba(212,168,78,0.08);border:1px solid rgba(212,168,78,0.18);border-radius:100px;color:#D4A84E;font-size:11px;font-weight:700;letter-spacing:2px;text-transform:uppercase;">Valuation Engine</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<h2 style="font-family:Playfair Display,serif;font-size:clamp(30px,4vw,48px);color:#FAF2EB;font-weight:700;padding:0 60px;letter-spacing:-1px;">Predict House Price</h2>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="font-size:17px;color:rgba(248,242,235,0.45);padding:0 60px;">Enter the property details below to generate an AI-powered valuation.</p>',
    unsafe_allow_html=True,
)

st.markdown("#")

col_left, col_center, col_right = st.columns(3)

with col_left:
    overall_quality = st.slider("Overall Quality", min_value=1, max_value=10, value=5,
                                help="Rates the overall material and finish of the house (1=Very Poor, 10=Very Excellent)")
    living_area = st.number_input("Living Area (sq ft)", min_value=400, max_value=6000, value=1500, step=50,
                                  help="Above grade (ground) living area in square feet")
    garage_cars = st.slider("Garage Capacity (cars)", min_value=0, max_value=5, value=2,
                            help="Size of garage in car capacity")

with col_center:
    bedrooms = st.slider("Bedrooms", min_value=1, max_value=8, value=3,
                         help="Number of bedrooms above grade")
    bathrooms = st.slider("Full Bathrooms", min_value=0, max_value=5, value=2,
                          help="Full bathrooms above grade")
    half_baths = st.slider("Half Bathrooms", min_value=0, max_value=4, value=0,
                           help="Half baths above grade")

with col_right:
    basement_area = st.number_input("Basement Area (sq ft)", min_value=0, max_value=3000, value=800, step=50,
                                    help="Total square feet of basement area")
    year_built = st.number_input("Year Built", min_value=1900, max_value=datetime.now().year, value=2005,
                                 help="Original construction date")
    neighborhood = st.selectbox("Neighborhood", options=NEIGHBORHOODS,
                                format_func=lambda x: NEIGHBORHOOD_LABELS.get(x, x),
                                index=NEIGHBORHOODS.index("CollgCr"),
                                help="Physical location within Ames city limits")

predict_clicked = st.button("Generate Valuation", width="stretch")

# ============================================================
# PREDICTION REPORT
# ============================================================

if predict_clicked:
    loading_placeholder = st.empty()
    timeline_steps = [
        ("Initializing AI Engine", 0.15),
        ("Processing Property Features", 0.30),
        ("Running Market Analysis", 0.50),
        ("Computing Valuation", 0.75),
        ("Generating Insights", 0.90),
        ("Finalizing Report", 1.0),
    ]
    completed_steps = []
    progress_bar = st.progress(0)
    status_text = loading_placeholder.empty()
    for step_label, progress in timeline_steps:
        completed_steps.append(step_label)
        status_text.info(f"**{step_label}**")
        progress_bar.progress(progress)
        time.sleep(0.12)
    progress_bar.empty()
    status_text.empty()
    loading_placeholder.empty()

    input_df = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)

    median_defaults = {
        "Id": 1460, "LotFrontage": 69.0, "LotArea": 9478,
        "OverallCond": 5, "YearRemodAdd": year_built, "MasVnrArea": 0.0,
        "BsmtFinSF1": 383.0, "BsmtFinSF2": 0.0, "BsmtUnfSF": 477.0,
        "1stFlrSF": 1087, "2ndFlrSF": 0, "LowQualFinSF": 0,
        "BsmtFullBath": 0, "BsmtHalfBath": 0, "KitchenAbvGr": 1,
        "TotRmsAbvGrd": 6, "Fireplaces": 1, "GarageYrBlt": year_built,
        "GarageArea": 480, "WoodDeckSF": 0, "OpenPorchSF": 25,
        "EnclosedPorch": 0, "3SsnPorch": 0, "ScreenPorch": 0,
        "PoolArea": 0, "MiscVal": 0, "MoSold": 6, "YrSold": 2023,
    }
    for col, val in median_defaults.items():
        if col in input_df.columns:
            input_df[col] = val

    input_df["OverallQual"] = overall_quality
    input_df["GrLivArea"] = living_area
    input_df["GarageCars"] = garage_cars
    input_df["TotalBsmtSF"] = basement_area
    input_df["FullBath"] = bathrooms
    input_df["HalfBath"] = half_baths
    input_df["BedroomAbvGr"] = bedrooms
    input_df["YearBuilt"] = year_built

    if f"Neighborhood_{neighborhood}" in model_columns:
        input_df[f"Neighborhood_{neighborhood}"] = 1

    try:
        prediction = np.expm1(model.predict(input_df)[0])
        if not np.isfinite(prediction) or prediction <= 0:
            raise ValueError("Invalid prediction value")
    except Exception:
        st.error("Unable to generate prediction. Please verify your inputs and try again.")
        st.stop()

    inputs = {
        "OverallQual": overall_quality,
        "GrLivArea": living_area,
        "GarageCars": garage_cars,
        "TotalBsmtSF": basement_area,
        "FullBath": bathrooms,
        "HalfBath": half_baths,
        "BedroomAbvGr": bedrooms,
        "YearBuilt": year_built,
        "Neighborhood": neighborhood,
    }

    scores = compute_investment_scores(inputs, prediction)
    market = get_market_position(prediction)
    badge_text, badge_class = get_price_badge(prediction)
    quality_score = compute_quality_score(inputs)
    ai_summary = generate_premium_summary(inputs, prediction, scores, market)
    recs = generate_recommendations(inputs, prediction)
    nb_insights = get_neighborhood_insights(neighborhood)
    comparison = compute_comparison(inputs)
    pred_interval = compute_prediction_interval(prediction)
    confidence_data = compute_confidence(inputs)
    similar_props = find_similar_properties(inputs, prediction, n=3)
    forecast = compute_appreciation_forecast(prediction, neighborhood, inputs)
    risk_analysis = compute_investment_risk(inputs, prediction, scores)
    model_metrics = get_model_reliability_metrics()
    strengths_weaknesses = compute_property_strengths_weaknesses(inputs)
    improvements = compute_improvement_simulator(inputs, prediction)
    scorecard = compute_property_report_scorecard(inputs, prediction, scores)
    benchmark_percentiles = compute_benchmark_percentiles(inputs, prediction)
    transparency = generate_prediction_transparency(inputs, prediction)
    dataset_insights = get_cached_dataset_insights()
    add_to_history(inputs, prediction, scores)

    profile = generate_property_profile(inputs, prediction)

    # ── PROPERTY PROFILE ──
    st.markdown('<div class="section-divider" style="max-width:1200px;margin:40px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(212,168,78,0.12),transparent);"></div>', unsafe_allow_html=True)

    col_info, col_img = st.columns([3, 2])
    with col_info:
        st.caption("Property Profile")
        st.subheader("Property Visualization")
        st.markdown(f"**{profile['label']}**")
        st.write(profile["description"])
        st.markdown("**Best Suited For**")
        for tag in profile["lifestyles"]:
            st.markdown(f"check {tag}")
    with col_img:
        has_image = profile["image_path"] and os.path.isfile(profile["image_path"])
        if has_image:
            st.image(profile["image_path"], width="stretch")
            st.caption(f"Representative Image - {profile['label']}")
            st.caption("Visualization: Representative image selected based on your property's predicted characteristics. This image is illustrative and not a reconstruction of the actual property.")
        else:
            st.info("Property Visualization")

    # PROPERTY FACTS
    st.markdown('<div class="section-divider" style="max-width:1200px;margin:40px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(212,168,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    st.caption("Details")
    st.subheader("Property Facts")

    facts = [
        ("House Style", profile["label"].replace(" Residence", "").replace(" Home", "").replace(" Property", "")),
        ("Bedrooms", str(inputs.get("BedroomAbvGr", 3))),
        ("Bathrooms", str(inputs.get("FullBath", 2))),
        ("Garage", f"{inputs.get('GarageCars', 2)} Car{'s' if inputs.get('GarageCars', 2) != 1 else ''}"),
        ("Living Area", f"{inputs.get('GrLivArea', 1500):,} sq.ft"),
        ("Year Built", str(inputs.get("YearBuilt", 2005))),
        ("Overall Quality", f"{inputs.get('OverallQual', 5)}/10"),
        ("Neighborhood", NEIGHBORHOOD_LABELS.get(inputs.get("Neighborhood", ""), inputs.get("Neighborhood", ""))),
        ("Est. Value", f"${prediction:,.0f}"),
    ]
    fact_cols = st.columns(3)
    for i, (label, value) in enumerate(facts):
        with fact_cols[i % 3]:
            st.metric(label=label, value=value)

    # MARKET SEGMENT
    st.markdown('<div class="section-divider" style="max-width:1200px;margin:40px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(212,168,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    st.caption("Positioning")
    st.subheader("Market Segment")
    st.progress(profile["segment_pct"] / 100)
    st.markdown(f"**{profile['market_segment']}**")

    # PROPERTY HIGHLIGHTS
    if profile["highlights"]:
        st.markdown('<div class="section-divider" style="max-width:1200px;margin:40px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(212,168,78,0.12),transparent);"></div>', unsafe_allow_html=True)
        st.caption("Strengths")
        st.subheader("Property Highlights")
        for h in profile["highlights"]:
            st.markdown(
                f'<div class="premium-card" style="margin-bottom:12px;border-left:3px solid #4ECB8D;">'
                f'<strong style="color:#FAF2EB;">{h["icon"]} {h["title"]}</strong><br>'
                f'<span style="color:rgba(248,242,235,0.55);font-size:14px;">{h["detail"]}</span></div>',
                unsafe_allow_html=True,
            )

    # DRAWBACKS
    if profile["drawbacks"]:
        st.markdown('<div class="section-divider" style="max-width:1200px;margin:40px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(212,168,78,0.12),transparent);"></div>', unsafe_allow_html=True)
        st.caption("Considerations")
        st.subheader("Potential Drawbacks")
        for d in profile["drawbacks"]:
            st.markdown(
                f'<div class="premium-card" style="margin-bottom:12px;border-left:3px solid #FFD060;">'
                f'<strong style="color:#FAF2EB;">{d["icon"]} {d["title"]}</strong><br>'
                f'<span style="color:rgba(248,242,235,0.55);font-size:14px;">{d["detail"]}</span></div>',
                unsafe_allow_html=True,
            )

    # PROFILE VERDICT
    st.markdown('<div class="section-divider" style="max-width:1200px;margin:40px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(212,168,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    rec_class = "excellent" if profile["score"] >= 8 else "good" if profile["score"] >= 6 else "fair"
    if profile["score"] >= 8:
        segment_desc = "Its excellent construction quality, generous living area and modern features position it well above the dataset average."
    elif profile["score"] >= 6:
        segment_desc = f"Its solid construction and balanced features position it competitively within the {profile['market_segment'].lower()} segment."
    else:
        segment_desc = "While it meets basic residential standards, there are areas where this property could improve to compete more effectively."

    st.caption("Verdict")
    st.subheader("HomeSense AI Verdict")
    st.markdown(
        f'<div class="premium-card" style="text-align:center;">'
        f'<div style="font-family:Playfair Display,serif;font-size:56px;color:#D4A84E;font-weight:700;">{profile["score"]:.1f}</div>'
        f'<div style="color:rgba(248,242,235,0.4);font-size:13px;text-transform:uppercase;letter-spacing:1px;margin-bottom:20px;">Overall Score</div>'
        f'<div style="display:inline-block;padding:8px 24px;border-radius:100px;font-size:14px;font-weight:700;margin-bottom:20px;'
        f'{"background:rgba(78,203,141,0.1);border:1px solid rgba(78,203,141,0.25);color:#4ECB8D;" if rec_class == "excellent" else "background:rgba(212,168,78,0.1);border:1px solid rgba(212,168,78,0.25);color:#D4A84E;" if rec_class == "good" else "background:rgba(255,208,96,0.1);border:1px solid rgba(255,208,96,0.25);color:#FFD060;"}'
        f'">{profile["recommendation"]}</div>'
        f'<div style="color:rgba(248,242,235,0.55);line-height:1.8;max-width:600px;margin:0 auto;">'
        f'This property belongs to the <strong>{profile["market_segment"]}</strong> market segment. {segment_desc}</div></div>',
        unsafe_allow_html=True,
    )

    # ── VALUATION CARD ──
    st.markdown('<div class="section-divider" style="max-width:1200px;margin:40px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(212,168,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    confidence = confidence_data["score"]
    confidence_color = "#4ECB8D" if confidence >= 85 else "#D4A84E" if confidence >= 70 else "#FFD060"
    verdict = generate_ai_verdict(inputs, prediction, scores, market)
    value_drivers = generate_value_drivers(inputs, prediction)
    investment_interp = generate_investment_interpretation(inputs, prediction, scores, market)

    st.caption("Property Valuation")
    st.subheader("Valuation")
    st.markdown(
        f'<div class="premium-card" style="text-align:center;">'
        f'<span style="display:inline-block;padding:6px 16px;border-radius:100px;font-size:12px;font-weight:700;'
        f'{"background:rgba(78,203,141,0.1);border:1px solid rgba(78,203,141,0.25);color:#4ECB8D;" if badge_class == "premium" else "background:rgba(212,168,78,0.1);border:1px solid rgba(212,168,78,0.25);color:#D4A84E;"}'
        f'">{badge_text}</span><br><br>'
        f'<div style="font-family:Playfair Display,serif;font-size:clamp(40px,5vw,64px);color:#D4A84E;font-weight:700;">${prediction:,.0f}</div>'
        f'<div style="color:rgba(248,242,235,0.45);font-size:15px;margin-bottom:20px;">Estimated Market Value</div>'
        f'<div style="color:rgba(248,242,235,0.35);font-size:13px;">Likely Range: ${pred_interval["lower"]:,.0f} - ${pred_interval["upper"]:,.0f} (95% CI)</div>'
        f'<div style="margin-top:12px;"><span style="color:rgba(248,242,235,0.45);font-size:13px;">Confidence: </span>'
        f'<span style="color:{confidence_color};font-weight:700;">{confidence}%</span> '
        f'<span style="color:rgba(248,242,235,0.35);font-size:12px;">{confidence_data["grade"]} Confidence</span></div></div>',
        unsafe_allow_html=True,
    )

    # ── AI VERDICT ──
    st.markdown('<div class="section-divider" style="max-width:1200px;margin:40px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(212,168,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    st.caption("Verdict")
    st.subheader("AI Valuation Verdict")
    st.markdown(
        f'<div class="premium-card" style="border-left:4px solid {verdict["color"]};">'
        f'<div style="display:inline-block;padding:6px 18px;border-radius:100px;font-size:13px;font-weight:700;border:1px solid {verdict["color"]};color:{verdict["color"]};margin-bottom:16px;">{verdict["verdict"]}</div>'
        f'<div style="color:rgba(248,242,235,0.65);line-height:1.7;margin-bottom:12px;">{verdict["reasoning"]}</div>'
        f'<div style="color:{verdict["color"]};font-weight:600;">{verdict["action"]}</div></div>',
        unsafe_allow_html=True,
    )

    # ── WHAT DRIVES THIS PRICE ──
    st.markdown('<div class="section-divider" style="max-width:1200px;margin:40px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(212,168,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    st.caption("Value Assessment")
    st.subheader("What Drives This Price")
    st.markdown('<p style="color:rgba(248,242,235,0.45);">Key factors that determine the estimated value of this property.</p>', unsafe_allow_html=True)
    for d in value_drivers:
        st.markdown(
            f'<div class="premium-card" style="margin-bottom:12px;">'
            f'<strong style="color:#FAF2EB;">{d["icon"]} {d["title"]}</strong><br>'
            f'<span style="color:rgba(248,242,235,0.55);font-size:14px;">{d["detail"]}</span></div>',
            unsafe_allow_html=True,
        )

    # ── INVESTMENT INTERPRETATION ──
    st.markdown('<div class="section-divider" style="max-width:1200px;margin:40px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(212,168,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    st.caption("Analyst Perspective")
    st.subheader("If This Were My Investment")
    st.markdown(
        f'<div class="premium-card" style="border-left:3px solid #D4A84E;">'
        f'<p style="color:rgba(248,242,235,0.65);line-height:1.8;font-style:italic;">{investment_interp}</p></div>',
        unsafe_allow_html=True,
    )

    # ── PROPERTY ASSESSMENT ──
    st.markdown('<div class="section-divider" style="max-width:1200px;margin:40px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(212,168,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    st.caption("Assessment")
    st.subheader("Property Assessment")
    st.markdown('<p style="color:rgba(248,242,235,0.45);">Areas where this property excels and where it falls short.</p>', unsafe_allow_html=True)

    sw_left, sw_right = st.columns(2)
    with sw_left:
        st.markdown("**Strengths**")
        for s in strengths_weaknesses["strengths"]:
            st.markdown(
                f'<div class="premium-card" style="margin-bottom:8px;border-left:3px solid #4ECB8D;padding:14px 18px;">'
                f'<strong style="color:#4ECB8D;">check {s["feature"]}</strong><br>'
                f'<span style="color:#D4A84E;font-weight:600;">{s["value"]}</span><br>'
                f'<span style="color:rgba(248,242,235,0.45);font-size:13px;">{s["detail"]}</span></div>',
                unsafe_allow_html=True,
            )
        if not strengths_weaknesses["strengths"]:
            st.info("No standout strengths detected.")
    with sw_right:
        st.markdown("**Weaknesses**")
        for w in strengths_weaknesses["weaknesses"]:
            st.markdown(
                f'<div class="premium-card" style="margin-bottom:8px;border-left:3px solid #FF6B6B;padding:14px 18px;">'
                f'<strong style="color:#FF6B6B;">close {w["feature"]}</strong><br>'
                f'<span style="color:#D4A84E;font-weight:600;">{w["value"]}</span><br>'
                f'<span style="color:rgba(248,242,235,0.45);font-size:13px;">{w["detail"]}</span></div>',
                unsafe_allow_html=True,
            )
        if not strengths_weaknesses["weaknesses"]:
            st.info("No significant weaknesses detected.")

    # ── EXECUTIVE SUMMARY ──
    st.markdown('<div class="section-divider" style="max-width:1200px;margin:40px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(212,168,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    st.caption("Summary")
    st.subheader("Executive Summary")
    st.markdown(
        f'<div class="premium-card" style="border-left:3px solid #D4A84E;">'
        f'<p style="color:rgba(248,242,235,0.65);line-height:1.8;">{ai_summary}</p></div>',
        unsafe_allow_html=True,
    )

    # ── INVESTMENT SCORES ──
    st.markdown('<div class="section-divider" style="max-width:1200px;margin:40px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(212,168,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    st.caption("Investment Analysis")
    st.subheader("Investment Scores")
    st.markdown('<p style="color:rgba(248,242,235,0.45);">Quantified assessment based on property characteristics and market position.</p>', unsafe_allow_html=True)

    score_items = [("Investment Score", "investment"), ("Rental Potential", "rental"), ("Resale Value", "resale"), ("Luxury Score", "luxury"), ("Market Safety", "risk")]
    score_cols = st.columns(len(score_items))
    for i, (label, key) in enumerate(score_items):
        val = scores[key]
        color = get_score_color(val)
        with score_cols[i]:
            st.metric(label=label, value=f"{val}/100")
            st.progress(val / 100)
            st.markdown(f'<span style="color:{color};font-weight:600;">{get_score_label(val)}</span>', unsafe_allow_html=True)

    # ── PROPERTY REPORT CARD ──
    st.markdown('<div class="section-divider" style="max-width:1200px;margin:40px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(212,168,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    st.caption("Scorecard")
    st.subheader("Property Report Card")
    st.markdown('<p style="color:rgba(248,242,235,0.45);">Comprehensive scoring across 7 key categories.</p>', unsafe_allow_html=True)

    total_score = sum(sc["score"] for sc in scorecard)
    max_total = sum(sc["max"] for sc in scorecard)
    overall_grade = "A" if total_score >= max_total * 0.8 else "B" if total_score >= max_total * 0.65 else "C" if total_score >= max_total * 0.5 else "D"

    st.markdown(
        f'<div class="premium-card" style="text-align:center;margin-bottom:20px;">'
        f'<div style="font-family:Playfair Display,serif;font-size:48px;color:#D4A84E;font-weight:700;">{overall_grade}</div>'
        f'<div style="color:rgba(248,242,235,0.4);font-size:14px;">{total_score}/{max_total} Overall</div></div>',
        unsafe_allow_html=True,
    )
    for sc in scorecard:
        sc_color = "#4ECB8D" if sc["score"] >= 8 else "#D4A84E" if sc["score"] >= 6 else "#FFD060" if sc["score"] >= 4 else "#FF6B6B"
        st.markdown(
            f'<div class="premium-card" style="margin-bottom:8px;padding:14px 20px;">'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:6px;">'
            f'<span style="color:#FAF2EB;font-weight:600;">{sc["category"]}</span>'
            f'<span style="color:{sc_color};font-weight:700;">{sc["score"]}/{sc["max"]}</span></div>'
            f'<div style="height:6px;background:rgba(255,255,255,0.04);border-radius:6px;overflow:hidden;margin-bottom:6px;">'
            f'<div style="height:100%;width:{sc["score"]/sc["max"]*100}%;background:{sc_color};border-radius:6px;"></div></div>'
            f'<div style="color:rgba(248,242,235,0.4);font-size:13px;">{sc["detail"]}</div></div>',
            unsafe_allow_html=True,
        )

    # ── MARKET POSITION ──
    st.markdown('<div class="section-divider" style="max-width:1200px;margin:40px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(212,168,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    pos = market["percentile"]
    tier_colors = {"Entry-Level": "#FF6B6B", "Value": "#FFD060", "Average": "#D4A84E", "Above Average": "#4ECB8D", "Premium": "#4ECB8D", "Luxury": "#D4A84E"}
    tier_color = tier_colors.get(market["tier"], "#D4A84E")
    vs_avg_sign = "+" if market["vs_avg"] >= 0 else ""

    st.caption("Market Analysis")
    st.subheader("Market Position")
    st.markdown(
        f'<div class="premium-card">'
        f'<div style="display:inline-block;padding:8px 20px;border-radius:100px;font-size:14px;font-weight:700;border:1px solid {tier_color};color:{tier_color};margin-bottom:12px;">{market["tier"]}</div>'
        f'<div style="color:rgba(248,242,235,0.55);margin-bottom:16px;">{market["tier_desc"]}</div>'
        f'<div style="height:8px;background:rgba(255,255,255,0.04);border-radius:8px;position:relative;margin:16px 0;">'
        f'<div style="height:100%;width:{pos}%;background:linear-gradient(90deg,rgba(212,168,78,0.3),rgba(212,168,78,0.7));border-radius:8px;"></div></div>'
        f'<div style="display:flex;justify-content:space-between;color:rgba(248,242,235,0.35);font-size:12px;margin-bottom:16px;">'
        f'<span>Budget</span><span>Average</span><span>Premium</span><span>Luxury</span></div></div>',
        unsafe_allow_html=True,
    )
    mp1, mp2, mp3 = st.columns(3)
    with mp1:
        st.metric(label="Percentile", value=f"{pos}th")
    with mp2:
        st.metric(label="vs Dataset Average", value=f"{vs_avg_sign}${market['vs_avg']:,.0f}")
    with mp3:
        st.metric(label="vs Average %", value=f"{vs_avg_sign}{market['vs_avg_pct']:.1f}%")

    # ── TOP VALUATION FACTORS ──
    st.markdown('<div class="section-divider" style="max-width:1200px;margin:40px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(212,168,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    insights = generate_explanations(inputs, prediction)
    st.caption("Key Drivers")
    st.subheader("Top Valuation Factors")
    st.markdown('<p style="color:rgba(248,242,235,0.45);">The most influential characteristics affecting this property estimated value.</p>', unsafe_allow_html=True)
    for insight in insights:
        ic = "#4ECB8D" if insight["type"] == "positive" else "#FF6B6B" if insight["type"] == "negative" else "#FFD060"
        st.markdown(
            f'<div class="premium-card" style="margin-bottom:12px;border-left:3px solid {ic};">'
            f'<strong style="color:#FAF2EB;">{insight["icon"]} {insight["title"]}</strong><br>'
            f'<span style="color:rgba(248,242,235,0.55);font-size:14px;">{insight["description"]}</span><br>'
            f'<span style="color:{ic};font-size:13px;font-weight:600;">{insight.get("contribution", "")}</span></div>',
            unsafe_allow_html=True,
        )

    # ── VISUAL INSIGHTS ──
    st.markdown('<div class="section-divider" style="max-width:1200px;margin:40px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(212,168,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    st.markdown('<div id="analytics"></div>', unsafe_allow_html=True)
    st.caption("Analytics")
    st.subheader("Visual Insights")
    st.markdown('<p style="color:rgba(248,242,235,0.45);">Interactive charts to understand the valuation from every angle.</p>', unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(create_feature_importance_chart(), width="stretch", config={"displayModeBar": False})
        st.plotly_chart(create_price_comparison_chart(prediction), width="stretch", config={"displayModeBar": False})
    with chart_col2:
        st.plotly_chart(create_input_summary_chart(inputs), width="stretch", config={"displayModeBar": False})

    # ── RECOMMENDATIONS ──
    st.markdown('<div class="section-divider" style="max-width:1200px;margin:40px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(212,168,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    st.caption("Recommendations")
    st.subheader("Recommendations")
    st.markdown('<p style="color:rgba(248,242,235,0.45);">Targeted improvements to increase property value and marketability.</p>', unsafe_allow_html=True)
    for rec in recs:
        st.markdown(
            f'<div class="premium-card" style="margin-bottom:12px;">'
            f'<strong style="color:#FAF2EB;">{rec["icon"]} {rec["title"]}</strong><br>'
            f'<span style="color:rgba(248,242,235,0.55);font-size:14px;">{rec["description"]}</span><br>'
            f'<span style="color:#D4A84E;font-size:13px;font-weight:600;">{rec["impact"]}</span></div>',
            unsafe_allow_html=True,
        )

    # ── SMART IMPROVEMENT SIMULATOR ──
    st.markdown('<div class="section-divider" style="max-width:1200px;margin:40px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(212,168,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    st.caption("Improvements")
    st.subheader("Smart Improvement Simulator")
    st.markdown('<p style="color:rgba(248,242,235,0.45);">Potential improvements with estimated costs and value impact.</p>', unsafe_allow_html=True)
    for imp in improvements:
        roi_color = "#4ECB8D" if imp["roi"] >= 100 else "#D4A84E" if imp["roi"] >= 50 else "#FFD060"
        cost_text = f"${imp['cost_low']:,.0f} - ${imp['cost_high']:,.0f}" if imp["cost_low"] > 0 else "N/A"
        value_text = f"+${imp['est_value']:,.0f}" if imp["est_value"] > 0 else "N/A"
        st.markdown(
            f'<div class="premium-card" style="margin-bottom:12px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
            f'<strong style="color:#FAF2EB;">{imp["feature"]}</strong>'
            f'<span style="color:{roi_color};font-weight:700;">{imp["roi"]}% ROI</span></div>'
            f'<div style="color:rgba(248,242,235,0.55);font-size:14px;margin-bottom:8px;">{imp["description"]}</div>'
            f'<div style="display:flex;gap:24px;">'
            f'<div><span style="color:rgba(248,242,235,0.4);font-size:12px;">Est. Cost</span><br><span style="color:#FAF2EB;font-weight:600;">{cost_text}</span></div>'
            f'<div><span style="color:rgba(248,242,235,0.4);font-size:12px;">Est. Value Add</span><br><span style="color:#4ECB8D;font-weight:600;">{value_text}</span></div></div></div>',
            unsafe_allow_html=True,
        )

    # ── NEIGHBORHOOD INSIGHTS ──
    st.markdown('<div class="section-divider" style="max-width:1200px;margin:40px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(212,168,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    nb = nb_insights
    st.caption("Location Intelligence")
    st.subheader("Neighborhood Insights")
    nb1, nb2 = st.columns([2, 1])
    with nb1:
        st.markdown(
            f'<div class="premium-card">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">'
            f'<strong style="color:#FAF2EB;font-size:18px;">{nb["name"]}</strong>'
            f'<span style="padding:4px 14px;border-radius:100px;font-size:12px;font-weight:600;background:rgba(212,168,78,0.1);border:1px solid rgba(212,168,78,0.2);color:#D4A84E;">{nb["tier"]}</span></div></div>',
            unsafe_allow_html=True,
        )
    with nb2:
        nbc1, nbc2 = st.columns(2)
        with nbc1:
            st.metric(label="Avg Price", value=f"${nb['avg_price']:,.0f}")
            st.metric(label="Demand", value=nb["demand"])
        with nbc2:
            st.metric(label="Typical Buyers", value=nb["buyers"])
            st.metric(label="Price Trend", value=nb["trend"])

    # ── COMPARE AGAINST AVERAGE ──
    st.markdown('<div class="section-divider" style="max-width:1200px;margin:40px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(212,168,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    st.caption("Benchmark Analysis")
    st.subheader("Compare Against Average")
    st.markdown('<p style="color:rgba(248,242,235,0.45);">How this property compares to the training dataset averages.</p>', unsafe_allow_html=True)
    for comp in comparison:
        diff_color = "#4ECB8D" if comp["better"] else "#FF6B6B"
        diff_sign = "+" if comp["difference"] > 0 else ""
        st.markdown(
            f'<div class="premium-card" style="margin-bottom:8px;padding:14px 20px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<span style="color:#FAF2EB;font-weight:600;min-width:120px;">{comp["name"]}</span>'
            f'<span style="color:#D4A84E;font-weight:700;">{comp["yours"]}{comp["unit"]}</span>'
            f'<span style="color:rgba(248,242,235,0.3);">vs</span>'
            f'<span style="color:rgba(248,242,235,0.55);">{comp["average"]}{comp["unit"]}</span>'
            f'<span style="color:{diff_color};font-weight:600;">{diff_sign}{comp["difference"]}{comp["unit"]} ({diff_sign}{comp["pct"]}%)</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    # ── BENCHMARK AGAINST DATASET ──
    if benchmark_percentiles:
        st.markdown('<div class="section-divider" style="max-width:1200px;margin:40px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(212,168,78,0.12),transparent);"></div>', unsafe_allow_html=True)
        st.caption("Percentile Rankings")
        st.subheader("Benchmark Against Dataset")
        st.markdown(f'<p style="color:rgba(248,242,235,0.45);">Where your property ranks among {dataset_insights["total_records"]:,} training records.</p>', unsafe_allow_html=True)
        for bp in benchmark_percentiles:
            tier_color_b = "#4ECB8D" if "Top" in bp["tier"] or "Above" in bp["tier"] else "#D4A84E" if bp["tier"] == "Average" else "#FFD060"
            st.markdown(
                f'<div class="premium-card" style="margin-bottom:8px;padding:14px 20px;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">'
                f'<span style="color:#FAF2EB;font-weight:600;">{bp["feature"]}</span>'
                f'<span style="color:#D4A84E;font-weight:600;">{bp["value"]}{bp["unit"]}</span></div>'
                f'<div style="display:flex;align-items:center;gap:12px;">'
                f'<div style="flex:1;height:6px;background:rgba(255,255,255,0.04);border-radius:6px;overflow:hidden;">'
                f'<div style="height:100%;width:{bp["percentile"]}%;background:linear-gradient(90deg,rgba(212,168,78,0.3),rgba(212,168,78,0.7));border-radius:6px;"></div></div>'
                f'<span style="color:rgba(248,242,235,0.4);font-size:13px;min-width:40px;">{bp["percentile"]:.0f}th</span>'
                f'<span style="color:{tier_color_b};font-weight:600;font-size:13px;min-width:80px;">{bp["tier"]}</span></div></div>',
                unsafe_allow_html=True,
            )

    # ── SIMILAR PROPERTIES ──
    if similar_props:
        st.markdown('<div class="section-divider" style="max-width:1200px;margin:40px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(212,168,78,0.12),transparent);"></div>', unsafe_allow_html=True)
        st.caption("Comparable Analysis")
        st.subheader("Similar Properties")
        st.markdown('<p style="color:rgba(248,242,235,0.45);">Properties from the training dataset most similar to yours.</p>', unsafe_allow_html=True)
        sim_cols = st.columns(len(similar_props))
        for i, prop in enumerate(similar_props):
            with sim_cols[i]:
                price_diff_sign = "+" if prop["price_diff"] >= 0 else ""
                diff_color_s = "#4ECB8D" if prop["price_diff"] >= 0 else "#FF6B6B"
                st.markdown(
                    f'<div class="premium-card" style="text-align:center;">'
                    f'<div style="display:flex;justify-content:space-between;margin-bottom:8px;">'
                    f'<span style="color:rgba(248,242,235,0.4);font-size:13px;">#{prop["rank"]}</span>'
                    f'<span style="color:#D4A84E;font-size:13px;font-weight:600;">{prop["similarity"]}% match</span></div>'
                    f'<div style="font-family:Playfair Display,serif;font-size:28px;color:#D4A84E;font-weight:700;margin-bottom:12px;">${prop["sale_price"]:,.0f}</div>'
                    f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;text-align:left;margin-bottom:12px;">'
                    f'<div><span style="color:rgba(248,242,235,0.4);font-size:11px;">Quality</span><br><span style="color:#FAF2EB;font-weight:600;">{prop["quality"]}/10</span></div>'
                    f'<div><span style="color:rgba(248,242,235,0.4);font-size:11px;">Area</span><br><span style="color:#FAF2EB;font-weight:600;">{prop["area"]:,} sqft</span></div>'
                    f'<div><span style="color:rgba(248,242,235,0.4);font-size:11px;">Garage</span><br><span style="color:#FAF2EB;font-weight:600;">{prop["garage"]} car</span></div>'
                    f'<div><span style="color:rgba(248,242,235,0.4);font-size:11px;">Year</span><br><span style="color:#FAF2EB;font-weight:600;">{prop["year"]}</span></div></div>'
                    f'<div style="color:{diff_color_s};font-weight:600;font-size:13px;">{price_diff_sign}${prop["price_diff"]:,.0f} vs your prediction</div></div>',
                    unsafe_allow_html=True,
                )

    # ── APPRECIATION FORECAST ──
    st.markdown('<div class="section-divider" style="max-width:1200px;margin:40px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(212,168,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    st.caption("Growth Projection")
    st.subheader("Appreciation Forecast")
    st.markdown(f'<p style="color:rgba(248,242,235,0.45);">Projected property value growth based on {forecast["rate_label"].lower()} ({forecast["annual_rate"]}% annual) trends in {nb_insights["name"]}.</p>', unsafe_allow_html=True)
    fc_cols = st.columns(len(forecast["forecasts"]))
    for i, fc in enumerate(forecast["forecasts"]):
        with fc_cols[i]:
            st.metric(label=fc["label"], value=f"${fc['value']:,.0f}", delta=f"+${fc['gain']:,.0f} (+{fc['gain_pct']}%)")
    st.caption(forecast["disclaimer"])

    # ── INVESTMENT RISK ANALYSIS ──
    st.markdown('<div class="section-divider" style="max-width:1200px;margin:40px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(212,168,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    st.caption("Risk Assessment")
    st.subheader("Investment Risk Analysis")
    st.markdown(f'<p style="color:rgba(248,242,235,0.45);">{risk_analysis["risk_desc"]}</p>', unsafe_allow_html=True)
    sev_colors = {"high": "#FF6B6B", "moderate": "#FFD060", "low": "#4ECB8D"}

    risk_header_l, risk_header_r = st.columns([1, 2])
    with risk_header_l:
        st.markdown(
            f'<div class="premium-card" style="text-align:center;">'
            f'<div style="display:inline-block;padding:8px 20px;border-radius:100px;font-size:14px;font-weight:700;border:1px solid {risk_analysis["risk_color"]};color:{risk_analysis["risk_color"]};">{risk_analysis["risk_level"]} Risk</div>'
            f'<div style="margin-top:12px;color:rgba(248,242,235,0.45);font-size:13px;">Protection Score: <strong style="color:#FAF2EB;">{risk_analysis["protection_score"]}/100</strong></div></div>',
            unsafe_allow_html=True,
        )
    with risk_header_r:
        for rf in risk_analysis["factors"]:
            sev_color = sev_colors.get(rf["severity"], "#FFD060")
            st.markdown(
                f'<div class="premium-card" style="margin-bottom:8px;padding:12px 16px;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<div><strong style="color:#FAF2EB;">{rf["icon"]} {rf["factor"]}</strong><br>'
                f'<span style="color:rgba(248,242,235,0.45);font-size:13px;">{rf["detail"]}</span></div>'
                f'<span style="padding:4px 12px;border-radius:100px;font-size:11px;font-weight:700;border:1px solid {sev_color};color:{sev_color};">{rf["severity"].upper()}</span></div></div>',
                unsafe_allow_html=True,
            )
        if not risk_analysis["factors"]:
            st.info("No significant risk factors detected. Strong investment profile.")

    # ── TECHNICAL DETAILS ──
    st.markdown('<div class="section-divider" style="max-width:1200px;margin:40px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(212,168,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    with st.expander("Technical Details - Model Performance & Methodology", expanded=False):
        tc1, tc2, tc3, tc4 = st.columns(4)
        with tc1:
            st.metric(label="R-squared (Log Space)", value=f"{model_metrics['r_squared_log']:.4f}")
            st.caption(f"Explains ~{model_metrics['r_squared_log']*100:.0f}% of price variance")
        with tc2:
            st.metric(label="RMSE", value=f"${model_metrics['rmse']:,}")
            st.caption("Root mean squared error")
        with tc3:
            st.metric(label="MAE", value=f"${model_metrics['mae']:,}")
            st.caption("Mean absolute error")
        with tc4:
            st.metric(label="Within 10%", value=f"{model_metrics['within_10_pct']:.0f}%")
            st.caption("Training predictions within 10%")

        st.markdown("**How This Prediction Was Generated**")
        for step in transparency["steps"]:
            st.markdown(f"check **{step['step']}** - {step['description']}")

        st.markdown("**Limitations**")
        for limit in transparency["limitations"]:
            st.warning(limit)

        ds = dataset_insights
        st.markdown(f"**Dataset Context** ({ds['total_records']:,} training records)")
        dc1, dc2, dc3, dc4, dc5 = st.columns(5)
        with dc1:
            st.metric(label="Median Price", value=f"${ds['median_price']:,}")
        with dc2:
            st.metric(label="Average Price", value=f"${ds['avg_price']:,}")
        with dc3:
            st.metric(label="Price Range", value=f"${ds['min_price']:,} - ${ds['max_price']:,}")
        with dc4:
            st.metric(label="Avg Living Area", value=f"{ds['avg_living_area']:,} sq ft")
        with dc5:
            st.metric(label="Avg Quality", value=f"{ds['avg_quality']}/10")

    # ── NEXT ACTIONS ──
    st.markdown('<div class="section-divider" style="max-width:1200px;margin:40px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(212,168,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    next_actions = generate_next_actions()
    st.caption("Next Steps")
    st.subheader("What Would You Like To Do?")
    for a in next_actions:
        st.markdown(
            f'<div class="premium-card" style="margin-bottom:12px;">'
            f'<strong style="color:#FAF2EB;">{a["icon"]} {a["title"]}</strong><br>'
            f'<span style="color:rgba(248,242,235,0.55);font-size:14px;">{a["desc"]}</span></div>',
            unsafe_allow_html=True,
        )

    # ── DOWNLOAD REPORT ──
    st.markdown('<div class="section-divider" style="max-width:1200px;margin:40px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(212,168,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    from utils.pdf_report import generate_pdf_report

    try:
        pdf_bytes = generate_pdf_report(inputs, prediction, quality_score, ai_summary, scores, forecast, risk_analysis)
    except Exception:
        pdf_bytes = None

    st.caption("Export")
    st.subheader("Download Report")
    st.markdown('<p style="color:rgba(248,242,235,0.45);">Export this valuation as a professional PDF report.</p>', unsafe_allow_html=True)

    dl_col1, dl_col2, dl_col3 = st.columns([1, 2, 1])
    with dl_col2:
        if pdf_bytes:
            st.download_button(label="Download PDF Report", data=pdf_bytes, file_name="HomeSense_Valuation_Report.pdf", mime="application/pdf", width="stretch")
        else:
            st.info("PDF report is temporarily unavailable. Please try again later.")

    # ── RECENT PREDICTIONS ──
    st.markdown('<div class="section-divider" style="max-width:1200px;margin:40px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(212,168,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    history = get_history()
    if history:
        st.caption("History")
        st.subheader("Recent Predictions")
        for h in history:
            st.markdown(
                f'<div class="premium-card" style="margin-bottom:8px;padding:14px 20px;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<div><span style="color:rgba(248,242,235,0.4);font-size:12px;">{h["timestamp"]}</span><br>'
                f'<span style="color:rgba(248,242,235,0.55);font-size:13px;">Q{h["quality"]} | {h["area"]:,} sq ft | {h["neighborhood"]}</span></div>'
                f'<div style="text-align:right;"><div style="font-family:Playfair Display,serif;font-size:20px;color:#D4A84E;font-weight:700;">${h["price"]:,.0f}</div>'
                f'<div style="color:{get_score_color(h["investment_score"])};font-weight:600;font-size:13px;">{h["investment_score"]}/100</div></div></div></div>',
                unsafe_allow_html=True,
            )

# ============================================================
# ABOUT
# ============================================================
st.markdown('<div class="section-divider" style="max-width:1200px;margin:40px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(212,168,78,0.12),transparent);"></div>', unsafe_allow_html=True)

st.markdown('<div id="about"></div>', unsafe_allow_html=True)
st.caption("About")
st.subheader("About HomeSense AI")

about_l, about_r = st.columns([2, 1])
with about_l:
    st.markdown("**Machine Learning Meets Real Estate**")
    st.markdown(
        "HomeSense AI uses a Linear Regression model trained on the Ames Housing Dataset - "
        "one of the most comprehensive real estate datasets available with 1,460 property records "
        "and 245 encoded features."
    )
    st.markdown(
        "Every prediction is backed by statistical rigor and full explainability. "
        "No black boxes - every factor contributing to the price is transparent and interpretable."
    )
with about_r:
    ac1, ac2 = st.columns(2)
    with ac1:
        st.metric(label="Training Records", value="1,460")
        st.metric(label="Within 10% Accuracy", value="68%")
    with ac2:
        st.metric(label="Features Analyzed", value="245")
        st.metric(label="Regression Model", value="Linear")

# ============================================================
# FOOTER
# ============================================================
st.markdown('<div class="section-divider" style="max-width:1200px;margin:40px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(212,168,78,0.12),transparent);"></div>', unsafe_allow_html=True)

fc1, fc2, fc3, fc4 = st.columns([2, 1, 1, 1])
with fc1:
    st.markdown("**HomeSense AI**")
    st.markdown('<p style="color:rgba(248,242,235,0.45);font-size:14px;">Premium AI-powered real estate valuation platform. Built with machine learning, explainable AI, and a commitment to transparency.</p>', unsafe_allow_html=True)
with fc2:
    st.markdown("**Product**")
    st.markdown("Predict")
    st.markdown("Analytics")
    st.markdown("About")
with fc3:
    st.markdown("**Technology**")
    st.markdown("Python")
    st.markdown("Streamlit")
    st.markdown("Scikit-learn")
    st.markdown("Plotly")
with fc4:
    st.markdown("**Details**")
    st.markdown("Ames Housing Dataset")
    st.markdown("68% Within 10% Accuracy")
    st.markdown("MAE: $27,773")
    st.markdown("Model Version 1.0")

st.markdown("---")
fc_b1, fc_b2 = st.columns([1, 1])
with fc_b1:
    st.caption("2026 HomeSense AI. All rights reserved.")
with fc_b2:
    st.caption("Python | Scikit-learn | Streamlit | Plotly")
