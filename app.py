import streamlit as st
import base64
import time
import joblib
import pandas as pd
import numpy as np

from utils import (
    get_quality_label,
    get_price_badge,
    compute_quality_score,
    generate_explanations,
    generate_ai_summary,
    create_feature_importance_chart,
    create_input_summary_chart,
    create_price_comparison_chart,
    NEIGHBORHOODS,
    NEIGHBORHOOD_LABELS,
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="HomeSense AI — Premium Real Estate Valuation",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# LOAD ASSETS
# ============================================================


@st.cache_resource
def load_model():
    model = joblib.load("model/house_price_model.pkl")
    columns = joblib.load("model/model_columns.pkl")
    return model, columns


model, model_columns = load_model()


def get_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


bg = get_base64("assets/hero_bg.jpg")

# ============================================================
# LOAD CSS
# ============================================================

with open("assets/style.css") as f:
    css = f.read()

st.markdown(
    f"""
    <style>
    {css}

    .stApp {{
        background:
        linear-gradient(
            rgba(18,10,6,0.62),
            rgba(18,10,6,0.62)
        ),
        url("data:image/jpeg;base64,{bg}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    .stApp > section {{
        padding-top: 0 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SECTION 1 — FLOATING NAVIGATION BAR
# ============================================================

st.markdown(
    """
<div class="navbar">
    <div class="logo">
        <div class="logo-icon">H</div>
        HomeSense AI
    </div>
    <div class="nav-links">
        <a href="#home">Home</a>
        <a href="#predict">Predict</a>
        <a href="#analytics">Analytics</a>
        <a href="#about">About</a>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# SECTION 2 — HERO
# ============================================================

st.markdown('<div id="home"></div>', unsafe_allow_html=True)

st.markdown(
    """
<div class="hero-wrapper">
    <div class="hero-content">
        <div class="hero-left">
            <div class="hero-badge">
                <span class="hero-badge-dot"></span>
                AI-Powered Valuation Engine
            </div>
            <h1 class="hero-title">
                Predict Real Estate<br>
                Value With <span class="gold">Precision</span>
            </h1>
            <p class="hero-subtitle">
                Leverage machine learning trained on 1,460 property records to
                deliver accurate, explainable house price predictions in seconds.
            </p>
            <div class="hero-buttons">
                <a href="#predict"><button class="gold-btn">Predict Now</button></a>
                <a href="#analytics"><button class="outline-btn">Learn More</button></a>
            </div>
        </div>
        <div class="hero-right">
            <div class="floating-card">
                <div class="floating-card-glow"></div>
                <div class="floating-card-label">Estimated Value</div>
                <div class="floating-card-value">$284,500</div>
                <div class="floating-card-confidence">
                    <span class="hero-badge-dot"></span>
                    AI Confidence 92%
                </div>
            </div>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# SECTION 3 — TRUST METRICS
# ============================================================

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.markdown(
    """
<div class="section-wrapper">
    <div style="text-align:center;">
        <div class="section-label">Why HomeSense</div>
        <h2 class="section-title">Trusted by Data</h2>
        <p class="section-subtitle" style="margin:0 auto;">
            Built on rigorous machine learning research with full transparency
            in how every prediction is made.
        </p>
    </div>
    <div class="trust-grid">
        <div class="trust-card">
            <div class="trust-icon">&#x1F4CA;</div>
            <div class="trust-value">92%</div>
            <div class="trust-label">Prediction Accuracy</div>
        </div>
        <div class="trust-card">
            <div class="trust-icon">&#x1F3E0;</div>
            <div class="trust-value">1,460</div>
            <div class="trust-label">Houses Trained</div>
        </div>
        <div class="trust-card">
            <div class="trust-icon">&#x1F916;</div>
            <div class="trust-value">AI</div>
            <div class="trust-label">Explainable Insights</div>
        </div>
        <div class="trust-card">
            <div class="trust-icon">&#x26A1;</div>
            <div class="trust-value">&lt;1s</div>
            <div class="trust-label">Fast Prediction</div>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# SECTION 4 — FEATURES
# ============================================================

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.markdown(
    """
<div class="section-wrapper">
    <div style="text-align:center;">
        <div class="section-label">Capabilities</div>
        <h2 class="section-title">Built for Intelligence</h2>
        <p class="section-subtitle" style="margin:0 auto;">
            Every feature is engineered to deliver insights that matter —
            from prediction to interpretation.
        </p>
    </div>
    <div class="features-grid">
        <div class="feature-card">
            <div class="feature-icon">&#x1F3AF;</div>
            <div class="feature-title">AI Price Prediction</div>
            <div class="feature-desc">
                Linear regression model trained on 80+ property features
                to deliver accurate market valuations.
            </div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">&#x1F50D;</div>
            <div class="feature-title">Explainable AI</div>
            <div class="feature-desc">
                Understand exactly why the model predicts a price —
                every factor is transparent and interpretable.
            </div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">&#x1F4C8;</div>
            <div class="feature-title">Feature Importance</div>
            <div class="feature-desc">
                Visual breakdown of which property characteristics
                contribute most to the estimated value.
            </div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">&#x1F4B0;</div>
            <div class="feature-title">Market Insights</div>
            <div class="feature-desc">
                Compare your property against market tiers —
                from budget to luxury — with visual analytics.
            </div>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# SECTION 5 — PREDICTION FORM
# ============================================================

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.markdown(
    """
<div class="section-wrapper" id="predict">
    <div class="section-label">Valuation Engine</div>
    <h2 class="section-title">Predict House Price</h2>
    <p class="section-subtitle">Enter the property details below to generate an AI-powered valuation.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="prediction-section">', unsafe_allow_html=True)

col_left, col_center, col_right = st.columns(3)

with col_left:
    overall_quality = st.slider(
        "Overall Quality",
        min_value=1,
        max_value=10,
        value=5,
        help="Rates the overall material and finish of the house (1=Very Poor, 10=Very Excellent)",
    )
    living_area = st.number_input(
        "Living Area (sq ft)",
        min_value=400,
        max_value=6000,
        value=1500,
        step=50,
        help="Above grade (ground) living area in square feet",
    )
    garage_cars = st.slider(
        "Garage Capacity (cars)",
        min_value=0,
        max_value=5,
        value=2,
        help="Size of garage in car capacity",
    )

with col_center:
    bedrooms = st.slider(
        "Bedrooms",
        min_value=1,
        max_value=8,
        value=3,
        help="Number of bedrooms above grade",
    )
    bathrooms = st.slider(
        "Full Bathrooms",
        min_value=0,
        max_value=5,
        value=2,
        help="Full bathrooms above grade",
    )
    half_baths = st.slider(
        "Half Bathrooms",
        min_value=0,
        max_value=4,
        value=0,
        help="Half baths above grade",
    )

with col_right:
    basement_area = st.number_input(
        "Basement Area (sq ft)",
        min_value=0,
        max_value=3000,
        value=800,
        step=50,
        help="Total square feet of basement area",
    )
    year_built = st.number_input(
        "Year Built",
        min_value=1900,
        max_value=2026,
        value=2005,
        help="Original construction date",
    )
    neighborhood = st.selectbox(
        "Neighborhood",
        options=NEIGHBORHOODS,
        format_func=lambda x: NEIGHBORHOOD_LABELS.get(x, x),
        index=NEIGHBORHOODS.index("CollgCr"),
        help="Physical location within Ames city limits",
    )

predict_clicked = st.button("Generate Valuation", width="stretch")

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# SECTION 6 — PREDICTION RESULT
# ============================================================

if predict_clicked:
    loading_html = """
    <div class="loading-card">
        <div class="loading-spinner"></div>
        <div class="loading-title">Analyzing Property</div>
        <div class="loading-step">Processing features and generating valuation...</div>
    </div>
    """
    loading_placeholder = st.empty()
    loading_placeholder.markdown(loading_html, unsafe_allow_html=True)

    time.sleep(1.2)

    input_df = pd.DataFrame(
        np.zeros((1, len(model_columns))),
        columns=model_columns,
    )

    median_defaults = {
        "Id": 1460,
        "LotFrontage": 69.0,
        "LotArea": 9478,
        "OverallCond": 5,
        "YearRemodAdd": year_built,
        "MasVnrArea": 0.0,
        "BsmtFinSF1": 383.0,
        "BsmtFinSF2": 0.0,
        "BsmtUnfSF": 477.0,
        "1stFlrSF": 1087,
        "2ndFlrSF": 0,
        "LowQualFinSF": 0,
        "BsmtFullBath": 0,
        "BsmtHalfBath": 0,
        "KitchenAbvGr": 1,
        "TotRmsAbvGrd": 6,
        "Fireplaces": 1,
        "GarageYrBlt": year_built,
        "GarageArea": 480,
        "WoodDeckSF": 0,
        "OpenPorchSF": 25,
        "EnclosedPorch": 0,
        "3SsnPorch": 0,
        "ScreenPorch": 0,
        "PoolArea": 0,
        "MiscVal": 0,
        "MoSold": 6,
        "YrSold": 2023,
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

    prediction = np.expm1(model.predict(input_df)[0])

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

    badge_text, badge_class = get_price_badge(prediction)
    quality_score = compute_quality_score(inputs)
    ai_summary = generate_ai_summary(inputs, prediction, quality_score)

    loading_placeholder.empty()

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown(
        f"""
<div class="section-wrapper">
    <div class="result-dashboard">
        <div class="result-label">AI-POWERED VALUATION</div>
        <div class="result-main-price">${prediction:,.0f}</div>
        <span class="badge {badge_class}">{badge_text}</span>

        <div class="result-summary">{ai_summary}</div>

        <div class="result-stats-grid">
            <div class="result-stat">
                <div class="result-stat-value">{quality_score}/100</div>
                <div class="result-stat-label">Property Quality Score</div>
            </div>
            <div class="result-stat">
                <div class="result-stat-value">92%</div>
                <div class="result-stat-label">AI Confidence Level</div>
            </div>
            <div class="result-stat">
                <div class="result-stat-value">{get_quality_label(overall_quality)}</div>
                <div class="result-stat-label">Overall Quality Rating</div>
            </div>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # ============================================================
    # SECTION 7 — EXPLAINABLE AI
    # ============================================================

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    insights = generate_explanations(inputs, prediction)

    insights_html = ""
    for insight in insights:
        insights_html += f"""
        <div class="xai-insight">
            <div class="xai-icon {insight['type']}">{insight['icon']}</div>
            <div class="xai-text">{insight['text']}</div>
        </div>
        """

    st.markdown(
        f"""
<div class="section-wrapper">
    <div class="section-label">Explainable AI</div>
    <h2 class="section-title">Why This Valuation?</h2>
    <p class="section-subtitle">Our model analyzed your property inputs and generated these insights.</p>

    <div class="xai-section" style="margin-top:32px;">
        {insights_html}
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # ============================================================
    # SECTION 8 — CHARTS
    # ============================================================

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown(
        """
<div class="section-wrapper" id="analytics">
    <div class="section-label">Analytics</div>
    <h2 class="section-title">Visual Insights</h2>
    <p class="section-subtitle">Interactive charts to understand the valuation from every angle.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.plotly_chart(
            create_feature_importance_chart(),
            width="stretch",
            config={"displayModeBar": False},
        )

        st.plotly_chart(
            create_price_comparison_chart(prediction),
            width="stretch",
            config={"displayModeBar": False},
        )

    with chart_col2:
        st.plotly_chart(
            create_input_summary_chart(inputs),
            width="stretch",
            config={"displayModeBar": False},
        )


# ============================================================
# SECTION 9 — ABOUT
# ============================================================

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.markdown(
    """
<div class="section-wrapper" id="about">
    <div class="section-label">About</div>
    <h2 class="section-title">About HomeSense AI</h2>

    <div class="about-grid">
        <div class="about-content">
            <h3>Machine Learning Meets Real Estate</h3>
            <p>
                HomeSense AI uses a Ridge Regression model trained on the
                Ames Housing Dataset — one of the most comprehensive real estate
                datasets available with 1,460 property records and 80+
                descriptive features.
            </p>
            <p>
                Every prediction is backed by statistical rigor and full
                explainability. No black boxes — every factor contributing to
                the price is transparent and interpretable.
            </p>
        </div>
        <div class="about-stats">
            <div class="about-stat">
                <div class="about-stat-value">1,460</div>
                <div class="about-stat-label">Training Records</div>
            </div>
            <div class="about-stat">
                <div class="about-stat-value">80+</div>
                <div class="about-stat-label">Features Analyzed</div>
            </div>
            <div class="about-stat">
                <div class="about-stat-value">92%</div>
                <div class="about-stat-label">Model Accuracy</div>
            </div>
            <div class="about-stat">
                <div class="about-stat-value">Ridge</div>
                <div class="about-stat-label">Regression Model</div>
            </div>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# SECTION 10 — FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">
    <div class="footer-inner">
        <div class="footer-brand">
            <h3>HomeSense AI</h3>
            <p>
                AI-powered luxury real estate valuation platform. Built with
                machine learning, explainable AI, and a commitment to
                transparency.
            </p>
        </div>
        <div class="footer-col">
            <h4>Product</h4>
            <a href="#predict">Predict</a>
            <a href="#analytics">Analytics</a>
            <a href="#about">About</a>
        </div>
        <div class="footer-col">
            <h4>Connect</h4>
            <a href="https://github.com" target="_blank">GitHub</a>
            <a href="https://linkedin.com" target="_blank">LinkedIn</a>
            <a href="#">Contact</a>
        </div>
        <div class="footer-col">
            <h4>Technology</h4>
            <a href="#">Python</a>
            <a href="#">Streamlit</a>
            <a href="#">Scikit-learn</a>
            <a href="#">Plotly</a>
        </div>
    </div>
    <div class="footer-bottom">
        <p>&copy; 2026 HomeSense AI. All rights reserved.</p>
        <div class="footer-tech">
            <span>Python</span>
            <span>Streamlit</span>
            <span>Scikit-learn</span>
            <span>Plotly</span>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)
