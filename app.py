import os
import random
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

# Load hero_bg.jpg as base64 for CSS background
def _load_bg_b64():
    try:
        with open("assets/hero_bg.jpg", "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""
_bg_b64 = _load_bg_b64()
_bg_css = f"background-image:url(data:image/jpeg;base64,{_bg_b64});" if _bg_b64 else ""

st.markdown(
    f"""<style>{css}
/* ===== PAGE BACKGROUND (hero_bg.jpg) ===== */
#root > div:first-child{{{_bg_css}background-size:cover;background-attachment:fixed;background-position:center;background-repeat:no-repeat;background-color:#120A06;}}
[data-testid="stApp"]{{{_bg_css}background-size:cover !important;background-attachment:fixed !important;background-position:center !important;background-repeat:no-repeat !important;background-color:#120A06 !important;}}
[data-testid="stAppViewBlockContainer"]{{background:transparent !important;}}
[data-testid="stMain"]{{background:transparent !important;}}
[data-testid="stMainBlockContainer"]{{background:transparent !important;}}
[data-testid="stHeader"]{{background:transparent !important;}}
[data-testid="stToolbar"]{{display:none !important;}}
#MainMenu{{visibility:hidden !important;}}
footer{{visibility:hidden !important;}}
[data-testid="stSidebar"]{{background:#120A06 !important;}}
.block-container{{padding-top:0 !important;padding-bottom:0 !important;background:transparent !important;max-width:1450px !important;margin:0 auto !important;padding-left:32px !important;padding-right:32px !important;}}

/* ===== CINEMATIC HERO ===== */
.hero-cinema{{position:relative;width:100%;min-height:100vh;overflow:hidden;display:flex;align-items:center;justify-content:center;border-radius:0 0 32px 32px;}}
.hero-slide{{position:absolute;inset:0;opacity:0;transition:opacity 0.6s ease-in-out;background-size:cover;background-position:center;animation:kenBurns 20s ease-in-out infinite alternate;}}
.hero-slide.active{{opacity:1;}}
@keyframes kenBurns{{from{{transform:scale(1);}}to{{transform:scale(1.08);}}}}
.hero-overlay{{position:absolute;inset:0;z-index:1;}}
.hero-content{{position:relative;z-index:2;text-align:center;max-width:1100px;padding:0 40px;}}
.hero-badge{{display:inline-flex;align-items:center;gap:8px;padding:8px 20px;background:rgba(215,162,78,0.10);border:1px solid rgba(215,162,78,0.30);border-radius:100px;color:#D7A24E;font-size:12px;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:28px;}}
.hero-badge-dot{{width:7px;height:7px;background:#D7A24E;border-radius:50%;animation:pulse 2s ease-in-out infinite;}}
.hero-title{{font-family:'Playfair Display',serif;font-size:clamp(32px,5.5vw,72px);line-height:1.05;color:#FFFFFF;font-weight:700;letter-spacing:-2px;margin:0 0 24px 0;max-width:760px;margin-left:auto;margin-right:auto;}}
.hero-title-gold{{color:#D7A24E;}}
.hero-subtitle{{font-size:18px;line-height:1.8;color:#F0ECE8;max-width:650px;margin:0 auto 36px;}}
.hero-stats{{display:flex;justify-content:center;gap:20px;flex-wrap:nowrap;margin-bottom:40px;}}
.hero-stat{{text-align:center;flex:1;min-width:0;max-width:320px;padding:36px;background:rgba(38,26,20,0.65);backdrop-filter:blur(40px) saturate(180%);-webkit-backdrop-filter:blur(40px) saturate(180%);border:1px solid rgba(215,162,78,0.15);border-radius:24px;box-shadow:0 10px 36px rgba(0,0,0,0.30),inset 0 1px 0 rgba(215,162,78,0.06);}}
.hero-stat-val{{font-family:'Playfair Display',serif;font-size:56px;color:#D7A24E;font-weight:700;}}
.hero-stat-label{{color:#B5AFA8;font-size:16px;margin-top:10px;letter-spacing:2px;text-transform:uppercase;font-weight:600;}}
.hero-stat-divider{{display:none;}}
.hero-pred-price{{font-family:'Playfair Display',serif;font-size:clamp(40px,7vw,54px);color:#D7A24E;font-weight:700;letter-spacing:-2px;margin:16px 0 8px 0;line-height:1;}}
.hero-cta{{display:inline-block;padding:16px 44px;background:linear-gradient(135deg,#D7A24E,#C4942A,#B8862A);color:#120A06 !important;font-weight:700;font-size:16px;border-radius:14px;text-decoration:none !important;box-shadow:0 4px 20px rgba(215,162,78,0.20),0 0 0 1px rgba(215,162,78,0.15) inset;transition:all 0.3s cubic-bezier(0.4,0,0.2,1);letter-spacing:0.3px;}}
.hero-cta:hover{{transform:translateY(-3px);box-shadow:0 8px 32px rgba(215,162,78,0.35),0 0 40px rgba(215,162,78,0.12),0 0 0 1px rgba(215,162,78,0.30) inset;}}
.hero-content a{{color:#120A06 !important;}}

/* ===== GLOBAL CONTAINER ===== */
.main-container{{max-width:1450px;margin:0 auto;padding:0 32px;}}

/* ===== SECTION TITLE SYSTEM ===== */
.section-badge{{display:inline-flex;align-items:center;gap:8px;padding:6px 16px;background:rgba(215,162,78,0.08);border:1px solid rgba(215,162,78,0.18);border-radius:100px;color:#D7A24E;font-size:11px;font-weight:700;letter-spacing:2px;text-transform:uppercase;}}
.section-title{{font-family:'Playfair Display',serif;font-size:34px;color:#FFFFFF;font-weight:700;text-align:center;letter-spacing:-1px;}}
.section-desc{{font-size:18px;color:#B5AFA8;text-align:center;max-width:540px;margin:0 auto 32px;}}

/* ===== TYPOGRAPHY ===== */
[data-testid="stMarkdownContainer"] p{{color:#F0ECE8;line-height:1.8;font-size:18px;}}
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4{{color:#FFFFFF !important;font-family:'Playfair Display',serif !important;letter-spacing:-0.3px;}}
.stSubheader{{color:#FFFFFF !important;font-family:'Playfair Display',serif !important;font-weight:700 !important;font-size:34px !important;letter-spacing:-0.3px;}}
[data-testid="stCaptionContainer"]{{color:#D7A24E !important;font-size:11px !important;text-transform:uppercase !important;letter-spacing:2.5px !important;font-weight:700 !important;}}

/* ===== METRIC CARDS (GLASSMORPHISM) ===== */
[data-testid="stMetric"]{{
  background:rgba(38,26,20,0.65) !important;
  border:1px solid rgba(215,162,78,0.15) !important;
  border-radius:24px !important;padding:36px !important;
  backdrop-filter:blur(40px) saturate(180%) !important;
  -webkit-backdrop-filter:blur(40px) saturate(180%) !important;
  box-shadow:0 8px 32px rgba(0,0,0,0.30),inset 0 1px 0 rgba(215,162,78,0.08) !important;
  transition:all 0.4s cubic-bezier(0.4,0,0.2,1) !important;
}}
[data-testid="stMetric"]:hover{{
  transform:translateY(-4px) !important;
  border-color:rgba(215,162,78,0.30) !important;
  box-shadow:0 16px 48px rgba(0,0,0,0.40),0 0 24px rgba(215,162,78,0.06),inset 0 1px 0 rgba(215,162,78,0.10) !important;
}}
[data-testid="stMetric"] label{{color:#B5AFA8 !important;font-size:15px !important;text-transform:none;letter-spacing:0;font-weight:600;}}
[data-testid="stMetric"] [data-testid="stMetricValue"]{{color:#D7A24E !important;font-weight:700 !important;font-size:50px !important;text-shadow:0 0 20px rgba(215,162,78,0.12);}}
[data-testid="stMetric"] [data-testid="stMetricDelta"]{{font-weight:600;}}

/* ===== HORIZONTAL BLOCKS (COLUMNS) ===== */
div[data-testid="stHorizontalBlock"]{{gap:16px !important;padding:0 !important;}}

/* ===== SECTION CONTAINERS (GLASSMORPHISM) ===== */
[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"]{{
  background:rgba(38,26,20,0.65) !important;
  backdrop-filter:blur(40px) saturate(180%) !important;
  -webkit-backdrop-filter:blur(40px) saturate(180%) !important;
  border:1px solid rgba(215,162,78,0.10) !important;
  border-radius:24px !important;
  padding:36px !important;
  box-shadow:0 10px 36px rgba(0,0,0,0.25),inset 0 1px 0 rgba(215,162,78,0.05) !important;
  margin-bottom:8px !important;
  transition:all 0.4s cubic-bezier(0.4,0,0.2,1) !important;
}}
[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"]:hover{{
  border-color:rgba(215,162,78,0.18) !important;
  box-shadow:0 16px 48px rgba(0,0,0,0.30),0 0 20px rgba(215,162,78,0.04),inset 0 1px 0 rgba(215,162,78,0.06) !important;
}}

/* ===== EXPANDER ===== */
.stExpander{{background:rgba(38,26,20,0.65) !important;border:1px solid rgba(215,162,78,0.10) !important;border-radius:24px !important;backdrop-filter:blur(40px) !important;-webkit-backdrop-filter:blur(40px) !important;padding:24px !important;}}
.stExpander summary{{color:#FFFFFF !important;font-size:18px !important;}}

/* ===== ALERTS ===== */
.stAlert{{background:rgba(38,26,20,0.65) !important;border:1px solid rgba(215,162,78,0.12) !important;border-radius:24px !important;backdrop-filter:blur(32px) !important;-webkit-backdrop-filter:blur(32px) !important;}}

/* ===== BUTTONS (PREMIUM GOLD) ===== */
.stButton > button, .stDownloadButton > button{{
  background:linear-gradient(135deg,#D7A24E,#C4942A,#B8862A) !important;color:#120A06 !important;border:none !important;border-radius:14px !important;font-weight:700 !important;font-size:18px !important;
  box-shadow:0 4px 24px rgba(215,162,78,0.35),0 0 0 1px rgba(215,162,78,0.30) inset,0 1px 0 rgba(255,255,255,0.10) inset !important;
  transition:all 0.3s cubic-bezier(0.4,0,0.2,1) !important;letter-spacing:0.3px;
}}
.stButton > button:hover, .stDownloadButton > button:hover{{
  background:linear-gradient(135deg,#E8C060,#D7A24E,#C4942A) !important;transform:translateY(-2px) !important;
  box-shadow:0 8px 36px rgba(215,162,78,0.50),0 0 40px rgba(215,162,78,0.15),0 0 0 1px rgba(215,162,78,0.45) inset !important;
}}
.stButton > button:active{{transform:translateY(0) !important;}}

/* ===== WIDGETS ===== */
.stSlider > div > div > div > div, .stSlider > div > div > div > div > div {{background-color:#D7A24E !important;border-color:#D7A24E !important;}}
.stNumberInput > div > div > input {{background:rgba(38,26,20,0.65) !important;border:1px solid rgba(215,162,78,0.10) !important;border-radius:14px !important;color:#FFFFFF !important;padding:14px 18px !important;font-size:18px !important;}}
.stNumberInput > div > div > input:focus {{border-color:rgba(215,162,78,0.4) !important;box-shadow:0 0 0 2px rgba(215,162,78,0.10) !important;}}
.stSelectbox > div > div {{background:rgba(38,26,20,0.65) !important;border:1px solid rgba(215,162,78,0.10) !important;border-radius:14px !important;color:#FFFFFF !important;}}
.stProgress > div > div > div{{background:linear-gradient(90deg,#C4942A,#D7A24E,#E8C060);}}

/* ===== SECTION DIVIDER ===== */
.section-divider{{max-width:1450px;margin:110px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(215,162,78,0.20),transparent);}}

/* ===== PREMIUM CARD ===== */
.premium-card{{background:rgba(38,26,20,0.65);backdrop-filter:blur(40px) saturate(180%);-webkit-backdrop-filter:blur(40px) saturate(180%);border:1px solid rgba(215,162,78,0.15);border-radius:24px;padding:36px;box-shadow:0 10px 36px rgba(0,0,0,0.30),inset 0 1px 0 rgba(215,162,78,0.06);transition:all 0.4s cubic-bezier(0.4,0,0.2,1);}}
.premium-card:hover{{transform:translateY(-4px);border-color:rgba(215,162,78,0.30);box-shadow:0 20px 56px rgba(0,0,0,0.40),0 0 28px rgba(215,162,78,0.06),inset 0 1px 0 rgba(215,162,78,0.10);}}
.card-title{{color:#D7A24E;font-size:24px;font-weight:600;margin-bottom:8px;}}
.card-content{{color:#F0ECE8;font-size:18px;line-height:1.8;}}
.card-label{{color:#B5AFA8;font-size:15px;}}

/* ===== PREDICTION FORM ===== */
.pred-form{{max-width:1450px;margin:0 auto;}}
.pred-form label{{color:#F0ECE8 !important;font-size:15px !important;font-weight:600 !important;}}

/* ===== PROPERTY IMAGE ===== */
.prop-image{{border-radius:24px;overflow:hidden;box-shadow:0 12px 40px rgba(0,0,0,0.40),0 0 0 1px rgba(215,162,78,0.15) inset;height:520px;}}
.prop-image img{{width:100%;height:100%;object-fit:cover;}}

/* ===== ESTIMATED VALUE ===== */
.est-price{{font-family:'Playfair Display',serif;font-size:54px;color:#D7A24E;font-weight:700;letter-spacing:-2px;}}
.est-confidence{{color:#4CAF50;font-size:18px;font-weight:600;}}

/* ===== EXECUTIVE SUMMARY ===== */
.exec-summary{{max-width:900px;margin:0 auto;padding:36px;}}
.exec-summary p{{font-size:18px;line-height:1.8;color:#F0ECE8;}}

/* ===== INVESTMENT SCORE ===== */
.invest-rating{{font-family:'Playfair Display',serif;font-size:54px;color:#D7A24E;font-weight:700;}}
.invest-stars{{font-size:34px;color:#D7A24E;}}

/* ===== 2-COLUMN FACTS ===== */
.facts-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px 24px;}}
.fact-item{{display:flex;justify-content:space-between;padding:16px 20px;background:rgba(38,26,20,0.50);border-radius:24px;border:1px solid rgba(215,162,78,0.08);}}
.fact-label{{color:#B5AFA8;font-size:15px;}}
.fact-value{{color:#FFFFFF;font-weight:600;font-size:18px;}}

/* ===== HIGHLIGHT CARDS ===== */
.highlight-card{{padding:20px 24px;background:rgba(38,26,20,0.50);border-radius:24px;border:1px solid rgba(215,162,78,0.10);margin-bottom:12px;}}
.highlight-card-title{{color:#FFFFFF;font-weight:600;font-size:20px;margin-bottom:6px;}}
.highlight-card-desc{{color:#B5AFA8;font-size:18px;line-height:1.8;}}

/* ===== DRIVE TABLE ===== */
.drive-row{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;padding:16px 24px;background:rgba(38,26,20,0.50);border-radius:24px;border:1px solid rgba(215,162,78,0.08);margin-bottom:10px;}}
.drive-factor{{color:#FFFFFF;font-weight:600;font-size:18px;}}
.drive-contrib{{color:#B5AFA8;font-size:18px;}}
.drive-impact{{font-weight:700;font-size:18px;}}

/* ===== ASSESSMENT CARDS ===== */
.assess-card{{text-align:center;padding:24px;background:rgba(38,26,20,0.50);border-radius:24px;border:1px solid rgba(215,162,78,0.10);}}
.assess-label{{color:#B5AFA8;font-size:15px;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;}}
.assess-value{{color:#FFFFFF;font-weight:700;font-size:24px;}}

/* ===== MARKET BENCHMARK ===== */
.benchmark-card{{padding:36px;background:rgba(38,26,20,0.50);border-radius:24px;border:1px solid rgba(215,162,78,0.10);text-align:center;}}

/* ===== SIMILAR PROPERTY IMAGE ===== */
.sim-img{{width:280px;height:180px;border-radius:24px;overflow:hidden;box-shadow:0 8px 24px rgba(0,0,0,0.30);}}
.sim-img img{{width:100%;height:100%;object-fit:cover;}}

/* ===== FORECAST CARDS ===== */
.forecast-card{{text-align:center;padding:36px;background:rgba(38,26,20,0.50);border-radius:24px;border:1px solid rgba(215,162,78,0.10);}}
.forecast-year{{color:#B5AFA8;font-size:15px;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;}}
.forecast-val{{font-family:'Playfair Display',serif;font-size:34px;color:#D7A24E;font-weight:700;}}

/* ===== RECOMMENDATION CARDS ===== */
.rec-card{{padding:20px 24px;background:rgba(38,26,20,0.50);border-radius:24px;border:1px solid rgba(215,162,78,0.10);margin-bottom:10px;border-left:4px solid;}}
.rec-card-green{{border-left-color:#4CAF50;}}
.rec-card-gold{{border-left-color:#D7A24E;}}
.rec-card-red{{border-left-color:#E57373;}}
.rec-card-blue{{border-left-color:#5B8DEF;}}

/* ===== SCROLL ANIMATION ===== */
.fade-in-section{{opacity:0;transform:translateY(20px);transition:opacity 0.6s ease-out,transform 0.6s ease-out;}}
.fade-in-section.visible{{opacity:1;transform:translateY(0);}}

/* ===== TEXT UTILITIES ===== */
.gold-text{{color:#D7A24E;}}.green-text{{color:#4CAF50;}}.red-text{{color:#E57373;}}.amber-text{{color:#FFD060;}}.muted-text{{color:#B5AFA8;}}.primary-text{{color:#FFFFFF;}}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar{{width:5px;}}::-webkit-scrollbar-track{{background:#120A06;}}::-webkit-scrollbar-thumb{{background:#D7A24E;border-radius:10px;}}

/* ===== ANIMATIONS ===== */
@keyframes fadeInUp{{from{{opacity:0;transform:translateY(20px);}}to{{opacity:1;transform:translateY(0);}}}}
@keyframes fadeIn{{from{{opacity:0;}}to{{opacity:1;}}}}
@keyframes pulse{{0%,100%{{opacity:1;transform:scale(1);}}50%{{opacity:0.4;transform:scale(1.4);}}}}
@keyframes slideIn{{from{{opacity:0;transform:translateY(30px);}}to{{opacity:1;transform:translateY(0);}}}}

/* ===== LOADING ANIMATION ===== */
.loading-bar{{height:4px;background:rgba(215,162,78,0.2);border-radius:4px;overflow:hidden;margin:8px 0;}}
.loading-bar-fill{{height:100%;background:linear-gradient(90deg,#C4942A,#D7A24E,#E8C060);border-radius:4px;transition:width 0.6s ease;}}

/* ===== RESPONSIVE: TABLET (max-width 900px) ===== */
@media(max-width:900px){{
  .nav-responsive{{padding:12px 24px !important;}}
  .nav-links{{gap:20px !important;font-size:13px !important;}}
  .hero-content{{padding:0 24px;max-width:100%;}}
  .hero-title{{letter-spacing:-1px;}}
  .hero-stats{{flex-wrap:wrap;gap:16px;}}
  .hero-stat{{max-width:100%;flex:0 0 calc(50% - 8px);}}
  [data-testid="stMetric"]{{padding:24px !important;border-radius:20px !important;}}
  [data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"]{{padding:28px !important;border-radius:20px !important;}}
  .premium-card{{padding:28px !important;border-radius:20px !important;}}
  .facts-grid{{grid-template-columns:1fr;}}
  .section-divider{{margin:80px auto !important;}}
  .assess-grid{{grid-template-columns:repeat(2,1fr) !important;}}
  .appr-grid{{grid-template-columns:repeat(2,1fr) !important;}}
  .footer-grid{{display:flex !important;flex-direction:column !important;}}
  .comparison-row{{flex-wrap:wrap !important;gap:8px !important;}}
  .comparison-row > span{{min-width:auto !important;font-size:14px !important;}}
  [style*="margin:0 -60px"]{{margin:0 -24px !important;padding:48px 24px 28px !important;}}
}}

/* ===== RESPONSIVE: MOBILE (max-width 640px) ===== */
@media(max-width:640px){{
  .nav-responsive{{padding:10px 16px !important;width:min(96%,1200px) !important;}}
  .nav-links{{gap:14px !important;font-size:12px !important;}}
  .hero-cinema{{min-height:100svh;border-radius:0 0 20px 20px;}}
  .hero-content{{padding:0 16px;}}
  .hero-badge{{font-size:10px;padding:6px 14px;margin-bottom:20px;}}
  .hero-title{{font-size:clamp(26px,7vw,36px);letter-spacing:-0.5px;margin-bottom:16px;}}
  .hero-subtitle{{font-size:16px;line-height:1.8;margin-bottom:24px;}}
  .hero-stats{{flex-direction:column;gap:12px;}}
  .hero-stat{{max-width:100%;padding:24px 20px;}}
  .hero-stat-val{{font-size:40px;}}
  .hero-stat-label{{font-size:14px;}}
  .hero-cta{{padding:12px 28px;font-size:14px;}}
  .hero-pred-price{{font-size:clamp(32px,10vw,48px);}}
  [data-testid="stMetric"]{{padding:20px !important;border-radius:24px !important;}}
  [data-testid="stMetric"] label{{font-size:13px !important;}}
  [data-testid="stMetric"] [data-testid="stMetricValue"]{{font-size:34px !important;}}
  [data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"]{{padding:20px !important;border-radius:24px !important;}}
  .premium-card{{padding:20px !important;border-radius:24px !important;}}
  div[data-testid="stHorizontalBlock"]{{gap:10px !important;}}
  .section-divider{{margin:60px auto !important;}}
  .prop-image{{height:300px;}}
  .est-price{{font-size:34px;}}
  .invest-rating{{font-size:34px;}}
  .forecast-val{{font-size:26px !important;}}
  .drive-row{{grid-template-columns:1fr;gap:8px;}}
  .sim-img{{width:100%;height:160px;}}
  .card-title{{font-size:20px;}}
  .card-content{{font-size:15px;}}
  .fact-item{{padding:12px 16px;}}
  .fact-label{{font-size:13px;}}
  .fact-value{{font-size:15px;}}
  .assess-grid{{grid-template-columns:1fr !important;}}
  .appr-grid{{grid-template-columns:1fr !important;}}
  .footer-grid{{grid-template-columns:1fr !important;}}
  .comparison-row{{flex-direction:column !important;align-items:flex-start !important;gap:6px !important;}}
  .comparison-row > span{{min-width:auto !important;font-size:13px !important;}}
  [style*="margin:0 -60px"]{{margin:0 -16px !important;padding:40px 16px 24px !important;}}
}}

/* ===== RESPONSIVE: VERY SMALL (max-width 400px) ===== */
@media(max-width:400px){{
  .nav-responsive{{padding:8px 12px !important;}}
  .nav-links{{gap:10px !important;font-size:11px !important;}}
  .hero-title{{font-size:22px;}}
  .hero-badge{{font-size:9px;padding:5px 10px;}}
  .hero-stat-val{{font-size:32px;}}
  [data-testid="stMetric"]{{padding:16px !important;}}
  [data-testid="stMetric"] [data-testid="stMetricValue"]{{font-size:28px !important;}}
}}
</style>""",
    unsafe_allow_html=True,
)


# ============================================================
# NAVBAR
# ============================================================
st.markdown(
    '<div class="nav-responsive" style="position:fixed;top:16px;left:50%;transform:translateX(-50%);width:min(92%,1200px);display:flex;justify-content:space-between;align-items:center;padding:14px 36px;background:rgba(13,7,5,0.75);backdrop-filter:blur(28px) saturate(180%);border:1px solid rgba(215,162,78,0.15);border-radius:20px;z-index:9999;box-shadow:0 8px 36px rgba(0,0,0,0.5),inset 0 1px 0 rgba(215,162,78,0.06);">'
    '<span style="font-family:Playfair Display,serif;font-size:clamp(18px,2.5vw,24px);color:#D7A24E;font-weight:700;">HomeSense AI</span>'
    '<span class="nav-links" style="display:flex;gap:32px;color:rgba(248,245,240,0.65);font-size:14px;font-weight:500;">'
    '<a href="#home" style="color:inherit;text-decoration:none;">Home</a> '
    '<a href="#predict" style="color:inherit;text-decoration:none;">Predict</a> '
    '<a href="#analytics" style="color:inherit;text-decoration:none;">Analytics</a> '
    '<a href="#about" style="color:inherit;text-decoration:none;">About</a>'
    '</span></div>',
    unsafe_allow_html=True,
)

# ============================================================
# HERO — CINEMATIC WITH ROTATING HOUSE IMAGES
# ============================================================
st.markdown('<div id="home"></div>', unsafe_allow_html=True)

# ── Load ALL house images per category (precomputed once) ──
HOUSE_CATS = ["starter","cottage","ranch","townhouse","suburban",
              "craftsman","colonial","modern","luxury","mansion"]
HOUSE_DIR = "assets/houses"

def _load_b64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""

# Precompute: ALL images per category, not just first
cat_images_all = {}   # {cat: [(b64, fname), ...]}
cat_images = {}       # {cat: (b64, fname)}  first image (for fallback)
for cat in HOUSE_CATS:
    cat_dir = os.path.join(HOUSE_DIR, cat)
    if not os.path.isdir(cat_dir):
        continue
    imgs = []
    for fname in sorted(os.listdir(cat_dir)):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            b64 = _load_b64(os.path.join(cat_dir, fname))
            if b64:
                imgs.append((b64, fname))
    if imgs:
        cat_images_all[cat] = imgs
        cat_images[cat] = imgs[0]

def _select_hero_image(category, inputs=None):
    """Deterministic image selection: pick random image from matching category."""
    if category in cat_images_all and cat_images_all[category]:
        imgs = cat_images_all[category]
        # Use hash of inputs for deterministic but varied selection
        if inputs:
            seed = hash(tuple(sorted(inputs.items())))
            idx = seed % len(imgs)
        else:
            idx = random.randint(0, len(imgs) - 1)
        return imgs[idx][0], category
    # Fallback: pick from any available category
    for fallback in HOUSE_CATS:
        if fallback in cat_images_all and cat_images_all[fallback]:
            return cat_images_all[fallback][0][0], fallback
    return "", "suburban"

def _get_hero_category_from_inputs(inputs):
    """Deterministic rule-based category selection from property inputs."""
    oq = inputs.get("OverallQual", 5)
    gla = inputs.get("GrLivArea", 1500)
    gc = inputs.get("GarageCars", 2)
    yb = inputs.get("YearBuilt", 2005)
    beds = inputs.get("BedroomAbvGr", 3)
    baths = inputs.get("FullBath", 2)
    lot = inputs.get("LotArea", 9478)

    # Mansion: highest tier
    if beds >= 5 and baths >= 4 and gc >= 3 and gla >= 3500:
        return "mansion"
    # Luxury: quality 9+, large home
    if oq >= 9 and gla > 2500 and gc >= 3:
        return "luxury"
    # Modern: recent build, high quality
    if yb >= 2005 and oq >= 8:
        return "modern"
    # Colonial: older, traditional, large
    if yb < 1980 and oq >= 6 and gla > 1800 and beds >= 4:
        return "colonial"
    # Craftsman: mid-century, quality
    if 1940 <= yb < 1970 and oq >= 6 and gla > 1500 and beds <= 3:
        return "craftsman"
    # Suburban: family home, 2-car garage
    if beds >= 3 and gc >= 2 and oq >= 6 and gla >= 1200:
        return "suburban"
    # Ranch: single floor, moderate
    if yb < 1970 and gla >= 1200 and beds <= 3 and oq >= 5:
        return "ranch"
    # Townhouse: smaller, recent, limited garage
    if yb >= 1970 and gla < 1800 and gla >= 1100 and gc <= 2 and beds <= 3:
        return "townhouse"
    # Cottage: small, older
    if oq <= 5 and gla < 1200:
        return "cottage"
    # Starter: basic
    if oq <= 4 or gla < 1000:
        return "starter"
    return "suburban"

# ── Determine hero state: prediction-driven or rotating showcase ──
_prediction_made = 'prediction' in dir() and prediction is not None and np.isfinite(prediction)
_pred_image_b64 = ""
_pred_value = 0
_pred_category = ""
_pred_label = ""

if _prediction_made:
    _pred_category = profile.get("category", "suburban") if profile else "suburban"
    _pred_label = profile.get("label", "Property") if profile else "Property"
    _pred_value = float(prediction)
    _pred_image_b64, _pred_category = _select_hero_image(_pred_category, inputs)

# ── Build showcase images (one from each category, shuffled) ──
showcase_items = [(cat, imgs[0]) for cat, imgs in cat_images_all.items() if imgs]
random.shuffle(showcase_items)
showcase_b64s = [item[1][0] for item in showcase_items if item[1]]
if not showcase_b64s and _pred_image_b64:
    showcase_b64s = [_pred_image_b64]

# ── Overlay strength adapts per category brightness ──
LIGHT_CATS  = {"starter", "cottage"}
MEDIUM_CATS = {"ranch", "townhouse", "suburban"}
DARK_CATS   = {"modern", "luxury", "mansion", "colonial", "craftsman"}

def _overlay_for(cat):
    if cat in LIGHT_CATS:
        return "0.40","0.55","0.75","0.88"
    if cat in DARK_CATS:
        return "0.25","0.40","0.60","0.85"
    return "0.30","0.45","0.65","0.85"

# ── Build hero HTML ──
if _prediction_made:
    # Prediction-driven: transition from rotating to prediction image
    o_top, o_mid, o_low, o_bot = _overlay_for(_pred_category)
    # Include the current rotating slide (first showcase image) and the new prediction image
    _transition_from_b64 = showcase_b64s[0] if showcase_b64s else _pred_image_b64
    hero_html = (
        f'<div class="hero-cinema" id="hero-cinema" '
        f'data-predicted="true" data-pred-value="{_pred_value:.0f}" '
        f'data-pred-category="{_pred_category}">'
        f'<div class="hero-slide" id="hero-slide-old" style="background-image:url(data:image/jpeg;base64,{_transition_from_b64});opacity:1;"></div>'
        f'<div class="hero-slide" id="hero-slide-new" style="background-image:url(data:image/jpeg;base64,{_pred_image_b64});opacity:0;"></div>'
        f'<div class="hero-overlay" style="background:linear-gradient(180deg,'
        f'rgba(13,7,5,{o_top}) 0%,rgba(13,7,5,{o_mid}) 35%,'
        f'rgba(13,7,5,{o_low}) 70%,rgba(13,7,5,{o_bot}) 100%);"></div>'
        f'<div class="hero-content" id="hero-content-transition">'
        f'<div class="hero-badge"><span class="hero-badge-dot"></span> {_pred_label}</div>'
        f'<h1 class="hero-title">Estimated Value</h1>'
        f'<div class="hero-pred-price" id="hero-pred-price" '
        f'data-target="{_pred_value:.0f}">$0</div>'
        f'<div class="hero-pred-sub" style="color:rgba(248,245,240,0.5);font-size:16px;margin-top:8px;">'
        f'Based on {_pred_category.title()} classification &middot; {inputs.get("GrLivArea",0):,} sq ft &middot; '
        f'Quality {inputs.get("OverallQual",5)}/10</div>'
        f'<a href="#analytics" class="hero-cta" style="margin-top:28px;">View Full Analysis</a>'
        f'</div>'
        f'</div>'
    )
    hero_js = """
    <script>
    (function(){
      // Smooth hero image transition
      var slideOld = document.getElementById('hero-slide-old');
      var slideNew = document.getElementById('hero-slide-new');
      if(slideOld && slideNew){
        // Fade out old, fade in new over 600ms
        slideOld.style.transition = 'opacity 0.6s ease-in-out';
        slideNew.style.transition = 'opacity 0.6s ease-in-out';
        setTimeout(function(){
          slideOld.style.opacity = '0';
          slideNew.style.opacity = '1';
        }, 100);
      }
      // Animate price counter
      var el = document.getElementById('hero-pred-price');
      if(!el) return;
      var target = parseFloat(el.getAttribute('data-target'));
      var duration = 1800;
      var start = performance.now();
      function animate(now){
        var elapsed = now - start;
        var progress = Math.min(elapsed / duration, 1);
        var eased = 1 - Math.pow(1 - progress, 3);
        var current = Math.round(target * eased);
        el.textContent = '$' + current.toLocaleString('en-US');
        if(progress < 1) requestAnimationFrame(animate);
      }
      requestAnimationFrame(animate);
    })();
    </script>
    """
else:
    # Rotating showcase: random start, cycle every 4s
    slides_html = ""
    for i, b64 in enumerate(showcase_b64s):
        active = " active" if i == 0 else ""
        slides_html += f'<div class="hero-slide{active}" style="background-image:url(data:image/jpeg;base64,{b64});"></div>'
    # First image determines initial overlay
    first_cat = None
    for cat in HOUSE_CATS:
        if cat in cat_images and cat_images[cat][0] == showcase_b64s[0]:
            first_cat = cat
            break
    o_top, o_mid, o_low, o_bot = _overlay_for(first_cat or "suburban")

    hero_html = (
        f'<div class="hero-cinema" id="hero-cinema" data-predicted="false">'
        f'{slides_html}'
        f'<div class="hero-overlay" style="background:linear-gradient(180deg,'
        f'rgba(13,7,5,{o_top}) 0%,rgba(13,7,5,{o_mid}) 35%,'
        f'rgba(13,7,5,{o_low}) 70%,rgba(13,7,5,{o_bot}) 100%);"></div>'
        f'<div class="hero-content">'
        f'<h1 class="hero-title">Predict Real Estate<br>Value With <span class="hero-title-gold">Precision</span></h1>'
        f'<p class="hero-subtitle">Leverage machine learning trained on 1,460 property records to deliver accurate, explainable house price predictions in seconds.</p>'
        f'<div class="hero-stats">'
        f'<div class="hero-stat"><div class="hero-stat-val">$163K</div><div class="hero-stat-label">Median Value</div></div>'
        f'<div class="hero-stat-divider"></div>'
        f'<div class="hero-stat"><div class="hero-stat-val">1,460</div><div class="hero-stat-label">Properties Analyzed</div></div>'
        f'<div class="hero-stat-divider"></div>'
        f'<div class="hero-stat"><div class="hero-stat-val">68%</div><div class="hero-stat-label">Within 10% Accuracy</div></div>'
        f'</div>'
        f'<a href="#predict" class="hero-cta">Predict Now</a>'
        f'</div>'
        f'</div>'
    )
    hero_js = """
    <script>
    (function(){
      var slides = document.querySelectorAll('#hero-cinema .hero-slide');
      if(slides.length <= 1) return;
      var current = 0;
      setInterval(function(){
        slides[current].classList.remove('active');
        current = (current + 1) % slides.length;
        slides[current].classList.add('active');
      }, 4000);
    })();
    </script>
    """

st.markdown(hero_html + hero_js, unsafe_allow_html=True)

# ============================================================
# TRUST METRICS
# ============================================================
st.markdown('<div class="section-divider" style="max-width:1450px;margin:110px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(215,162,78,0.12),transparent);"></div>', unsafe_allow_html=True)

st.markdown(
    '<div style="text-align:center;"><span style="display:inline-flex;align-items:center;gap:8px;padding:6px 16px;background:rgba(215,162,78,0.08);border:1px solid rgba(215,162,78,0.18);border-radius:100px;color:#D7A24E;font-size:11px;font-weight:700;letter-spacing:2px;text-transform:uppercase;">Why HomeSense</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<h2 style="font-family:Playfair Display,serif;font-size:34px;color:#FFFFFF;font-weight:700;text-align:center;letter-spacing:-1px;">Trusted by Data</h2>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="font-size:18px;color:#B5AFA8;text-align:center;max-width:520px;margin:0 auto 40px;">Built on rigorous machine learning research with full transparency in how every prediction is made.</p>',
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
st.markdown('<div class="section-divider" style="max-width:1450px;margin:110px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(215,162,78,0.12),transparent);"></div>', unsafe_allow_html=True)

st.markdown(
    '<div style="text-align:center;"><span style="display:inline-flex;align-items:center;gap:8px;padding:6px 16px;background:rgba(215,162,78,0.08);border:1px solid rgba(215,162,78,0.18);border-radius:100px;color:#D7A24E;font-size:11px;font-weight:700;letter-spacing:2px;text-transform:uppercase;">Capabilities</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<h2 style="font-family:Playfair Display,serif;font-size:34px;color:#FFFFFF;font-weight:700;text-align:center;letter-spacing:-1px;">Built for Intelligence</h2>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="font-size:18px;color:#B5AFA8;text-align:center;max-width:540px;margin:0 auto 32px;">Every feature is engineered to deliver insights that matter.</p>',
    unsafe_allow_html=True,
)

f1, f2, f3, f4 = st.columns(4)
with f1:
    st.markdown(
        '<div class="premium-card" style="text-align:center;">'
        '<div style="width:56px;height:56px;margin:0 auto 16px;background:rgba(215,162,78,0.1);border:1px solid rgba(215,162,78,0.2);border-radius:24px;display:flex;align-items:center;justify-content:center;">'
        '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#D7A24E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg></div>'
        '<div style="color:#FFFFFF;font-weight:700;font-size:16px;margin-bottom:8px;">AI Price Prediction</div>'
        '<div style="color:#B5AFA8;font-size:14px;line-height:1.8;">ML-powered valuation trained on 245 property features.</div></div>',
        unsafe_allow_html=True,
    )
with f2:
    st.markdown(
        '<div class="premium-card" style="text-align:center;">'
        '<div style="width:56px;height:56px;margin:0 auto 16px;background:rgba(215,162,78,0.1);border:1px solid rgba(215,162,78,0.2);border-radius:24px;display:flex;align-items:center;justify-content:center;">'
        '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#D7A24E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/></svg></div>'
        '<div style="color:#FFFFFF;font-weight:700;font-size:16px;margin-bottom:8px;">Explainable AI</div>'
        '<div style="color:#B5AFA8;font-size:14px;line-height:1.8;">See exactly why the model predicts any price with full transparency.</div></div>',
        unsafe_allow_html=True,
    )
with f3:
    st.markdown(
        '<div class="premium-card" style="text-align:center;">'
        '<div style="width:56px;height:56px;margin:0 auto 16px;background:rgba(215,162,78,0.1);border:1px solid rgba(215,162,78,0.2);border-radius:24px;display:flex;align-items:center;justify-content:center;">'
        '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#D7A24E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg></div>'
        '<div style="color:#FFFFFF;font-weight:700;font-size:16px;margin-bottom:8px;">Feature Importance</div>'
        '<div style="color:#B5AFA8;font-size:14px;line-height:1.8;">Visual breakdowns of which factors contribute most to value.</div></div>',
        unsafe_allow_html=True,
    )
with f4:
    st.markdown(
        '<div class="premium-card" style="text-align:center;">'
        '<div style="width:56px;height:56px;margin:0 auto 16px;background:rgba(215,162,78,0.1);border:1px solid rgba(215,162,78,0.2);border-radius:24px;display:flex;align-items:center;justify-content:center;">'
        '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#D7A24E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12V7H5a2 2 0 010-4h14v4"/><path d="M3 5v14a2 2 0 002 2h16v-5"/><path d="M18 12a2 2 0 000 4h4v-4h-4z"/></svg></div>'
        '<div style="color:#FFFFFF;font-weight:700;font-size:16px;margin-bottom:8px;">Market Insights</div>'
        '<div style="color:#B5AFA8;font-size:14px;line-height:1.8;">Neighborhood and market tier comparisons at a glance.</div></div>',
        unsafe_allow_html=True,
    )

# ============================================================
# PREDICTION FORM
# ============================================================
st.markdown(
    '<div style="max-width:1450px;margin:110px auto 80px;display:flex;align-items:center;justify-content:center;gap:24px;">'
    '<div style="flex:1;height:1px;background:linear-gradient(90deg,transparent,rgba(215,162,78,0.18));"></div>'
    '<span style="color:#B5AFA8;font-size:12px;font-weight:600;letter-spacing:3px;text-transform:uppercase;">Trusted by Machine Learning</span>'
    '<div style="flex:1;height:1px;background:linear-gradient(90deg,rgba(215,162,78,0.18),transparent);"></div>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown('<div id="predict"></div>', unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center;"><span style="display:inline-flex;align-items:center;gap:8px;padding:6px 16px;background:rgba(215,162,78,0.08);border:1px solid rgba(215,162,78,0.18);border-radius:100px;color:#D7A24E;font-size:11px;font-weight:700;letter-spacing:2px;text-transform:uppercase;">Valuation Engine</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<h2 style="font-family:Playfair Display,serif;font-size:34px;color:#FFFFFF;font-weight:700;text-align:center;letter-spacing:-1px;">Predict House Price</h2>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="font-size:18px;color:#B5AFA8;text-align:center;max-width:520px;margin:0 auto 40px;">Enter the property details below to generate an AI-powered valuation.</p>',
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
        ("Analyzing Property", 0.25),
        ("Evaluating Market", 0.50),
        ("Generating Valuation Report", 0.75),
        ("Finalizing Insights", 1.0),
    ]
    completed_steps = []
    progress_bar = st.progress(0)
    status_text = loading_placeholder.empty()

    # Premium loading animation
    for step_label, progress in timeline_steps:
        completed_steps.append(step_label)
        status_text.markdown(
            f'<div style="text-align:center;padding:20px;">'
            f'<div style="color:#D7A24E;font-size:14px;font-weight:600;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px;">{step_label}</div>'
            f'<div style="color:#B5AFA8;font-size:13px;">Please wait while we process your property data</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        progress_bar.progress(progress)
        time.sleep(0.6)  # 0.6s x 4 steps = 2.4s total

    # Complete animation
    status_text.markdown(
        '<div style="text-align:center;padding:20px;">'
        '<div style="color:#4CAF50;font-size:14px;font-weight:600;letter-spacing:2px;text-transform:uppercase;">✓ Analysis Complete</div>'
        '</div>',
        unsafe_allow_html=True
    )
    time.sleep(0.3)
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
    st.markdown('<div class="section-divider" style="max-width:1450px;margin:110px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(215,162,78,0.12),transparent);"></div>', unsafe_allow_html=True)

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
    st.markdown('<div class="section-divider" style="max-width:1450px;margin:110px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(215,162,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="margin-bottom:32px;">'
        '<div style="color:#B5AFA8;font-size:14px;font-weight:600;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">Details</div>'
        '<h2 style="font-family:Playfair Display,serif;font-size:34px;color:#D7A24E;font-weight:700;letter-spacing:-0.5px;margin:0;">Property Facts</h2>'
        '</div>',
        unsafe_allow_html=True,
    )

    fact_bedrooms = str(inputs.get("BedroomAbvGr", 3))
    fact_bathrooms = str(inputs.get("FullBath", 2))
    fact_garage = f"{inputs.get('GarageCars', 2)} Car{'s' if inputs.get('GarageCars', 2) != 1 else ''}"
    fact_year_built = str(inputs.get("YearBuilt", 2005))
    fact_living_area = f"{inputs.get('GrLivArea', 1500):,} sq.ft"
    fact_neighborhood = NEIGHBORHOOD_LABELS.get(inputs.get("Neighborhood", ""), inputs.get("Neighborhood", ""))

    facts_html = f"""
    <div class="facts-grid">
      <div class="fact-item">
        <span class="fact-label">Bedrooms</span>
        <span class="fact-value">{fact_bedrooms}</span>
      </div>
      <div class="fact-item">
        <span class="fact-label">Bathrooms</span>
        <span class="fact-value">{fact_bathrooms}</span>
      </div>
      <div class="fact-item">
        <span class="fact-label">Garage</span>
        <span class="fact-value">{fact_garage}</span>
      </div>
      <div class="fact-item">
        <span class="fact-label">Year Built</span>
        <span class="fact-value">{fact_year_built}</span>
      </div>
      <div class="fact-item">
        <span class="fact-label">Living Area</span>
        <span class="fact-value">{fact_living_area}</span>
      </div>
      <div class="fact-item">
        <span class="fact-label">Neighborhood</span>
        <span class="fact-value">{fact_neighborhood}</span>
      </div>
    </div>
    """
    st.markdown(facts_html, unsafe_allow_html=True)

    # MARKET SEGMENT
    st.markdown('<div class="section-divider" style="max-width:1450px;margin:110px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(215,162,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    st.caption("Positioning")
    st.subheader("Market Segment")
    st.progress(profile["segment_pct"] / 100)
    st.markdown(f"**{profile['market_segment']}**")

    # PROPERTY HIGHLIGHTS
    if profile["highlights"]:
        st.markdown('<div class="section-divider" style="max-width:1450px;margin:110px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(215,162,78,0.12),transparent);"></div>', unsafe_allow_html=True)
        st.caption("Strengths")
        st.subheader("Property Highlights")
        for h in profile["highlights"]:
            st.markdown(
                f'<div class="highlight-card">'
                f'<div class="highlight-card-title">{h["icon"]} {h["title"]}</div>'
                f'<div class="highlight-card-desc">{h["detail"]}</div></div>',
                unsafe_allow_html=True,
            )

    # DRAWBACKS
    if profile["drawbacks"]:
        st.markdown('<div class="section-divider" style="max-width:1450px;margin:110px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(215,162,78,0.12),transparent);"></div>', unsafe_allow_html=True)
        st.caption("Considerations")
        st.subheader("Potential Drawbacks")
        for d in profile["drawbacks"]:
            st.markdown(
                f'<div class="premium-card" style="margin-bottom:8px;border-left:3px solid #FFD060;">'
                f'<strong style="color:#FFFFFF;">{d["icon"]} {d["title"]}</strong><br>'
                f'<span style="color:#F0ECE8;font-size:14px;">{d["detail"]}</span></div>',
                unsafe_allow_html=True,
            )

    # PROFILE VERDICT
    st.markdown('<div class="section-divider" style="max-width:1450px;margin:110px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(215,162,78,0.12),transparent);"></div>', unsafe_allow_html=True)
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
        f'<div style="font-family:Playfair Display,serif;font-size:56px;color:#D7A24E;font-weight:700;">{profile["score"]:.1f}</div>'
        f'<div style="color:rgba(248,245,240,0.4);font-size:13px;text-transform:uppercase;letter-spacing:1px;margin-bottom:20px;">Overall Score</div>'
        f'<div style="display:inline-block;padding:8px 24px;border-radius:100px;font-size:14px;font-weight:700;margin-bottom:20px;'
        f'{"background:rgba(102,209,122,0.1);border:1px solid rgba(102,209,122,0.25);color:#4CAF50;" if rec_class == "excellent" else "background:rgba(215,162,78,0.1);border:1px solid rgba(215,162,78,0.25);color:#D7A24E;" if rec_class == "good" else "background:rgba(255,208,96,0.1);border:1px solid rgba(255,208,96,0.25);color:#FFD060;"}'
        f'">{profile["recommendation"]}</div>'
        f'<div style="color:#F0ECE8;line-height:1.8;max-width:600px;margin:0 auto;">'
        f'This property belongs to the <strong>{profile["market_segment"]}</strong> market segment. {segment_desc}</div></div>',
        unsafe_allow_html=True,
    )

    # ── VALUATION CARD ──
    st.markdown('<div class="section-divider" style="max-width:1450px;margin:110px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(215,162,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    confidence = confidence_data["score"]
    confidence_color = "#4CAF50" if confidence >= 85 else "#D7A24E" if confidence >= 70 else "#FFD060"
    verdict = generate_ai_verdict(inputs, prediction, scores, market)
    value_drivers = generate_value_drivers(inputs, prediction)
    investment_interp = generate_investment_interpretation(inputs, prediction, scores, market)

    st.caption("Property Valuation")
    st.subheader("Valuation")
    st.markdown(
        f'<div class="premium-card" style="text-align:center;">'
        f'<span style="display:inline-block;padding:6px 16px;border-radius:100px;font-size:12px;font-weight:700;'
        f'{"background:rgba(102,209,122,0.1);border:1px solid rgba(102,209,122,0.25);color:#4CAF50;" if badge_class == "premium" else "background:rgba(215,162,78,0.1);border:1px solid rgba(215,162,78,0.25);color:#D7A24E;"}'
        f'">{badge_text}</span><br><br>'
        f'<div style="font-family:Playfair Display,serif;font-size:clamp(40px,5vw,64px);color:#D7A24E;font-weight:700;">${prediction:,.0f}</div>'
        f'<div style="color:#B5AFA8;font-size:15px;margin-bottom:20px;">Estimated Market Value</div>'
        f'<div style="color:rgba(248,245,240,0.35);font-size:13px;">Likely Range: ${pred_interval["lower"]:,.0f} - ${pred_interval["upper"]:,.0f} (95% CI)</div>'
        f'<div style="margin-top:12px;"><span style="color:#B5AFA8;font-size:13px;">Confidence: </span>'
        f'<span style="color:{confidence_color};font-weight:700;">{confidence}%</span> '
        f'<span style="color:rgba(248,245,240,0.35);font-size:12px;">{confidence_data["grade"]} Confidence</span></div></div>',
        unsafe_allow_html=True,
    )

    # ── AI VERDICT ──
    st.markdown('<div class="section-divider" style="max-width:1450px;margin:110px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(215,162,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    st.caption("Verdict")
    st.subheader("AI Valuation Verdict")
    st.markdown(
        f'<div class="premium-card" style="border-left:3px solid {verdict["color"]};">'
        f'<div style="display:inline-block;padding:6px 18px;border-radius:100px;font-size:13px;font-weight:700;border:1px solid {verdict["color"]};color:{verdict["color"]};margin-bottom:16px;">{verdict["verdict"]}</div>'
        f'<div style="color:rgba(248,245,240,0.65);line-height:1.8;margin-bottom:12px;">{verdict["reasoning"]}</div>'
        f'<div style="color:{verdict["color"]};font-weight:600;">{verdict["action"]}</div></div>',
        unsafe_allow_html=True,
    )

    # ── WHAT DRIVES THIS PRICE ──
    st.markdown('<div class="section-divider" style="max-width:1450px;margin:110px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(215,162,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    st.caption("Value Assessment")
    st.subheader("What Drives This Price")
    st.markdown('<p style="color:#B5AFA8;">Key factors that determine the estimated value of this property.</p>', unsafe_allow_html=True)
    for d in value_drivers:
        st.markdown(
            f'<div class="drive-row">'
            f'<span class="drive-factor">{d["icon"]} {d["title"]}</span>'
            f'<span class="drive-contrib">High</span>'
            f'<span class="drive-impact" style="color:#4CAF50;">+</span></div>',
            unsafe_allow_html=True,
        )

    # ── INVESTMENT INTERPRETATION ──
    st.markdown('<div class="section-divider" style="max-width:1450px;margin:110px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(215,162,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    st.caption("Analyst Perspective")
    st.subheader("If This Were My Investment")
    st.markdown(
        f'<div class="premium-card" style="border-left:3px solid #D7A24E;">'
        f'<p style="color:rgba(248,245,240,0.65);line-height:1.8;font-style:italic;">{investment_interp}</p></div>',
        unsafe_allow_html=True,
    )

    # ── PROPERTY ASSESSMENT ──
    st.markdown('<div class="section-divider" style="max-width:1450px;margin:110px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(215,162,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="margin-bottom:32px;">'
        '<div style="color:#B5AFA8;font-size:14px;font-weight:600;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">Assessment</div>'
        '<h2 style="font-family:Playfair Display,serif;font-size:34px;color:#D7A24E;font-weight:700;letter-spacing:-0.5px;margin:0;">Property Assessment</h2>'
        '</div>',
        unsafe_allow_html=True,
    )
    overall_label = get_score_label(scores["investment"])
    condition_label = get_score_label(scores["resale"])
    luxury_label = get_score_label(scores["luxury"])
    investment_label = get_score_label(scores["investment"])
    st.markdown(
        '<div class="assess-grid" style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;">'
        f'<div class="assess-card"><div class="assess-label">Overall</div><div class="assess-value">{overall_label}</div></div>'
        f'<div class="assess-card"><div class="assess-label">Condition</div><div class="assess-value">{condition_label}</div></div>'
        f'<div class="assess-card"><div class="assess-label">Luxury</div><div class="assess-value">{luxury_label}</div></div>'
        f'<div class="assess-card"><div class="assess-label">Investment</div><div class="assess-value">{investment_label}</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── EXECUTIVE SUMMARY ──
    st.markdown('<div class="section-divider" style="max-width:1450px;margin:110px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(215,162,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="margin-bottom:32px;">'
        '<div style="color:#B5AFA8;font-size:14px;font-weight:600;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">Summary</div>'
        '<h2 style="font-family:Playfair Display,serif;font-size:34px;color:#D7A24E;font-weight:700;letter-spacing:-0.5px;margin:0;">Executive Summary</h2>'
        '</div>',
        unsafe_allow_html=True,
    )
    summary_parts = ai_summary.split("Key Drivers")
    summary_html = (
        f'<div class="premium-card" style="border-left:3px solid #D7A24E;">'
        f'<div style="color:#B5AFA8;font-size:14px;font-weight:600;letter-spacing:1px;text-transform:uppercase;margin-bottom:12px;">Overall Assessment</div>'
        f'<p style="color:#F0ECE8;line-height:1.8;">{summary_parts[0].strip()}</p>'
    )
    if len(summary_parts) > 1:
        summary_html += (
            f'<div style="color:#B5AFA8;font-size:14px;font-weight:600;letter-spacing:1px;text-transform:uppercase;margin:24px 0 12px;">Key Drivers</div>'
            f'<p style="color:#F0ECE8;line-height:1.8;">{summary_parts[1].strip()}</p>'
        )
    summary_html += '</div>'
    st.markdown(summary_html, unsafe_allow_html=True)

    # ── INVESTMENT SCORES ──
    st.markdown('<div class="section-divider" style="max-width:1450px;margin:110px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(215,162,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="margin-bottom:32px;">'
        '<div style="color:#B5AFA8;font-size:14px;font-weight:600;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">Investment Analysis</div>'
        '<h2 style="font-family:Playfair Display,serif;font-size:34px;color:#D7A24E;font-weight:700;letter-spacing:-0.5px;margin:0;">Investment Scores</h2>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<p style="color:#B5AFA8;">Quantified assessment based on property characteristics and market position.</p>', unsafe_allow_html=True)

    invest_score = scores["investment"]
    invest_stars_full = invest_score // 20
    invest_stars_half = 1 if (invest_score % 20) >= 10 else 0
    invest_stars_empty = 5 - invest_stars_full - invest_stars_half
    stars_html = "★" * invest_stars_full + ("½" if invest_stars_half else "") + "☆" * invest_stars_empty
    st.markdown(
        f'<div class="premium-card" style="text-align:center;margin-bottom:16px;">'
        f'<div class="invest-rating">{invest_score}/100</div>'
        f'<div class="invest-stars">{stars_html}</div>'
        f'<div style="color:#B5AFA8;font-size:14px;margin-top:8px;">{get_score_label(invest_score)}</div></div>',
        unsafe_allow_html=True,
    )

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
    st.markdown('<div class="section-divider" style="max-width:1450px;margin:110px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(215,162,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    st.caption("Scorecard")
    st.subheader("Property Report Card")
    st.markdown('<p style="color:#B5AFA8;">Comprehensive scoring across 7 key categories.</p>', unsafe_allow_html=True)

    filtered_scorecard = [sc for sc in scorecard if sc["category"] != "Quality"]
    total_score = sum(sc["score"] for sc in filtered_scorecard)
    max_total = sum(sc["max"] for sc in filtered_scorecard)
    overall_grade = "A" if total_score >= max_total * 0.8 else "B" if total_score >= max_total * 0.65 else "C" if total_score >= max_total * 0.5 else "D"

    st.markdown(
        f'<div class="premium-card" style="text-align:center;margin-bottom:20px;">'
        f'<div style="font-family:Playfair Display,serif;font-size:48px;color:#D7A24E;font-weight:700;">{overall_grade}</div>'
        f'<div style="color:rgba(248,245,240,0.4);font-size:14px;">{total_score}/{max_total} Overall</div></div>',
        unsafe_allow_html=True,
    )
    for sc in filtered_scorecard:
        sc_color = "#4CAF50" if sc["score"] >= 8 else "#D7A24E" if sc["score"] >= 6 else "#FFD060" if sc["score"] >= 4 else "#E57373"
        st.markdown(
            f'<div class="premium-card" style="margin-bottom:8px;padding:14px 18px;">'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:6px;">'
            f'<span style="color:#FFFFFF;font-weight:600;">{sc["category"]}</span>'
            f'<span style="color:{sc_color};font-weight:700;">{sc["score"]}/{sc["max"]}</span></div>'
            f'<div style="height:6px;background:rgba(255,255,255,0.04);border-radius:6px;overflow:hidden;margin-bottom:6px;">'
            f'<div style="height:100%;width:{sc["score"]/sc["max"]*100}%;background:{sc_color};border-radius:6px;"></div></div>'
            f'<div style="color:rgba(248,245,240,0.4);font-size:13px;">{sc["detail"]}</div></div>',
            unsafe_allow_html=True,
        )

    # ── MARKET POSITION ──
    st.markdown('<div class="section-divider" style="max-width:1450px;margin:110px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(215,162,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    pos = market["percentile"]
    tier_colors = {"Entry-Level": "#E57373", "Value": "#FFD060", "Average": "#D7A24E", "Above Average": "#4CAF50", "Premium": "#4CAF50", "Luxury": "#D7A24E"}
    tier_color = tier_colors.get(market["tier"], "#D7A24E")
    vs_avg_sign = "+" if market["vs_avg"] >= 0 else ""
    vs_avg_color = "#4CAF50" if market["vs_avg"] >= 0 else "#E57373"
    vs_avg_arrow = "▲" if market["vs_avg"] >= 0 else "▼"

    st.markdown(
        '<div style="margin-bottom:32px;">'
        '<div style="color:#B5AFA8;font-size:14px;font-weight:600;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">Market Analysis</div>'
        '<h2 style="font-family:Playfair Display,serif;font-size:34px;color:#D7A24E;font-weight:700;letter-spacing:-0.5px;margin:0;">Market Position</h2>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="premium-card">'
        f'<div style="display:inline-block;padding:8px 20px;border-radius:100px;font-size:14px;font-weight:700;border:1px solid {tier_color};color:{tier_color};margin-bottom:12px;">{market["tier"]}</div>'
        f'<div style="color:#F0ECE8;margin-bottom:16px;">{market["tier_desc"]}</div>'
        f'<div style="height:8px;background:rgba(255,255,255,0.04);border-radius:8px;position:relative;margin:16px 0;">'
        f'<div style="height:100%;width:{pos}%;background:linear-gradient(90deg,rgba(215,162,78,0.3),rgba(215,162,78,0.7));border-radius:8px;"></div></div>'
        f'<div style="display:flex;justify-content:space-between;color:rgba(248,245,240,0.35);font-size:12px;margin-bottom:16px;">'
        f'<span>Budget</span><span>Average</span><span>Premium</span><span>Luxury</span></div></div>',
        unsafe_allow_html=True,
    )
    mp1, mp2, mp3 = st.columns(3)
    with mp1:
        st.metric(label="Percentile", value=f"{pos}th")
    with mp2:
        st.markdown(
            f'<div style="background:rgba(38,26,20,0.65);border:1px solid rgba(215,162,78,0.15);border-radius:14px;padding:20px 24px;text-align:center;">'
            f'<div style="color:#B5AFA8;font-size:15px;margin-bottom:8px;">vs Average</div>'
            f'<div style="color:{vs_avg_color};font-size:28px;font-weight:700;">{vs_avg_arrow} {vs_avg_sign}${market["vs_avg"]:,.0f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with mp3:
        st.markdown(
            f'<div style="background:rgba(38,26,20,0.65);border:1px solid rgba(215,162,78,0.15);border-radius:14px;padding:20px 24px;text-align:center;">'
            f'<div style="color:#B5AFA8;font-size:15px;margin-bottom:8px;">vs Average %</div>'
            f'<div style="color:{vs_avg_color};font-size:28px;font-weight:700;">{vs_avg_arrow} {vs_avg_sign}{market["vs_avg_pct"]:.1f}%</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── TOP VALUATION FACTORS ──
    st.markdown('<div class="section-divider" style="max-width:1450px;margin:110px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(215,162,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    insights = generate_explanations(inputs, prediction)
    st.markdown(
        '<div style="margin-bottom:32px;">'
        '<div style="color:#B5AFA8;font-size:14px;font-weight:600;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">Key Drivers</div>'
        '<h2 style="font-family:Playfair Display,serif;font-size:34px;color:#D7A24E;font-weight:700;letter-spacing:-0.5px;margin:0;">Top Valuation Factors</h2>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<p style="color:#B5AFA8;">The most influential characteristics affecting this property estimated value.</p>', unsafe_allow_html=True)
    for insight in insights:
        ic = "#4CAF50" if insight["type"] == "positive" else "#E57373" if insight["type"] == "negative" else "#FFD060"
        st.markdown(
            f'<div class="premium-card" style="margin-bottom:8px;border-left:3px solid {ic};">'
            f'<strong style="color:#FFFFFF;">{insight["icon"]} {insight["title"]}</strong><br>'
            f'<span style="color:#F0ECE8;font-size:14px;">{insight["description"]}</span><br>'
            f'<span style="color:{ic};font-size:13px;font-weight:600;">{insight.get("contribution", "")}</span></div>',
            unsafe_allow_html=True,
        )

    # ── VISUAL INSIGHTS ──
    st.markdown('<div class="section-divider" style="max-width:1450px;margin:110px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(215,162,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    st.markdown('<div id="analytics"></div>', unsafe_allow_html=True)
    st.caption("Analytics")
    st.subheader("Visual Insights")
    st.markdown('<p style="color:#B5AFA8;">Interactive charts to understand the valuation from every angle.</p>', unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(create_feature_importance_chart(), width="stretch", config={"displayModeBar": False})
        st.plotly_chart(create_price_comparison_chart(prediction), width="stretch", config={"displayModeBar": False})
    with chart_col2:
        st.plotly_chart(create_input_summary_chart(inputs), width="stretch", config={"displayModeBar": False})

    # ── RECOMMENDATIONS ──
    st.markdown('<div class="section-divider" style="max-width:1450px;margin:110px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(215,162,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    st.caption("Recommendations")
    st.subheader("Recommendations")
    st.markdown('<p style="color:#B5AFA8;">Targeted improvements to increase property value and marketability.</p>', unsafe_allow_html=True)
    for rec in recs:
        st.markdown(
            f'<div class="premium-card" style="margin-bottom:8px;">'
            f'<strong style="color:#FFFFFF;">{rec["icon"]} {rec["title"]}</strong><br>'
            f'<span style="color:#F0ECE8;font-size:14px;">{rec["description"]}</span><br>'
            f'<span style="color:#D7A24E;font-size:13px;font-weight:600;">{rec["impact"]}</span></div>',
            unsafe_allow_html=True,
        )

    # ── SMART IMPROVEMENT SIMULATOR ──
    st.markdown('<div class="section-divider" style="max-width:1450px;margin:110px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(215,162,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    st.caption("Improvements")
    st.subheader("Smart Improvement Simulator")
    st.markdown('<p style="color:#B5AFA8;">Potential improvements with estimated costs and value impact.</p>', unsafe_allow_html=True)
    for imp in improvements:
        roi_color = "#4CAF50" if imp["roi"] >= 100 else "#D7A24E" if imp["roi"] >= 50 else "#FFD060"
        cost_text = f"${imp['cost_low']:,.0f} - ${imp['cost_high']:,.0f}" if imp["cost_low"] > 0 else "N/A"
        value_text = f"+${imp['est_value']:,.0f}" if imp["est_value"] > 0 else "N/A"
        st.markdown(
            f'<div class="premium-card" style="margin-bottom:8px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
            f'<strong style="color:#FFFFFF;">{imp["feature"]}</strong>'
            f'<span style="color:{roi_color};font-weight:700;">{imp["roi"]}% ROI</span></div>'
            f'<div style="color:#F0ECE8;font-size:14px;margin-bottom:8px;">{imp["description"]}</div>'
            f'<div style="display:flex;gap:24px;">'
            f'<div><span style="color:rgba(248,245,240,0.4);font-size:12px;">Est. Cost</span><br><span style="color:#FFFFFF;font-weight:600;">{cost_text}</span></div>'
            f'<div><span style="color:rgba(248,245,240,0.4);font-size:12px;">Est. Value Add</span><br><span style="color:#4CAF50;font-weight:600;">{value_text}</span></div></div></div>',
            unsafe_allow_html=True,
        )

    # ── NEIGHBORHOOD INSIGHTS ──
    st.markdown('<div class="section-divider" style="max-width:1450px;margin:110px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(215,162,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    nb = nb_insights
    st.caption("Location Intelligence")
    st.subheader("Neighborhood Insights")
    nb1, nb2 = st.columns([2, 1])
    with nb1:
        st.markdown(
            f'<div class="premium-card">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">'
            f'<strong style="color:#FFFFFF;font-size:18px;">{nb["name"]}</strong>'
            f'<span style="padding:4px 14px;border-radius:100px;font-size:12px;font-weight:600;background:rgba(215,162,78,0.1);border:1px solid rgba(215,162,78,0.2);color:#D7A24E;">{nb["tier"]}</span></div></div>',
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

    # ── MARKET BENCHMARK (merged) ──
    st.markdown('<div class="section-divider" style="max-width:1450px;margin:110px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(215,162,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    st.caption("Benchmark Analysis")
    st.subheader("Market Benchmark")
    st.markdown('<p style="color:#B5AFA8;">How your property compares to dataset averages and percentile rankings.</p>', unsafe_allow_html=True)

    # Comparison rows
    for comp in comparison:
        diff_color = "#4CAF50" if comp["better"] else "#E57373"
        diff_sign = "+" if comp["difference"] > 0 else ""
        st.markdown(
            f'<div class="premium-card" style="margin-bottom:8px;padding:14px 18px;">'
            f'<div class="comparison-row" style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">'
            f'<span style="color:#FFFFFF;font-weight:600;min-width:120px;">{comp["name"]}</span>'
            f'<span style="color:#D7A24E;font-weight:700;">{comp["yours"]}{comp["unit"]}</span>'
            f'<span style="color:rgba(248,245,240,0.3);">vs</span>'
            f'<span style="color:#F0ECE8;">{comp["average"]}{comp["unit"]}</span>'
            f'<span style="color:{diff_color};font-weight:600;word-break:break-word;">{diff_sign}{comp["difference"]}{comp["unit"]} ({diff_sign}{comp["pct"]}%)</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    # Percentile benchmarks
    if benchmark_percentiles:
        st.markdown('<div style="margin:20px 0 12px;"><span style="color:#B5AFA8;font-size:14px;text-transform:uppercase;letter-spacing:1.5px;font-weight:600;">Percentile Rankings</span></div>', unsafe_allow_html=True)
        for bp in benchmark_percentiles:
            tier_color_b = "#4CAF50" if "Top" in bp["tier"] or "Above" in bp["tier"] else "#D7A24E" if bp["tier"] == "Average" else "#FFD060"
            st.markdown(
                f'<div class="premium-card" style="margin-bottom:8px;padding:14px 18px;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">'
                f'<span style="color:#FFFFFF;font-weight:600;">{bp["feature"]}</span>'
                f'<span style="color:#D7A24E;font-weight:600;">{bp["value"]}{bp["unit"]}</span></div>'
                f'<div style="display:flex;align-items:center;gap:12px;">'
                f'<div style="flex:1;height:6px;background:rgba(255,255,255,0.04);border-radius:6px;overflow:hidden;">'
                f'<div style="height:100%;width:{bp["percentile"]}%;background:linear-gradient(90deg,rgba(215,162,78,0.3),rgba(215,162,78,0.7));border-radius:6px;"></div></div>'
                f'<span style="color:rgba(248,245,240,0.4);font-size:13px;min-width:40px;">{bp["percentile"]:.0f}th</span>'
                f'<span style="color:{tier_color_b};font-weight:600;font-size:13px;min-width:80px;">{bp["tier"]}</span></div></div>',
                unsafe_allow_html=True,
            )

    # ── SIMILAR PROPERTIES ──
    if similar_props:
        st.markdown('<div class="section-divider" style="max-width:1450px;margin:110px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(215,162,78,0.12),transparent);"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="margin-bottom:32px;">'
            '<div style="color:#B5AFA8;font-size:14px;font-weight:600;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">Comparable Analysis</div>'
            '<h2 style="font-family:Playfair Display,serif;font-size:34px;color:#D7A24E;font-weight:700;letter-spacing:-0.5px;margin:0;">Similar Properties</h2>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<p style="color:#B5AFA8;">Properties from the training dataset most similar to yours.</p>', unsafe_allow_html=True)

        # Load local house images for similar properties
        def _get_property_category(inputs, prediction):
            oq = inputs.get("OverallQual", 5)
            gla = inputs.get("GrLivArea", 1500)
            gc = inputs.get("GarageCars", 2)
            yb = inputs.get("YearBuilt", 2005)
            beds = inputs.get("BedroomAbvGr", 3)

            if oq >= 9 and prediction > 326100 and gla > 3000 and gc >= 3:
                return "mansion"
            if oq >= 8 and prediction > 214000 and gla > 2500:
                return "luxury"
            if yb >= 2000 and oq >= 7 and gla > 1800:
                return "modern"
            if oq >= 6 and yb < 1970 and gla > 1800 and beds >= 4:
                return "colonial"
            if oq >= 6 and 1940 <= yb < 1970 and gla > 1500 and beds <= 3:
                return "craftsman"
            if oq >= 5 and gla >= 1200 and gc >= 2:
                return "suburban"
            if oq >= 5 and yb >= 1970 and gla < 1800 and gla >= 1100 and gc <= 2 and beds <= 3:
                return "townhouse"
            if oq >= 5 and yb < 1970 and gla >= 1200 and beds <= 3:
                return "ranch"
            if oq <= 5 and gla < 1100:
                return "starter"
            if oq <= 5 and gla < 1200:
                return "cottage"
            return "suburban"

        def _load_property_images(category, n=3):
            """Load images prioritizing same category, then closely related categories."""
            # Category affinity: related categories to fall back to
            AFFINITY = {
                "mansion": ["luxury", "modern", "colonial"],
                "luxury": ["mansion", "modern", "craftsman"],
                "modern": ["luxury", "craftsman", "suburban"],
                "colonial": ["craftsman", "suburban", "ranch"],
                "craftsman": ["colonial", "ranch", "suburban"],
                "suburban": ["craftsman", "townhouse", "ranch"],
                "townhouse": ["suburban", "starter", "cottage"],
                "ranch": ["craftsman", "suburban", "cottage"],
                "cottage": ["starter", "ranch", "townhouse"],
                "starter": ["cottage", "townhouse", "suburban"],
            }
            # Build ordered list: primary category first, then related
            ordered_cats = [category] + AFFINITY.get(category, []) + [c for c in HOUSE_CATS if c != category]
            result = []
            for cat in ordered_cats:
                if cat not in cat_images_all or not cat_images_all[cat]:
                    continue
                imgs = cat_images_all[cat]
                available = [b64 for b64, _ in imgs if b64]
                if not available:
                    continue
                # Pick all available images from this category (up to remaining needed)
                pick = random.sample(available, min(len(available), n - len(result)))
                result.extend([f"data:image/jpeg;base64,{b}" for b in pick])
                if len(result) >= n:
                    break
            return result[:n]

        # Get property category and load images (prioritize predicted category)
        prop_category = _get_property_category(inputs, prediction)
        property_images = _load_property_images(prop_category, n=3)

        # Category labels for display
        CATEGORY_LABELS = {
            "starter": "Starter Home",
            "cottage": "Charming Cottage",
            "ranch": "Ranch Residence",
            "townhouse": "Townhouse",
            "suburban": "Suburban Family Home",
            "craftsman": "Craftsman Home",
            "colonial": "Colonial Estate",
            "modern": "Modern Family Home",
            "luxury": "Luxury Villa",
            "mansion": "Executive Residence"
        }

        sim_cols = st.columns(len(similar_props))
        for i, prop in enumerate(similar_props):
            with sim_cols[i]:
                price_diff_sign = "+" if prop["price_diff"] >= 0 else ""
                diff_color_s = "#4CAF50" if prop["price_diff"] >= 0 else "#E57373"

                # Use local image (cycle through available images)
                img_url = property_images[i % len(property_images)] if property_images else ""

                # Get category label
                cat_label = CATEGORY_LABELS.get(prop_category, "Premium Property")

                st.markdown(
                    f'<div class="premium-card" style="text-align:center;">'
                    f'<div style="color:#D7A24E;font-size:13px;font-weight:600;letter-spacing:1px;text-transform:uppercase;margin-bottom:12px;">{cat_label}</div>'
                    f'<div style="font-family:Playfair Display,serif;font-size:32px;color:#D7A24E;font-weight:700;margin-bottom:12px;">${prop["sale_price"]:,.0f}</div>'
                    f'<div class="sim-img" style="margin:0 auto 16px;"><img src="{img_url}" style="width:280px;height:180px;border-radius:24px;object-fit:cover;" /></div>'
                    f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;text-align:left;margin-bottom:16px;">'
                    f'<div style="padding:10px 14px;background:rgba(38,26,20,0.4);border-radius:12px;"><span style="color:#B5AFA8;font-size:13px;">Quality</span><br><span style="color:#FFFFFF;font-weight:600;font-size:18px;">{prop["quality"]}/10</span></div>'
                    f'<div style="padding:10px 14px;background:rgba(38,26,20,0.4);border-radius:12px;"><span style="color:#B5AFA8;font-size:13px;">Area</span><br><span style="color:#FFFFFF;font-weight:600;font-size:18px;">{prop["area"]:,} sqft</span></div>'
                    f'<div style="padding:10px 14px;background:rgba(38,26,20,0.4);border-radius:12px;"><span style="color:#B5AFA8;font-size:13px;">Garage</span><br><span style="color:#FFFFFF;font-weight:600;font-size:18px;">{prop["garage"]} car</span></div>'
                    f'<div style="padding:10px 14px;background:rgba(38,26,20,0.4);border-radius:12px;"><span style="color:#B5AFA8;font-size:13px;">Year</span><br><span style="color:#FFFFFF;font-weight:600;font-size:18px;">{prop["year"]}</span></div></div>'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;padding:12px 16px;background:rgba(38,26,20,0.4);border-radius:12px;">'
                    f'<span style="color:#B5AFA8;font-size:13px;">{prop["similarity"]}% match</span>'
                    f'<span style="color:{diff_color_s};font-weight:600;font-size:14px;">{price_diff_sign}${prop["price_diff"]:,.0f}</span></div></div>',
                    unsafe_allow_html=True,
                )

    # ── APPRECIATION FORECAST ──
    st.markdown('<div class="section-divider" style="max-width:1450px;margin:110px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(215,162,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="margin-bottom:32px;">'
        '<div style="color:#B5AFA8;font-size:14px;font-weight:600;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">Growth Projection</div>'
        '<h2 style="font-family:Playfair Display,serif;font-size:34px;color:#D7A24E;font-weight:700;letter-spacing:-0.5px;margin:0;">Appreciation Forecast</h2>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<p style="color:#B5AFA8;">Projected property value growth based on {forecast["rate_label"].lower()} ({forecast["annual_rate"]}% annual) trends in {nb_insights["name"]}.</p>', unsafe_allow_html=True)
    fc_html = '<div class="appr-grid" style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;">'
    for fc in forecast["forecasts"]:
        fc_html += (
            f'<div class="forecast-card" style="background:rgba(215,162,78,0.06);border:1px solid rgba(215,162,78,0.15);border-radius:12px;padding:20px;text-align:center;">'
            f'<div class="forecast-year" style="color:rgba(248,245,240,0.5);font-size:13px;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">{fc["label"]}</div>'
            f'<div class="forecast-val" style="font-family:Playfair Display,serif;font-size:32px;color:#D7A24E;font-weight:700;">${fc["value"]:,.0f}</div>'
            f'<div style="color:#4CAF50;font-size:14px;font-weight:600;margin-top:8px;">+${fc["gain"]:,.0f} (+{fc["gain_pct"]}%)</div>'
            f'</div>'
        )
    fc_html += '</div>'
    st.markdown(fc_html, unsafe_allow_html=True)
    st.caption(forecast["disclaimer"])

    # ── INVESTMENT RISK ANALYSIS ──
    st.markdown('<div class="section-divider" style="max-width:1450px;margin:110px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(215,162,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    st.caption("Risk Assessment")
    st.subheader("Investment Risk Analysis")
    st.markdown(f'<p style="color:#B5AFA8;">{risk_analysis["risk_desc"]}</p>', unsafe_allow_html=True)
    sev_colors = {"high": "#E57373", "moderate": "#FFD060", "low": "#4CAF50"}

    risk_header_l, risk_header_r = st.columns([1, 2])
    with risk_header_l:
        st.markdown(
            f'<div class="premium-card" style="text-align:center;">'
            f'<div style="display:inline-block;padding:8px 20px;border-radius:100px;font-size:14px;font-weight:700;border:1px solid {risk_analysis["risk_color"]};color:{risk_analysis["risk_color"]};">{risk_analysis["risk_level"]} Risk</div>'
            f'<div style="margin-top:12px;color:#B5AFA8;font-size:13px;">Protection Score: <strong style="color:#FFFFFF;">{risk_analysis["protection_score"]}/100</strong></div></div>',
            unsafe_allow_html=True,
        )
    with risk_header_r:
        for rf in risk_analysis["factors"]:
            sev_color = sev_colors.get(rf["severity"], "#FFD060")
            st.markdown(
                f'<div class="premium-card" style="margin-bottom:8px;padding:14px 18px;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<div><strong style="color:#FFFFFF;">{rf["icon"]} {rf["factor"]}</strong><br>'
                f'<span style="color:#B5AFA8;font-size:13px;">{rf["detail"]}</span></div>'
                f'<span style="padding:4px 12px;border-radius:100px;font-size:11px;font-weight:700;border:1px solid {sev_color};color:{sev_color};">{rf["severity"].upper()}</span></div></div>',
                unsafe_allow_html=True,
            )
        if not risk_analysis["factors"]:
            st.info("No significant risk factors detected. Strong investment profile.")

    # ── TECHNICAL DETAILS ──
    st.markdown('<div class="section-divider" style="max-width:1450px;margin:110px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(215,162,78,0.12),transparent);"></div>', unsafe_allow_html=True)
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
    st.markdown('<div class="section-divider" style="max-width:1450px;margin:110px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(215,162,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    next_actions = generate_next_actions()
    st.caption("Next Steps")
    st.subheader("What Would You Like To Do?")
    for a in next_actions:
        st.markdown(
            f'<div class="premium-card" style="margin-bottom:8px;">'
            f'<strong style="color:#FFFFFF;">{a["icon"]} {a["title"]}</strong><br>'
            f'<span style="color:#F0ECE8;font-size:14px;">{a["desc"]}</span></div>',
            unsafe_allow_html=True,
        )

    # ── DOWNLOAD REPORT ──
    st.markdown('<div class="section-divider" style="max-width:1450px;margin:110px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(215,162,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    from utils.pdf_report import generate_pdf_report

    try:
        pdf_bytes = generate_pdf_report(inputs, prediction, quality_score, ai_summary, scores, forecast, risk_analysis)
    except Exception:
        pdf_bytes = None

    st.caption("Export")
    st.subheader("Download Report")
    st.markdown('<p style="color:#B5AFA8;">Export this valuation as a professional PDF report.</p>', unsafe_allow_html=True)

    dl_col1, dl_col2, dl_col3 = st.columns([1, 2, 1])
    with dl_col2:
        if pdf_bytes:
            st.download_button(label="Download PDF Report", data=pdf_bytes, file_name="HomeSense_Valuation_Report.pdf", mime="application/pdf", width="stretch")
        else:
            st.info("PDF report is temporarily unavailable. Please try again later.")

    # ── RECENT PREDICTIONS ──
    st.markdown('<div class="section-divider" style="max-width:1450px;margin:110px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(215,162,78,0.12),transparent);"></div>', unsafe_allow_html=True)
    history = get_history()
    if history:
        st.caption("History")
        st.subheader("Recent Predictions")
        for h in history[-5:]:
            st.markdown(
                f'<div class="premium-card" style="margin-bottom:8px;padding:14px 18px;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<div><span style="color:rgba(248,245,240,0.4);font-size:12px;">{h["timestamp"]}</span><br>'
                f'<span style="color:#F0ECE8;font-size:13px;">Q{h["quality"]} | {h["area"]:,} sq ft | {h["neighborhood"]}</span></div>'
                f'<div style="text-align:right;"><div style="font-family:Playfair Display,serif;font-size:20px;color:#D7A24E;font-weight:700;">${h["price"]:,.0f}</div>'
                f'<div style="color:{get_score_color(h["investment_score"])};font-weight:600;font-size:13px;">{h["investment_score"]}/100</div></div></div></div>',
                unsafe_allow_html=True,
            )

# ============================================================
# ABOUT
# ============================================================
st.markdown('<div class="section-divider" style="max-width:1450px;margin:110px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(215,162,78,0.12),transparent);"></div>', unsafe_allow_html=True)

st.markdown('<div id="about"></div>', unsafe_allow_html=True)
st.caption("About")
st.subheader("About HomeSense AI")

about_l, about_r = st.columns([2, 1])
with about_l:
    st.markdown("**Machine Learning Meets Real Estate**")
    st.markdown(
        "HomeSense AI uses a Linear Regression model trained on the Ames Housing Dataset "
        "with 1,460 property records and 245 encoded features. Every prediction includes full "
        "explainability - no black boxes."
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
st.markdown('<div class="section-divider" style="max-width:1450px;margin:110px auto;height:1px;background:linear-gradient(90deg,transparent,rgba(215,162,78,0.20),transparent);"></div>', unsafe_allow_html=True)
st.markdown(
    '<div style="background:rgba(18,10,6,0.80);backdrop-filter:blur(32px);-webkit-backdrop-filter:blur(32px);'
    'border-top:1px solid rgba(215,162,78,0.08);margin:0 -60px;padding:60px 60px 32px;overflow:hidden;">'
    '<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:32px;">'

    # Col 1 - Brand
    '<div>'
    '<div style="color:#FFFFFF;font-size:18px;font-weight:700;margin-bottom:12px;">HomeSense AI</div>'
    '<div style="color:#B5AFA8;font-size:15px;line-height:1.8;">Luxury AI Real Estate Valuation<br>Built using</div>'
    '<div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:12px;">'
    '<span style="padding:6px 14px;background:rgba(215,162,78,0.1);border:1px solid rgba(215,162,78,0.2);border-radius:8px;color:#D7A24E;font-size:13px;font-weight:600;">Python</span>'
    '<span style="padding:6px 14px;background:rgba(215,162,78,0.1);border:1px solid rgba(215,162,78,0.2);border-radius:8px;color:#D7A24E;font-size:13px;font-weight:600;">Streamlit</span>'
    '<span style="padding:6px 14px;background:rgba(215,162,78,0.1);border:1px solid rgba(215,162,78,0.2);border-radius:8px;color:#D7A24E;font-size:13px;font-weight:600;">Scikit-learn</span>'
    '<span style="padding:6px 14px;background:rgba(215,162,78,0.1);border:1px solid rgba(215,162,78,0.2);border-radius:8px;color:#D7A24E;font-size:13px;font-weight:600;">Plotly</span>'
    '<span style="padding:6px 14px;background:rgba(215,162,78,0.1);border:1px solid rgba(215,162,78,0.2);border-radius:8px;color:#D7A24E;font-size:13px;font-weight:600;">Linear Regression</span>'
    '<span style="padding:6px 14px;background:rgba(215,162,78,0.1);border:1px solid rgba(215,162,78,0.2);border-radius:8px;color:#D7A24E;font-size:13px;font-weight:600;">Explainable AI</span>'
    '<span style="padding:6px 14px;background:rgba(215,162,78,0.1);border:1px solid rgba(215,162,78,0.2);border-radius:8px;color:#D7A24E;font-size:13px;font-weight:600;">Ames Housing Dataset</span>'
    '</div></div>'

    # Col 2+3 - Metrics and Contact side by side
    '<div style="display:flex;gap:40px;">'
    # Metrics
    '<div>'
    '<div style="color:#FFFFFF;font-size:18px;font-weight:700;margin-bottom:12px;">Metrics</div>'
    '<div style="color:#FFFFFF;font-size:15px;line-height:2;font-weight:600;">'
    'R\u00b2 = 92.07%<br>MAE: $27,773<br>RMSE: $64,802</div></div>'
    # Contact
    '<div>'
    '<div style="color:#FFFFFF;font-size:18px;font-weight:700;margin-bottom:12px;">Contact Us</div>'
    '<div style="color:#FFFFFF;font-size:15px;line-height:2;font-weight:600;">'
    '<a href="https://github.com" style="color:#D7A24E;text-decoration:none;">GitHub</a><br>'
    '<a href="https://linkedin.com" style="color:#D7A24E;text-decoration:none;">LinkedIn</a><br>'
    '<a href="mailto:bobby2992006@gmail.com" style="color:#D7A24E;text-decoration:none;">Email</a></div></div>'
    '</div>'

    '</div>'
    '<div style="border-top:1px solid rgba(215,162,78,0.08);margin-top:32px;padding-top:20px;'
    'display:flex;justify-content:space-between;align-items:center;">'
    '<span style="color:#B5AFA8;font-size:14px;">2026 HomeSense AI. All rights reserved.</span>'
    '<span style="color:#B5AFA8;font-size:13px;">Version 1.0 | R\u00b2 = 92.07%</span></div></div>',
    unsafe_allow_html=True,
)
