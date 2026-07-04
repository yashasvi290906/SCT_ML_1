import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from functools import lru_cache

CURRENT_YEAR = datetime.now().year

# ============================================================
# FEATURE METADATA
# ============================================================

QUALITY_MAP = {
    1: "Very Poor",
    2: "Poor",
    3: "Fair",
    4: "Below Average",
    5: "Average",
    6: "Above Average",
    7: "Good",
    8: "Very Good",
    9: "Excellent",
    10: "Very Excellent",
}

NEIGHBORHOODS = [
    "Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr",
    "CollgCr", "Crawfor", "Edwards", "Gilbert", "IDOTRR",
    "MeadowV", "Mitchel", "NAmes", "NoRidge", "NPkVill",
    "NridgHt", "NWAmes", "OldTown", "SWISU", "Sawyer",
    "SawyerW", "Somerst", "StoneBr", "Timber", "Veenker",
]

NEIGHBORHOOD_LABELS = {
    "Blmngtn": "Bloomington Heights",
    "Blueste": "Bluestem",
    "BrDale": "Briardale",
    "BrkSide": "Brookside",
    "ClearCr": "Clear Creek",
    "CollgCr": "College Creek",
    "Crawfor": "Crawford",
    "Edwards": "Edwards",
    "Gilbert": "Gilbert",
    "IDOTRR": "Iowa DOT & Rail Road",
    "MeadowV": "Meadow Village",
    "Mitchel": "Mitchell",
    "NAmes": "North Ames",
    "NoRidge": "Northridge",
    "NPkVill": "Northpark Villa",
    "NridgHt": "Northridge Heights",
    "NWAmes": "Northwest Ames",
    "OldTown": "Old Town",
    "SWISU": "South & West Iowa State",
    "Sawyer": "Sawyer",
    "SawyerW": "Sawyer West",
    "Somerst": "Somerset",
    "StoneBr": "Stone Brook",
    "Timber": "Timberland",
    "Veenker": "Veenker",
}

# Actual dataset averages computed from training data
DATASET_AVERAGES = {
    "OverallQual": 6.1,
    "GrLivArea": 1515.5,
    "GarageCars": 1.8,
    "TotalBsmtSF": 1057.4,
    "FullBath": 1.6,
    "HalfBath": 0.4,
    "BedroomAbvGr": 2.9,
    "YearBuilt": 1971.3,
    "SalePrice": 180921.2,
}

NEIGHBORHOOD_STATS = {
    "Blmngtn": {"avg_price": 194871, "tier": "Mid", "demand": "Steady", "buyers": "Young Professionals", "trend": "Stable"},
    "Blueste": {"avg_price": 137500, "tier": "Value", "demand": "Limited", "buyers": "First-Time Buyers", "trend": "Stable"},
    "BrDale": {"avg_price": 104494, "tier": "Value", "demand": "Moderate", "buyers": "Budget-Conscious", "trend": "Stable"},
    "BrkSide": {"avg_price": 124834, "tier": "Value", "demand": "Moderate", "buyers": "Families", "trend": "Stable"},
    "ClearCr": {"avg_price": 212565, "tier": "Mid-High", "demand": "Strong", "buyers": "Established Families", "trend": "Appreciating"},
    "CollgCr": {"avg_price": 197966, "tier": "Mid", "demand": "Strong", "buyers": "Young Professionals", "trend": "Appreciating"},
    "Crawfor": {"avg_price": 210625, "tier": "Mid-High", "demand": "Strong", "buyers": "Established Families", "trend": "Appreciating"},
    "Edwards": {"avg_price": 128220, "tier": "Value", "demand": "Moderate", "buyers": "First-Time Buyers", "trend": "Stable"},
    "Gilbert": {"avg_price": 192855, "tier": "Mid", "demand": "Strong", "buyers": "Young Professionals", "trend": "Appreciating"},
    "IDOTRR": {"avg_price": 100124, "tier": "Value", "demand": "Limited", "buyers": "Budget-Conscious", "trend": "Stable"},
    "MeadowV": {"avg_price": 98576, "tier": "Value", "demand": "Limited", "buyers": "First-Time Buyers", "trend": "Stable"},
    "Mitchel": {"avg_price": 156270, "tier": "Mid", "demand": "Moderate", "buyers": "Families", "trend": "Stable"},
    "NAmes": {"avg_price": 145847, "tier": "Mid", "demand": "Strong", "buyers": "Families", "trend": "Stable"},
    "NPkVill": {"avg_price": 142694, "tier": "Mid", "demand": "Moderate", "buyers": "Young Professionals", "trend": "Stable"},
    "NWAmes": {"avg_price": 189050, "tier": "Mid", "demand": "Strong", "buyers": "Established Families", "trend": "Appreciating"},
    "NoRidge": {"avg_price": 335295, "tier": "Premium", "demand": "Very High", "buyers": "High-Income Professionals", "trend": "Strong Appreciation"},
    "NridgHt": {"avg_price": 316271, "tier": "Premium", "demand": "Very High", "buyers": "High-Income Professionals", "trend": "Strong Appreciation"},
    "OldTown": {"avg_price": 128225, "tier": "Value", "demand": "Moderate", "buyers": "First-Time Buyers", "trend": "Stable"},
    "SWISU": {"avg_price": 142591, "tier": "Mid", "demand": "Moderate", "buyers": "Students & Faculty", "trend": "Stable"},
    "Sawyer": {"avg_price": 136793, "tier": "Mid", "demand": "Moderate", "buyers": "Families", "trend": "Stable"},
    "SawyerW": {"avg_price": 186556, "tier": "Mid", "demand": "Strong", "buyers": "Young Professionals", "trend": "Appreciating"},
    "Somerst": {"avg_price": 225380, "tier": "Mid-High", "demand": "Very High", "buyers": "High-Income Professionals", "trend": "Strong Appreciation"},
    "StoneBr": {"avg_price": 310499, "tier": "Premium", "demand": "Very High", "buyers": "High-Income Professionals", "trend": "Strong Appreciation"},
    "Timber": {"avg_price": 242247, "tier": "High", "demand": "High", "buyers": "Established Families", "trend": "Appreciating"},
    "Veenker": {"avg_price": 238773, "tier": "High", "demand": "High", "buyers": "High-Income Professionals", "trend": "Appreciating"},
}


def get_quality_label(score: int) -> str:
    return QUALITY_MAP.get(score, "Unknown")


def classify_price(price: float) -> str:
    if price >= 350000:
        return "premium"
    elif price >= 200000:
        return "mid"
    else:
        return "value"


def get_price_badge(price: float) -> tuple[str, str]:
    category = classify_price(price)
    badges = {
        "premium": ("Premium Investment", "badge-green"),
        "mid": ("Strong Value", "badge-yellow"),
        "value": ("Affordable Entry", "badge-red"),
    }
    return badges[category]


def compute_quality_score(inputs: dict) -> float:
    weights = {
        "OverallQual": 0.30,
        "GrLivArea_norm": 0.18,
        "GarageCars_norm": 0.12,
        "TotalBsmtSF_norm": 0.12,
        "FullBath_norm": 0.10,
        "YearBuilt_norm": 0.10,
        "BedroomAbvGr_norm": 0.08,
    }
    quality = inputs.get("OverallQual", 5) / 10.0
    living = min(inputs.get("GrLivArea", 1500) / 4000.0, 1.0)
    garage = min(inputs.get("GarageCars", 2) / 4.0, 1.0)
    basement = min(inputs.get("TotalBsmtSF", 800) / 2500.0, 1.0)
    bath = min(inputs.get("FullBath", 2) / 4.0, 1.0)
    year = max((inputs.get("YearBuilt", 2005) - 1950) / 75.0, 0)
    bedroom = min(inputs.get("BedroomAbvGr", 3) / 6.0, 1.0)

    score = (
        weights["OverallQual"] * quality
        + weights["GrLivArea_norm"] * living
        + weights["GarageCars_norm"] * garage
        + weights["TotalBsmtSF_norm"] * basement
        + weights["FullBath_norm"] * bath
        + weights["YearBuilt_norm"] * year
        + weights["BedroomAbvGr_norm"] * bedroom
    )
    return round(min(score, 1.0) * 100, 1)


# ============================================================
# INVESTMENT SCORES
# ============================================================

def compute_investment_scores(inputs: dict, prediction: float) -> dict:
    oq = inputs.get("OverallQual", 5)
    yb = inputs.get("YearBuilt", 2005)
    gla = inputs.get("GrLivArea", 1500)
    gc = inputs.get("GarageCars", 2)
    tbs = inputs.get("TotalBsmtSF", 800)
    fb = inputs.get("FullBath", 2)
    ba = inputs.get("BedroomAbvGr", 3)
    age = CURRENT_YEAR - yb

    investment = min(100, max(0, (oq / 10) * 35 + max(0, (1 - age / 50)) * 25 + min(gla / 3000, 1) * 20 + min(gc / 3, 1) * 10 + (10 if prediction >= 200000 else 0)))
    rental = min(100, max(0, (min(ba / 3, 1) * 25 + min(fb / 2, 1) * 20 + min(gla / 2500, 1) * 20 + (15 if gc >= 2 else 5) + max(0, (1 - age / 40)) * 20)))
    resale = min(100, max(0, (oq / 10) * 30 + min(gla / 2500, 1) * 25 + max(0, (1 - age / 60)) * 20 + min(gc / 3, 1) * 15 + (10 if tbs > 0 else 0)))
    luxury = min(100, max(0, (oq / 10) * 40 + min(gla / 4000, 1) * 25 + min(gc / 4, 1) * 15 + max(0, (1 - age / 50)) * 10 + (10 if prediction >= 300000 else 0)))
    risk = min(100, max(0, 100 - ((10 - oq) / 10) * 30 - min(age / 80, 1) * 25 - (15 if gc == 0 else 0) - (10 if tbs == 0 else 0)))

    return {
        "investment": round(investment),
        "rental": round(rental),
        "resale": round(resale),
        "luxury": round(luxury),
        "risk": round(risk),
    }


def get_score_color(score: float) -> str:
    if score >= 75:
        return "#4ECB8D"
    if score >= 50:
        return "#D4A84E"
    if score >= 30:
        return "#FFD060"
    return "#FF6B6B"


def get_score_label(score: float) -> str:
    if score >= 80: return "Excellent"
    if score >= 60: return "Strong"
    if score >= 40: return "Moderate"
    if score >= 20: return "Developing"
    return "Limited"


# ============================================================
# MARKET POSITION
# ============================================================

def get_market_position(prediction: float) -> dict:
    min_price = 34900
    max_price = 755000
    avg_price = 180921
    pctile = (prediction - min_price) / (max_price - min_price) * 100
    pctile = max(0, min(100, pctile))

    if prediction >= 326100:
        tier = "Luxury"
        tier_desc = "Above 95th percentile of market"
    elif prediction >= 278000:
        tier = "Premium"
        tier_desc = "Above 90th percentile of market"
    elif prediction >= 214000:
        tier = "Above Average"
        tier_desc = "Above 75th percentile"
    elif prediction >= 163000:
        tier = "Average"
        tier_desc = "Near median price"
    elif prediction >= 106475:
        tier = "Value"
        tier_desc = "Below median, good entry point"
    else:
        tier = "Entry-Level"
        tier_desc = "Below 10th percentile"

    return {
        "percentile": round(pctile),
        "tier": tier,
        "tier_desc": tier_desc,
        "vs_avg": prediction - avg_price,
        "vs_avg_pct": ((prediction - avg_price) / avg_price) * 100,
    }


# ============================================================
# EXPLAINABLE AI
# ============================================================

def generate_explanations(inputs: dict, prediction: float) -> list[dict]:
    insights = []

    oq = inputs.get("OverallQual", 5)
    if oq >= 8:
        insights.append({
            "type": "positive", "icon": "\u2b50",
            "title": f"Outstanding Quality ({oq}/10)",
            "description": "Exceptional materials and finish significantly increased the property valuation.",
            "contribution": "+12-18%",
        })
    elif oq >= 6:
        insights.append({
            "type": "positive", "icon": "\u2191",
            "title": f"Good Quality ({oq}/10)",
            "description": "Above-average construction quality contributed positively to the estimated price.",
            "contribution": "+6-12%",
        })
    elif oq <= 4:
        insights.append({
            "type": "negative", "icon": "\u2193",
            "title": f"Below-Average Quality ({oq}/10)",
            "description": "Lower construction quality reduced the overall valuation.",
            "contribution": "-8-15%",
        })
    else:
        insights.append({
            "type": "neutral", "icon": "\u2192",
            "title": f"Average Quality ({oq}/10)",
            "description": "Standard construction quality has a neutral impact on pricing.",
            "contribution": "Baseline",
        })

    gla = inputs.get("GrLivArea", 1500)
    if gla >= 2500:
        insights.append({
            "type": "positive", "icon": "\u25b2",
            "title": f"{gla:,} sq ft Living Area",
            "description": "Spacious living space is a major driver of premium valuations.",
            "contribution": "+10-15%",
        })
    elif gla >= 1800:
        insights.append({
            "type": "positive", "icon": "\u25b2",
            "title": f"{gla:,} sq ft Living Area",
            "description": "Generous living area contributed positively to the price.",
            "contribution": "+5-10%",
        })
    elif gla < 1000:
        insights.append({
            "type": "negative", "icon": "\u25bc",
            "title": f"{gla:,} sq ft Living Area",
            "description": "Compact living space may limit resale potential.",
            "contribution": "-5-10%",
        })
    else:
        insights.append({
            "type": "neutral", "icon": "\u2500",
            "title": f"{gla:,} sq ft Living Area",
            "description": "Moderate living area has a balanced effect on valuation.",
            "contribution": "Baseline",
        })

    gc = inputs.get("GarageCars", 2)
    if gc >= 3:
        insights.append({
            "type": "positive", "icon": "\u2713",
            "title": f"{gc}-Car Garage",
            "description": "Large garage capacity enhances convenience and adds tangible value.",
            "contribution": "+4-8%",
        })
    elif gc == 0:
        insights.append({
            "type": "negative", "icon": "\u2717",
            "title": "No Garage",
            "description": "Absence of parking reduces overall property attractiveness.",
            "contribution": "-6-10%",
        })

    tbs = inputs.get("TotalBsmtSF", 800)
    if tbs >= 1500:
        insights.append({
            "type": "positive", "icon": "\u25b2",
            "title": f"{tbs:,} sq ft Basement",
            "description": "Expansive basement adds significant usable space and value.",
            "contribution": "+5-10%",
        })
    elif tbs == 0:
        insights.append({
            "type": "negative", "icon": "\u25bc",
            "title": "No Basement",
            "description": "Property lacks basement space, reducing overall square footage.",
            "contribution": "-4-8%",
        })

    fb = inputs.get("FullBath", 2)
    hb = inputs.get("HalfBath", 0)
    total_bath = fb + hb * 0.5
    if total_bath >= 3:
        insights.append({
            "type": "positive", "icon": "\u2713",
            "title": f"{int(fb)} Full + {int(hb)} Half Baths",
            "description": "Ample bathroom count increases daily comfort and market appeal.",
            "contribution": "+3-6%",
        })

    yb = inputs.get("YearBuilt", 2005)
    age = CURRENT_YEAR - yb
    if age <= 10:
        insights.append({
            "type": "positive", "icon": "\u2b50",
            "title": f"Built in {yb}",
            "description": f"Recent construction ({age} years old) commands a premium for modern standards.",
            "contribution": "+6-10%",
        })
    elif age >= 60:
        insights.append({
            "type": "neutral", "icon": "\u2192",
            "title": f"Built in {yb}",
            "description": f"Older construction ({age} years) may require updates but offers classic character.",
            "contribution": "Variable",
        })

    ba = inputs.get("BedroomAbvGr", 3)
    if ba >= 5:
        insights.append({
            "type": "positive", "icon": "\u2713",
            "title": f"{ba} Bedrooms",
            "description": "Large bedroom count makes this property attractive for families.",
            "contribution": "+3-5%",
        })
    elif ba <= 1:
        insights.append({
            "type": "negative", "icon": "\u2717",
            "title": f"{ba} Bedroom(s)",
            "description": "Limited bedroom count restricts target buyer pool.",
            "contribution": "-4-7%",
        })

    return insights


# ============================================================
# EXECUTIVE AI SUMMARY
# ============================================================

def generate_premium_summary(inputs: dict, prediction: float, scores: dict, market: dict) -> str:
    oq = inputs.get("OverallQual", 5)
    gla = inputs.get("GrLivArea", 1500)
    gc = inputs.get("GarageCars", 2)
    tbs = inputs.get("TotalBsmtSF", 800)
    yb = inputs.get("YearBuilt", 2005)
    fb = inputs.get("FullBath", 2)
    hb = inputs.get("HalfBath", 0)
    ba = inputs.get("BedroomAbvGr", 3)
    nb = inputs.get("Neighborhood", "CollgCr")
    age = CURRENT_YEAR - yb
    nb_name = NEIGHBORHOOD_LABELS.get(nb, nb)

    sentences = []

    if prediction >= 350000:
        sentences.append(f"This property demonstrates characteristics commonly associated with premium homes in the Ames market, with an estimated value of ${prediction:,.0f} placing it in the {market['tier']} segment.")
    elif prediction >= 200000:
        sentences.append(f"This property presents strong market positioning with an estimated value of ${prediction:,.0f}, positioning it in the {market['tier']} segment of the Ames housing market.")
    else:
        sentences.append(f"This property offers an accessible entry point into the Ames market at an estimated ${prediction:,.0f}, falling within the {market['tier']} segment.")

    if oq >= 7:
        sentences.append(f"Exceptional construction quality ({oq}/10) serves as the primary value driver, indicating superior materials, craftsmanship, and finish standards.")
    elif oq <= 4:
        sentences.append(f"Below-average construction quality ({oq}/10) represents the most significant factor limiting the valuation ceiling.")

    if gla >= 2000:
        sentences.append(f"The generous {gla:,} sq ft of living space exceeds the dataset average of {DATASET_AVERAGES['GrLivArea']:,.0f} sq ft, providing above-average room for comfortable living.")
    if gc >= 3:
        sentences.append(f"A {gc}-car garage adds practical convenience and enhances resale appeal in this market segment.")
    if tbs >= 1200:
        sentences.append(f"The {tbs:,} sq ft basement provides substantial additional usable area beyond the living quarters.")
    if age <= 15:
        sentences.append(f"Built {age} years ago, the relatively modern construction benefits from current building standards and reduced maintenance concerns.")
    elif age >= 50:
        sentences.append(f"At {age} years old, the property carries classic character though may benefit from targeted updates to maximize value.")

    total_bath = fb + hb * 0.5
    if total_bath >= 3:
        sentences.append(f"With {int(fb)} full and {int(hb)} half bathrooms, the property meets the demands of modern family living.")

    sentences.append(f"Situated in {nb_name}, the property benefits from established neighborhood infrastructure and community amenities.")

    return " ".join(sentences)


# ============================================================
# RECOMMENDATIONS
# ============================================================

def generate_recommendations(inputs: dict, prediction: float) -> list[dict]:
    recs = []
    oq = inputs.get("OverallQual", 5)
    yb = inputs.get("YearBuilt", 2005)
    gla = inputs.get("GrLivArea", 1500)
    gc = inputs.get("GarageCars", 2)
    tbs = inputs.get("TotalBsmtSF", 800)
    fb = inputs.get("FullBath", 2)
    age = CURRENT_YEAR - yb

    if oq <= 6:
        recs.append({
            "icon": "\u2728",
            "title": "Upgrade Kitchen & Bath Finishes",
            "description": "Investing in high-quality countertops, cabinetry, and fixtures could elevate the overall quality rating.",
            "impact": "Estimated +8-15% value increase",
        })
    if tbs < 800:
        recs.append({
            "icon": "\u2693",
            "title": "Finish or Expand Basement",
            "description": "Converting unfinished basement space to livable square footage provides strong return on investment.",
            "impact": "Estimated +5-10% value increase",
        })
    if gc < 2:
        recs.append({
            "icon": "\U0001F9F7",
            "title": "Garage Modernization",
            "description": "Adding or expanding garage capacity improves convenience and aligns with buyer expectations in this price range.",
            "impact": "Estimated +4-8% value increase",
        })
    if age >= 25:
        recs.append({
            "icon": "\u2728",
            "title": "Exterior & Curb Appeal",
            "description": "Fresh paint, updated landscaping, and modern entry fixtures create strong first impressions.",
            "impact": "Estimated +3-6% value increase",
        })
    if fb < 2:
        recs.append({
            "icon": "\U0001F6BF",
            "title": "Add Bathroom",
            "description": "Properties with fewer than 2 full baths often leave value on the table. Adding a bathroom is a high-ROI improvement.",
            "impact": "Estimated +4-8% value increase",
        })
    if gla < 1500:
        recs.append({
            "icon": "\u2B1A",
            "title": "Expand Living Space",
            "description": "Adding room through extension or finishing attic space can significantly increase per-square-foot valuation.",
            "impact": "Estimated +6-12% value increase",
        })
    if not recs:
        recs.append({
            "icon": "\u2705",
            "title": "Property is Well-Optimized",
            "description": "This property already scores well across key metrics. Focus on maintenance and minor cosmetic updates.",
            "impact": "Preserve current valuation",
        })
    return recs


# ============================================================
# NEIGHBORHOOD INSIGHTS
# ============================================================

def get_neighborhood_insights(neighborhood: str) -> dict:
    stats = NEIGHBORHOOD_STATS.get(neighborhood, {
        "avg_price": 180921, "tier": "Mid", "demand": "Moderate",
        "buyers": "Mixed Demographics", "trend": "Stable"
    })
    return {
        "name": NEIGHBORHOOD_LABELS.get(neighborhood, neighborhood),
        "avg_price": stats["avg_price"],
        "tier": stats["tier"],
        "demand": stats["demand"],
        "buyers": stats["buyers"],
        "trend": stats["trend"],
    }


# ============================================================
# COMPARISON
# ============================================================

def compute_comparison(inputs: dict) -> list[dict]:
    items = [
        ("Living Area", inputs.get("GrLivArea", 1500), DATASET_AVERAGES["GrLivArea"], "sq ft"),
        ("Garage Capacity", inputs.get("GarageCars", 2), DATASET_AVERAGES["GarageCars"], "cars"),
        ("Full Bathrooms", inputs.get("FullBath", 2), DATASET_AVERAGES["FullBath"], ""),
        ("Overall Quality", inputs.get("OverallQual", 5), DATASET_AVERAGES["OverallQual"], "/10"),
        ("Basement Area", inputs.get("TotalBsmtSF", 800), DATASET_AVERAGES["TotalBsmtSF"], "sq ft"),
        ("Year Built", inputs.get("YearBuilt", 2005), DATASET_AVERAGES["YearBuilt"], ""),
        ("Bedrooms", inputs.get("BedroomAbvGr", 3), DATASET_AVERAGES["BedroomAbvGr"], ""),
    ]
    result = []
    for name, yours, avg, unit in items:
        diff = yours - avg
        pct = (diff / avg * 100) if avg != 0 else 0
        result.append({
            "name": name,
            "yours": yours,
            "average": round(avg, 1),
            "difference": round(diff, 1),
            "pct": round(pct, 1),
            "unit": unit,
            "better": diff > 0,
        })
    return result


# ============================================================
# PREDICTION HISTORY
# ============================================================

def add_to_history(inputs: dict, prediction: float, scores: dict):
    import streamlit as st
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    entry = {
        "timestamp": datetime.now().strftime("%b %d, %Y %I:%M %p"),
        "price": prediction,
        "quality": inputs.get("OverallQual", 5),
        "area": inputs.get("GrLivArea", 1500),
        "neighborhood": NEIGHBORHOOD_LABELS.get(inputs.get("Neighborhood", ""), inputs.get("Neighborhood", "")),
        "investment_score": scores.get("investment", 0),
    }
    st.session_state.prediction_history.insert(0, entry)
    if len(st.session_state.prediction_history) > 10:
        st.session_state.prediction_history = st.session_state.prediction_history[:10]


def get_history() -> list[dict]:
    import streamlit as st
    return st.session_state.get("prediction_history", [])


# ============================================================
# CHARTS
# ============================================================

CHART_COLORS = {
    "primary": "#D4A84E",
    "secondary": "#F2BF67",
    "bg": "#1A0E0A",
    "card": "#261A14",
    "text": "#FAF2EB",
    "text_dim": "rgba(245,240,235,0.5)",
    "positive": "#4ECB8D",
    "negative": "#FF6B6B",
    "neutral": "#FFD060",
    "grid": "rgba(255,255,255,0.05)",
}

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=CHART_COLORS["text"], family="Inter, sans-serif", size=13),
    margin=dict(l=20, r=20, t=50, b=30),
    xaxis=dict(gridcolor=CHART_COLORS["grid"], zeroline=False),
    yaxis=dict(gridcolor=CHART_COLORS["grid"], zeroline=False),
)


@lru_cache(maxsize=1)
def _compute_feature_importance():
    import joblib
    try:
        model = joblib.load("model/house_price_model.pkl")
        columns = joblib.load("model/model_columns.pkl")
        df = pd.read_csv("data/train.csv")
        key_features = {
            "GrLivArea": "Living Area",
            "OverallQual": "Overall Quality",
            "YearBuilt": "Year Built",
            "GarageCars": "Garage Size",
            "TotalBsmtSF": "Basement",
            "FullBath": "Bathrooms",
            "BedroomAbvGr": "Bedrooms",
            "LotArea": "Lot Size",
        }
        coefs = dict(zip(columns, model.coef_))
        stds = {}
        for col in key_features:
            if col in df.columns:
                stds[col] = float(df[col].std())
            else:
                stds[col] = 1.0
        importance = {}
        for col, label in key_features.items():
            coef = coefs.get(col, 0)
            std = stds.get(col, 1)
            importance[label] = abs(float(coef)) * std
        total = sum(importance.values()) or 1
        result = {k: v / total for k, v in sorted(importance.items(), key=lambda x: -x[1])}
        return list(result.keys()), list(result.values())
    except Exception:
        return (
            ["Living Area", "Overall Quality", "Year Built", "Garage Size", "Basement", "Bathrooms", "Bedrooms", "Lot Size"],
            [0.34, 0.21, 0.13, 0.13, 0.06, 0.06, 0.04, 0.03],
        )


def create_feature_importance_chart() -> go.Figure:
    feature_names, importance = _compute_feature_importance()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=importance,
        y=feature_names,
        orientation="h",
        marker=dict(
            color=importance,
            colorscale=[[0, "rgba(212,168,78,0.25)"], [1, "#D4A84E"]],
            cornerradius=6,
        ),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.0%}<extra></extra>",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        height=380,
        showlegend=False,
        title=dict(text="Feature Contribution to Valuation", font=dict(size=16, color="#FAF2EB"), x=0, xanchor="left"),
    )
    fig.update_xaxes(title_text="Relative Importance", tickformat=".0%")
    return fig


def create_input_summary_chart(inputs: dict) -> go.Figure:
    labels = ["Quality", "Living Area", "Garage", "Basement", "Bathrooms", "Bedrooms", "Year Built"]
    max_vals = [10, 4000, 5, 3000, 5, 8, CURRENT_YEAR]
    raw_vals = [
        inputs.get("OverallQual", 5),
        inputs.get("GrLivArea", 1500),
        inputs.get("GarageCars", 2),
        inputs.get("TotalBsmtSF", 800),
        inputs.get("FullBath", 2),
        inputs.get("BedroomAbvGr", 3),
        inputs.get("YearBuilt", 2005),
    ]
    normalized = [v / m for v, m in zip(raw_vals, max_vals)]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=normalized + [normalized[0]],
        theta=labels + [labels[0]],
        fill="toself",
        fillcolor="rgba(212,168,78,0.12)",
        line=dict(color="#D4A84E", width=2),
        marker=dict(size=8, color="#D4A84E"),
        hovertemplate="<b>%{theta}</b><br>Score: %{r:.0%}<extra></extra>",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        height=400,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=CHART_COLORS["grid"], tickformat=".0%"),
            angularaxis=dict(gridcolor=CHART_COLORS["grid"]),
        ),
        showlegend=False,
        title=dict(text="Property Profile Radar", font=dict(size=16, color="#FAF2EB"), x=0, xanchor="left"),
    )
    return fig


def create_price_comparison_chart(prediction: float) -> go.Figure:
    categories = ["P10", "P25", "Median", "Your Home", "P75", "P90"]
    avg_prices = [106475, 129975, 163000, prediction, 214000, 278000]
    colors = [
        CHART_COLORS["text_dim"],
        CHART_COLORS["text_dim"],
        CHART_COLORS["text_dim"],
        CHART_COLORS["primary"],
        CHART_COLORS["text_dim"],
        CHART_COLORS["text_dim"],
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=categories,
        y=avg_prices,
        marker=dict(color=colors, cornerradius=8),
        text=[f"${p:,.0f}" for p in avg_prices],
        textposition="outside",
        textfont=dict(color=CHART_COLORS["text"], size=12),
        hovertemplate="<b>%{x}</b><br>$%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        height=360,
        title=dict(text="Price Distribution (Dataset Percentiles)", font=dict(size=16, color="#FAF2EB"), x=0, xanchor="left"),
    )
    fig.update_yaxes(tickformat="$,.0f")
    fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
    return fig


# ============================================================
# PREDICTION INTERVAL & CONFIDENCE
# ============================================================

MODEL_RMSE_ESTIMATE = 64802.0

CONFIDENCE_FACTORS = {
    "OverallQual": {"optimal": (5, 8), "weight": 0.25},
    "GrLivArea": {"optimal": (1200, 2800), "weight": 0.20},
    "GarageCars": {"optimal": (1, 3), "weight": 0.10},
    "TotalBsmtSF": {"optimal": (400, 1500), "weight": 0.10},
    "YearBuilt": {"optimal": (1980, 2020), "weight": 0.15},
    "FullBath": {"optimal": (1, 3), "weight": 0.10},
    "BedroomAbvGr": {"optimal": (2, 5), "weight": 0.10},
}


def compute_prediction_interval(prediction: float) -> dict:
    rmse = MODEL_RMSE_ESTIMATE
    lower = max(0, prediction - 1.96 * rmse)
    upper = prediction + 1.96 * rmse
    margin = 1.96 * rmse
    return {
        "lower": round(lower),
        "upper": round(upper),
        "margin": round(margin),
        "confidence_level": 95,
        "rmse": rmse,
    }


def compute_confidence(inputs: dict) -> dict:
    total_score = 0.0
    total_weight = 0.0
    factor_details = []
    for feature, config in CONFIDENCE_FACTORS.items():
        value = inputs.get(feature, 0)
        low, high = config["optimal"]
        weight = config["weight"]
        if low <= value <= high:
            factor_score = 1.0
            status = "optimal"
        elif value < low:
            deviation = (low - value) / max(low, 1)
            factor_score = max(0.3, 1.0 - deviation * 0.5)
            status = "below"
        else:
            deviation = (value - high) / max(high, 1)
            factor_score = max(0.3, 1.0 - deviation * 0.3)
            status = "above"
        total_score += factor_score * weight
        total_weight += weight
        factor_details.append({
            "feature": feature,
            "value": value,
            "optimal_range": f"{low}-{high}",
            "score": round(factor_score, 2),
            "status": status,
        })
    confidence = round(total_score / total_weight * 100) if total_weight > 0 else 75
    confidence = max(50, min(98, confidence))
    return {
        "score": confidence,
        "factors": factor_details,
        "grade": "High" if confidence >= 85 else "Moderate" if confidence >= 65 else "Standard",
    }



# ============================================================
# SIMILAR PROPERTIES (NEAREST NEIGHBORS)
# ============================================================

@lru_cache(maxsize=1)
def _load_training_data():
    try:
        df = pd.read_csv("data/train.csv")
        feature_cols = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF",
                        "FullBath", "HalfBath", "BedroomAbvGr", "YearBuilt", "SalePrice"]
        available = [c for c in feature_cols if c in df.columns]
        return df[available].dropna()
    except Exception:
        return pd.DataFrame()


def find_similar_properties(inputs: dict, prediction: float, n: int = 3) -> list[dict]:
    from sklearn.neighbors import NearestNeighbors
    df = _load_training_data()
    if df.empty or len(df) < n:
        return []
    feature_cols = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF",
                    "FullBath", "HalfBath", "BedroomAbvGr", "YearBuilt"]
    available = [c for c in feature_cols if c in df.columns]
    if len(available) < 4:
        return []
    X = df[available].values
    query_vals = []
    for col in available:
        query_vals.append(inputs.get(col, DATASET_AVERAGES.get(col, 0)))
    query = np.array([query_vals])
    scaler_mean = X.mean(axis=0)
    scaler_std = X.std(axis=0)
    scaler_std[scaler_std == 0] = 1
    X_scaled = (X - scaler_mean) / scaler_std
    query_scaled = (query - scaler_mean) / scaler_std
    nn = NearestNeighbors(n_neighbors=min(n, len(X)), metric="euclidean")
    nn.fit(X_scaled)
    distances, indices = nn.kneighbors(query_scaled)
    ref_distance = np.median(np.sqrt(np.sum((X_scaled - X_scaled.mean(axis=0)) ** 2, axis=1)))
    results = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        row = df.iloc[idx]
        raw_sim = max(0, 100 * np.exp(-dist / max(ref_distance, 0.01)))
        similarity = round(min(99, max(5, raw_sim)), 1)
        results.append({
            "rank": i + 1,
            "quality": int(row.get("OverallQual", 0)),
            "area": int(row.get("GrLivArea", 0)),
            "garage": int(row.get("GarageCars", 0)),
            "basement": int(row.get("TotalBsmtSF", 0)),
            "year": int(row.get("YearBuilt", 0)),
            "sale_price": float(row.get("SalePrice", 0)),
            "similarity": similarity,
            "price_diff": round(float(row.get("SalePrice", 0)) - prediction),
        })
    return results


# ============================================================
# APPRECIATION FORECAST
# ============================================================

APPRECIATION_RATES = {
    "Strong Appreciation": {"annual": 0.045, "label": "High Growth"},
    "Appreciating": {"annual": 0.035, "label": "Growing"},
    "Stable": {"annual": 0.025, "label": "Steady"},
    "Limited": {"annual": 0.015, "label": "Slow"},
}


def compute_appreciation_forecast(prediction: float, neighborhood: str, inputs: dict) -> dict:
    stats = NEIGHBORHOOD_STATS.get(neighborhood, {"trend": "Stable", "tier": "Mid"})
    trend = stats.get("trend", "Stable")
    rate_info = APPRECIATION_RATES.get(trend, APPRECIATION_RATES["Stable"])
    annual_rate = rate_info["annual"]
    age = CURRENT_YEAR - inputs.get("YearBuilt", 2005)
    if age <= 10:
        annual_rate *= 1.15
    elif age >= 50:
        annual_rate *= 0.85
    oq = inputs.get("OverallQual", 5)
    if oq >= 8:
        annual_rate *= 1.10
    elif oq <= 4:
        annual_rate *= 0.90
    forecasts = []
    for years in [1, 3, 5]:
        future_value = prediction * ((1 + annual_rate) ** years)
        gain = future_value - prediction
        gain_pct = (gain / prediction) * 100
        forecasts.append({
            "years": years,
            "value": round(future_value),
            "gain": round(gain),
            "gain_pct": round(gain_pct, 1),
            "label": f"{years} Year{'s' if years > 1 else ''}",
        })
    return {
        "forecasts": forecasts,
        "annual_rate": round(annual_rate * 100, 2),
        "trend": trend,
        "rate_label": rate_info["label"],
        "disclaimer": "Projections are estimates based on historical neighborhood trends and property characteristics. Actual appreciation may vary due to market conditions, economic factors, and property maintenance.",
    }


# ============================================================
# INVESTMENT RISK ANALYSIS
# ============================================================

def compute_investment_risk(inputs: dict, prediction: float, scores: dict) -> dict:
    risk_factors = []
    age = CURRENT_YEAR - inputs.get("YearBuilt", 2005)
    oq = inputs.get("OverallQual", 5)
    gc = inputs.get("GarageCars", 2)
    tbs = inputs.get("TotalBsmtSF", 800)
    gla = inputs.get("GrLivArea", 1500)
    nb = inputs.get("Neighborhood", "CollgCr")
    risk_score = 0
    if age >= 50:
        risk_score += 20
        risk_factors.append({
            "factor": "Property Age",
            "severity": "moderate",
            "detail": f"Built {age} years ago — older properties may have maintenance risks.",
            "icon": "\u23f0",
        })
    if oq <= 4:
        risk_score += 25
        risk_factors.append({
            "factor": "Below-Average Quality",
            "severity": "high",
            "detail": f"Quality rating {oq}/10 may limit resale appeal.",
            "icon": "\u26a0",
        })
    if gc == 0:
        risk_score += 15
        risk_factors.append({
            "factor": "No Garage",
            "severity": "moderate",
            "detail": "Lack of garage reduces buyer interest in this price range.",
            "icon": "\U0001f17f",
        })
    if tbs == 0:
        risk_score += 10
        risk_factors.append({
            "factor": "No Basement",
            "severity": "low",
            "detail": "Property lacks basement space, reducing total usable area.",
            "icon": "\u25bc",
        })
    if gla < 1000:
        risk_score += 15
        risk_factors.append({
            "factor": "Small Living Area",
            "severity": "moderate",
            "detail": "Compact living space may limit buyer pool.",
            "icon": "\U0001f4cf",
        })
    stats = NEIGHBORHOOD_STATS.get(nb, {})
    if stats.get("tier") == "Value":
        risk_score += 10
        risk_factors.append({
            "factor": "Neighborhood Tier",
            "severity": "low",
            "detail": f"Value-tier neighborhood may appreciate slower than premium areas.",
            "icon": "\U0001f4cd",
        })
    risk_score = min(100, risk_score)
    if risk_score <= 15:
        risk_level = "Low"
        risk_color = "#4ECB8D"
        risk_desc = "Minimal risk factors detected. Strong investment profile."
    elif risk_score <= 35:
        risk_level = "Moderate"
        risk_color = "#FFD060"
        risk_desc = "Some risk factors present. Standard for this market segment."
    elif risk_score <= 60:
        risk_level = "Elevated"
        risk_color = "#FF8C42"
        risk_desc = "Multiple risk factors identified. Due diligence recommended."
    else:
        risk_level = "High"
        risk_color = "#FF6B6B"
        risk_desc = "Significant risk factors. Careful consideration advised."
    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "risk_desc": risk_desc,
        "factors": risk_factors,
        "protection_score": scores.get("risk", 50),
    }


# ============================================================
# ENHANCED RECOMMENDATIONS WITH DOLLAR VALUES
# ============================================================


# ============================================================
# MODEL RELIABILITY METRICS
# ============================================================

MODEL_METRICS = {
    "r_squared_log": 0.3819,
    "rmse": 64802,
    "mae": 27773,
    "mape": 21.7,
    "median_ape": 6.6,
    "within_10_pct": 68.3,
    "within_20_pct": 87.5,
    "training_records": 1460,
    "features": 245,
    "model_type": "Linear Regression",
}


def get_model_reliability_metrics() -> dict:
    return dict(MODEL_METRICS)


# ============================================================
# "WHY THIS PRICE?" NARRATIVE
# ============================================================


# ============================================================
# PROPERTY STRENGTHS & WEAKNESSES
# ============================================================

def compute_property_strengths_weaknesses(inputs: dict) -> dict:
    strengths = []
    weaknesses = []

    oq = inputs.get("OverallQual", 5)
    if oq >= 7:
        strengths.append({"feature": "Overall Quality", "value": f"{oq}/10", "detail": f"Exceeds dataset average of {DATASET_AVERAGES['OverallQual']:.1f}"})
    elif oq <= 4:
        weaknesses.append({"feature": "Overall Quality", "value": f"{oq}/10", "detail": f"Below dataset average of {DATASET_AVERAGES['OverallQual']:.1f}"})

    gla = inputs.get("GrLivArea", 1500)
    if gla >= 2000:
        strengths.append({"feature": "Living Area", "value": f"{gla:,} sq ft", "detail": f"Exceeds average of {DATASET_AVERAGES['GrLivArea']:,.0f} sq ft"})
    elif gla < 1000:
        weaknesses.append({"feature": "Living Area", "value": f"{gla:,} sq ft", "detail": f"Below average of {DATASET_AVERAGES['GrLivArea']:,.0f} sq ft"})

    gc = inputs.get("GarageCars", 2)
    if gc >= 3:
        strengths.append({"feature": "Garage", "value": f"{gc} cars", "detail": f"Exceeds average of {DATASET_AVERAGES['GarageCars']:.1f} cars"})
    elif gc == 0:
        weaknesses.append({"feature": "Garage", "value": "None", "detail": f"Below average of {DATASET_AVERAGES['GarageCars']:.1f} cars"})

    tbs = inputs.get("TotalBsmtSF", 800)
    if tbs >= 1200:
        strengths.append({"feature": "Basement", "value": f"{tbs:,} sq ft", "detail": f"Exceeds average of {DATASET_AVERAGES['TotalBsmtSF']:,.0f} sq ft"})
    elif tbs == 0:
        weaknesses.append({"feature": "Basement", "value": "None", "detail": f"Below average of {DATASET_AVERAGES['TotalBsmtSF']:,.0f} sq ft"})

    yb = inputs.get("YearBuilt", 2005)
    age = CURRENT_YEAR - yb
    if age <= 15:
        strengths.append({"feature": "Age", "value": f"{age} years", "detail": f"Newer than average ({int(DATASET_AVERAGES['YearBuilt'])})"})
    elif age >= 50:
        weaknesses.append({"feature": "Age", "value": f"{age} years", "detail": f"Older than average ({int(DATASET_AVERAGES['YearBuilt'])})"})

    fb = inputs.get("FullBath", 2)
    if fb >= 3:
        strengths.append({"feature": "Bathrooms", "value": f"{fb} full", "detail": f"Exceeds average of {DATASET_AVERAGES['FullBath']:.1f} full baths"})
    elif fb < 2:
        weaknesses.append({"feature": "Bathrooms", "value": f"{fb} full", "detail": f"Below average of {DATASET_AVERAGES['FullBath']:.1f} full baths"})

    ba = inputs.get("BedroomAbvGr", 3)
    if ba >= 5:
        strengths.append({"feature": "Bedrooms", "value": str(ba), "detail": f"Exceeds average of {DATASET_AVERAGES['BedroomAbvGr']:.1f}"})
    elif ba <= 1:
        weaknesses.append({"feature": "Bedrooms", "value": str(ba), "detail": f"Below average of {DATASET_AVERAGES['BedroomAbvGr']:.1f}"})

    return {"strengths": strengths, "weaknesses": weaknesses}


# ============================================================
# SMART IMPROVEMENT SIMULATOR
# ============================================================

IMPROVEMENT_COEFFICIENTS = {
    "OverallQual": {"cost_range": (5000, 25000), "description": "Kitchen/bath finish upgrade"},
    "GrLivArea": {"cost_range": (50, 150), "per_unit": "sq ft", "description": "Room addition or extension"},
    "GarageCars": {"cost_range": (10000, 30000), "description": "Garage expansion or addition"},
    "TotalBsmtSF": {"cost_range": (30, 80), "per_unit": "sq ft", "description": "Basement finishing"},
    "FullBath": {"cost_range": (15000, 35000), "description": "Add full bathroom"},
    "BedroomAbvGr": {"cost_range": (10000, 25000), "description": "Add bedroom (extension)"},
}


def compute_improvement_simulator(inputs: dict, prediction: float) -> list[dict]:
    improvements = []
    oq = inputs.get("OverallQual", 5)
    if oq <= 6:
        cost = IMPROVEMENT_COEFFICIENTS["OverallQual"]["cost_range"]
        est_value = round(prediction * 0.10)
        improvements.append({
            "feature": "Upgrade Quality",
            "description": IMPROVEMENT_COEFFICIENTS["OverallQual"]["description"],
            "cost_low": cost[0], "cost_high": cost[1],
            "est_value": est_value,
            "roi": round(est_value / ((cost[0] + cost[1]) / 2) * 100),
        })

    gc = inputs.get("GarageCars", 2)
    if gc < 2:
        cost = IMPROVEMENT_COEFFICIENTS["GarageCars"]["cost_range"]
        est_value = round(prediction * 0.06)
        improvements.append({
            "feature": "Add Garage",
            "description": IMPROVEMENT_COEFFICIENTS["GarageCars"]["description"],
            "cost_low": cost[0], "cost_high": cost[1],
            "est_value": est_value,
            "roi": round(est_value / ((cost[0] + cost[1]) / 2) * 100),
        })

    tbs = inputs.get("TotalBsmtSF", 800)
    if tbs < 600:
        cost_per_sqft = IMPROVEMENT_COEFFICIENTS["TotalBsmtSF"]["cost_range"]
        add_sqft = 400
        cost = (cost_per_sqft[0] * add_sqft, cost_per_sqft[1] * add_sqft)
        est_value = round(prediction * 0.05)
        improvements.append({
            "feature": "Finish Basement",
            "description": f"Finish {add_sqft} sq ft of basement",
            "cost_low": cost[0], "cost_high": cost[1],
            "est_value": est_value,
            "roi": round(est_value / ((cost[0] + cost[1]) / 2) * 100),
        })

    fb = inputs.get("FullBath", 2)
    if fb < 2:
        cost = IMPROVEMENT_COEFFICIENTS["FullBath"]["cost_range"]
        est_value = round(prediction * 0.05)
        improvements.append({
            "feature": "Add Bathroom",
            "description": IMPROVEMENT_COEFFICIENTS["FullBath"]["description"],
            "cost_low": cost[0], "cost_high": cost[1],
            "est_value": est_value,
            "roi": round(est_value / ((cost[0] + cost[1]) / 2) * 100),
        })

    yb = inputs.get("YearBuilt", 2005)
    age = CURRENT_YEAR - yb
    if age >= 25:
        est_value = round(prediction * 0.04)
        improvements.append({
            "feature": "Curb Appeal",
            "description": "Exterior refresh and landscaping",
            "cost_low": 3000, "cost_high": 10000,
            "est_value": est_value,
            "roi": round(est_value / 6500 * 100),
        })

    if not improvements:
        improvements.append({
            "feature": "Well-Maintained",
            "description": "Property is already well-optimized — focus on maintenance",
            "cost_low": 0, "cost_high": 0,
            "est_value": 0,
            "roi": 0,
        })

    return improvements


# ============================================================
# PROPERTY REPORT SCORECARD
# ============================================================

def compute_property_report_scorecard(inputs: dict, prediction: float, scores: dict) -> list[dict]:
    categories = []

    oq = inputs.get("OverallQual", 5)
    q_score = min(10, max(1, round(oq)))
    categories.append({"category": "Quality", "score": q_score, "max": 10, "detail": f"Overall quality rating: {oq}/10"})

    gla = inputs.get("GrLivArea", 1500)
    if gla >= 3000: ls = 10
    elif gla >= 2500: ls = 9
    elif gla >= 2000: ls = 8
    elif gla >= 1800: ls = 7
    elif gla >= 1500: ls = 6
    elif gla >= 1200: ls = 5
    elif gla >= 1000: ls = 4
    else: ls = 3
    categories.append({"category": "Living Space", "score": ls, "max": 10, "detail": f"{gla:,} sq ft of living area"})

    gc = inputs.get("GarageCars", 2)
    if gc >= 3: gs = 9
    elif gc == 2: gs = 7
    elif gc == 1: gs = 5
    else: gs = 2
    categories.append({"category": "Garage", "score": gs, "max": 10, "detail": f"{gc}-car garage capacity"})

    fb = inputs.get("FullBath", 2)
    hb = inputs.get("HalfBath", 0)
    total_bath = fb + hb * 0.5
    if total_bath >= 3.5: bs = 10
    elif total_bath >= 3: bs = 9
    elif total_bath >= 2.5: bs = 8
    elif total_bath >= 2: bs = 7
    elif total_bath >= 1.5: bs = 5
    else: bs = 3
    categories.append({"category": "Bathrooms", "score": bs, "max": 10, "detail": f"{int(fb)} full + {int(hb)} half baths"})

    tbs = inputs.get("TotalBsmtSF", 800)
    if tbs >= 1500: bm = 9
    elif tbs >= 1000: bm = 7
    elif tbs >= 600: bm = 5
    elif tbs > 0: bm = 3
    else: bm = 1
    categories.append({"category": "Basement", "score": bm, "max": 10, "detail": f"{tbs:,} sq ft basement"})

    yb = inputs.get("YearBuilt", 2005)
    age = CURRENT_YEAR - yb
    if age <= 5: md = 10
    elif age <= 10: md = 9
    elif age <= 20: md = 8
    elif age <= 30: md = 7
    elif age <= 50: md = 5
    else: md = 3
    categories.append({"category": "Modernity", "score": md, "max": 10, "detail": f"Built {yb} ({age} years ago)"})

    inv = scores.get("investment", 50)
    ip = min(10, max(1, round(inv / 10)))
    categories.append({"category": "Investment Potential", "score": ip, "max": 10, "detail": f"Investment score: {inv}/100"})

    return categories


# ============================================================
# BENCHMARK PERCENTILES
# ============================================================

def compute_benchmark_percentiles(inputs: dict, prediction: float) -> list[dict]:
    training_data = _load_training_data()
    if training_data.empty:
        return []

    benchmarks = []
    feature_map = {
        "OverallQual": ("Overall Quality", "/10"),
        "GrLivArea": ("Living Area", " sq ft"),
        "GarageCars": ("Garage", " cars"),
        "TotalBsmtSF": ("Basement", " sq ft"),
        "FullBath": ("Full Bathrooms", ""),
        "BedroomAbvGr": ("Bedrooms", ""),
    }

    for col, (label, unit) in feature_map.items():
        if col not in training_data.columns:
            continue
        values = training_data[col].dropna()
        user_val = inputs.get(col, 0)
        percentile = (values < user_val).sum() / len(values) * 100
        percentile = round(percentile, 1)
        if percentile >= 90:
            tier = "Top 10%"
        elif percentile >= 75:
            tier = "Top 25%"
        elif percentile >= 50:
            tier = "Above Average"
        elif percentile >= 25:
            tier = "Average"
        else:
            tier = "Below Average"
        benchmarks.append({
            "feature": label,
            "value": user_val,
            "unit": unit,
            "percentile": percentile,
            "tier": tier,
            "dataset_median": float(values.median()),
        })

    if prediction > 0 and "SalePrice" in training_data.columns:
        prices = training_data["SalePrice"].dropna()
        price_percentile = (prices < prediction).sum() / len(prices) * 100
        benchmarks.append({
            "feature": "Predicted Price",
            "value": prediction,
            "unit": "",
            "percentile": round(price_percentile, 1),
            "tier": "Top 10%" if price_percentile >= 90 else "Top 25%" if price_percentile >= 75 else "Above Average" if price_percentile >= 50 else "Average" if price_percentile >= 25 else "Below Average",
            "dataset_median": float(prices.median()),
        })

    return benchmarks


# ============================================================
# PREDICTION TRANSPARENCY
# ============================================================

def generate_prediction_transparency(inputs: dict, prediction: float) -> dict:
    steps = [
        {
            "step": "Data Collection",
            "description": "Your property inputs (quality, size, garage, basement, etc.) were collected and validated against dataset ranges.",
            "status": "complete",
        },
        {
            "step": "Feature Engineering",
            "description": f"Inputs were encoded into {MODEL_METRICS['features']} numeric features using one-hot encoding for categorical variables like neighborhood.",
            "status": "complete",
        },
        {
            "step": "Preprocessing",
            "description": "Missing values were filled with dataset medians. Numeric features were used as-is since the model handles raw scales.",
            "status": "complete",
        },
        {
            "step": "Model Prediction",
            "description": f"A {MODEL_METRICS['model_type']} model trained on {MODEL_METRICS['training_records']:,} property records computed the prediction in log-space, then inverse-transformed via exp(x)-1.",
            "status": "complete",
        },
        {
            "step": "Confidence Estimation",
            "description": f"Confidence was computed based on how typical your inputs are relative to the training data distribution. The model's MAE is ${MODEL_METRICS['mae']:,}.",
            "status": "complete",
        },
        {
            "step": "Post-Processing",
            "description": f"A 95% prediction interval was computed using RMSE (${MODEL_METRICS['rmse']:,}) to show the likely range of actual sale prices.",
            "status": "complete",
        },
    ]

    limitations = [
        "The model explains ~38% of price variance (R²=0.3819), meaning other factors not in the dataset also influence prices.",
        f"Typical prediction error is ±${MODEL_METRICS['mae']:,} (MAE), so actual prices may differ from this estimate.",
        f"About {MODEL_METRICS['within_10_pct']:.0f}% of predictions fall within 10% of actual sale prices.",
        "The model was trained on Ames, Iowa data from 2006-2010 and may not generalize to other markets.",
    ]

    return {"steps": steps, "limitations": limitations}


# ============================================================
# DATASET INSIGHTS
# ============================================================

@lru_cache(maxsize=1)
def _compute_dataset_insights() -> dict:
    try:
        df = pd.read_csv("data/train.csv")
        prices = df["SalePrice"].dropna()
        gla = df["GrLivArea"].dropna() if "GrLivArea" in df.columns else pd.Series([0])
        qual = df["OverallQual"].dropna() if "OverallQual" in df.columns else pd.Series([0])
        return {
            "avg_price": round(float(prices.mean())),
            "median_price": round(float(prices.median())),
            "min_price": round(float(prices.min())),
            "max_price": round(float(prices.max())),
            "avg_living_area": round(float(gla.mean())),
            "avg_quality": round(float(qual.mean()), 1),
            "total_records": len(df),
            "price_std": round(float(prices.std())),
        }
    except Exception:
        return {
            "avg_price": 180921, "median_price": 163000, "min_price": 34900,
            "max_price": 755000, "avg_living_area": 1516, "avg_quality": 6.1,
            "total_records": 1460, "price_std": 79443,
        }


def get_dataset_insights() -> dict:
    return _compute_dataset_insights()


# ============================================================
# AI VERDICT
# ============================================================

def generate_ai_verdict(
    inputs: dict, prediction: float, scores: dict, market: dict
) -> dict:
    """Generate a professional appraisal verdict."""
    quality = inputs.get("OverallQual", 5)
    area = inputs.get("GrLivArea", 1500)
    year = inputs.get("YearBuilt", 2000)
    investment = scores.get("investment", 50)
    resale = scores.get("resale", 50)
    risk = scores.get("risk", 50)
    percentile = market.get("percentile", 50)
    nb = inputs.get("Neighborhood", "CollgCr")

    from utils import NEIGHBORHOOD_LABELS
    nb_label = NEIGHBORHOOD_LABELS.get(nb, nb)

    strengths = 0
    weaknesses = 0

    if quality >= 7:
        strengths += 1
    elif quality <= 4:
        weaknesses += 1

    if area >= 1800:
        strengths += 1
    elif area <= 1000:
        weaknesses += 1

    if year >= 2000:
        strengths += 1
    elif year <= 1970:
        weaknesses += 1

    if investment >= 70:
        strengths += 1
    elif investment <= 40:
        weaknesses += 1

    if percentile >= 60:
        strengths += 1
    elif percentile <= 30:
        weaknesses += 1

    score = strengths - weaknesses

    if score >= 3 and investment >= 70:
        verdict = "Strong Value"
        color = "#4ECB8D"
        icon = "&#x2713;"
        reasoning = (
            f"This property scores above average across multiple dimensions. "
            f"With quality {quality}/10, {area:,} sq ft of living space, and a "
            f"{percentile}th percentile market position, it represents solid "
            f"value in the {nb_label} area. The estimated price of "
            f"${prediction:,.0f} aligns with its characteristics."
        )
        action = "Well-positioned for both living and long-term investment."
    elif score >= 1 and investment >= 50:
        verdict = "Fair Value"
        color = "#D4A84E"
        icon = "&#x25C6;"
        reasoning = (
            f"This property offers reasonable value for its estimated price of "
            f"${prediction:,.0f}. It meets most benchmarks for the {nb_label} area. "
            f"With quality {quality}/10 and {area:,} sq ft, it sits near the "
            f"middle of the market with room for improvement."
        )
        action = "A balanced option — suitable for buyers prioritizing location over premium features."
    else:
        verdict = "Cautious Assessment"
        color = "#FFD060"
        icon = "&#x26A0;"
        reasoning = (
            f"At ${prediction:,.0f}, this property has trade-offs to consider. "
            f"With quality {quality}/10 and a {percentile}th percentile position, "
            f"there may be better value alternatives in {nb_label} or nearby areas."
        )
        action = "Consider negotiating on price or exploring adjacent neighborhoods."

    return {
        "verdict": verdict,
        "color": color,
        "icon": icon,
        "reasoning": reasoning,
        "action": action,
        "strengths": strengths,
        "weaknesses": weaknesses,
    }


# ============================================================
# VALUE DRIVERS
# ============================================================

def generate_value_drivers(inputs: dict, prediction: float) -> list:
    """Generate 3-5 concise reasons why this property has its estimated value."""
    drivers = []
    quality = inputs.get("OverallQual", 5)
    area = inputs.get("GrLivArea", 1500)
    garage = inputs.get("GarageCars", 2)
    basement = inputs.get("TotalBsmtSF", 800)
    year = inputs.get("YearBuilt", 2000)
    beds = inputs.get("BedroomAbvGr", 3)
    baths = inputs.get("FullBath", 2)
    nb = inputs.get("Neighborhood", "CollgCr")

    from utils import NEIGHBORHOOD_LABELS, NEIGHBORHOOD_STATS
    nb_label = NEIGHBORHOOD_LABELS.get(nb, nb)
    nb_data = NEIGHBORHOOD_STATS.get(nb, {})
    nb_avg = nb_data.get("avg_price", 180921)

    if quality >= 7:
        drivers.append({
            "title": f"High Quality Finish ({quality}/10)",
            "detail": "Above-average material quality and workmanship increase resale appeal.",
            "icon": "&#x2B50;",
        })
    elif quality <= 4:
        drivers.append({
            "title": f"Basic Finish ({quality}/10)",
            "detail": "Standard builder-grade finishes limit premium positioning.",
            "icon": "&#x1F527;",
        })

    if area >= 1800:
        drivers.append({
            "title": f"{area:,} sq ft Living Space",
            "detail": f"Larger than the dataset average of 1,516 sq ft — appeals to family buyers.",
            "icon": "&#x1F3E0;",
        })
    elif area <= 1000:
        drivers.append({
            "title": f"{area:,} sq ft Living Space",
            "detail": "Compact footprint limits the buyer pool but reduces maintenance costs.",
            "icon": "&#x1F3E0;",
        })

    if garage >= 2:
        drivers.append({
            "title": f"{garage}-Car Garage",
            "detail": "Meets or exceeds the typical buyer expectation for the area.",
            "icon": "&#x1F697;",
        })
    elif garage == 0:
        drivers.append({
            "title": "No Garage",
            "detail": "Lack of covered parking may reduce resale value in suburban neighborhoods.",
            "icon": "&#x1F697;",
        })

    if basement > 1000:
        drivers.append({
            "title": f"{basement:,} sq ft Basement",
            "detail": "Full basement adds significant usable space and storage value.",
            "icon": "&#x1F4CF;",
        })

    if year >= 2005:
        drivers.append({
            "title": f"Built in {year}",
            "detail": "Modern construction with updated building codes and lower maintenance needs.",
            "icon": "&#x23F1;",
        })
    elif year <= 1965:
        age = 2024 - year
        drivers.append({
            "title": f"Built in {year} ({age}+ years old)",
            "detail": "Character home, but may require updated systems (plumbing, electrical).",
            "icon": "&#x1F3DB;",
        })

    if prediction > nb_avg * 1.1:
        drivers.append({
            "title": f"Premium to {nb_label} Average",
            "detail": f"Estimated ${prediction - nb_avg:,.0f} above the neighborhood average of ${nb_avg:,.0f}.",
            "icon": "&#x1F4C8;",
        })
    elif prediction < nb_avg * 0.9:
        drivers.append({
            "title": f"Below {nb_label} Average",
            "detail": f"Estimated ${nb_avg - prediction:,.0f} below the neighborhood average — potential value opportunity.",
            "icon": "&#x1F4C9;",
        })

    if len(drivers) < 3:
        drivers.append({
            "title": f"{beds} Bedrooms, {baths} Bathrooms",
            "detail": "Standard layout suitable for most family configurations.",
            "icon": "&#x1F6CF;",
        })

    return drivers[:5]


# ============================================================
# INVESTMENT INTERPRETATION
# ============================================================

def generate_investment_interpretation(
    inputs: dict, prediction: float, scores: dict, market: dict
) -> str:
    """Generate an analyst-style investment interpretation."""
    quality = inputs.get("OverallQual", 5)
    area = inputs.get("GrLivArea", 1500)
    year = inputs.get("YearBuilt", 2000)
    investment = scores.get("investment", 50)
    rental = scores.get("rental", 50)
    resale = scores.get("resale", 50)
    risk = scores.get("risk", 50)
    percentile = market.get("percentile", 50)
    nb = inputs.get("Neighborhood", "CollgCr")

    from utils import NEIGHBORHOOD_LABELS
    nb_label = NEIGHBORHOOD_LABELS.get(nb, nb)

    from utils import MODEL_METRICS
    mae = MODEL_METRICS.get("mae", 27773)

    paragraphs = []

    paragraphs.append(
        f"At ${prediction:,.0f}, this {area:,}-sqft property in {nb_label} "
        f"ranks in the {percentile}th percentile of comparable homes. The model's "
        f"typical error margin is ${mae:,.0f}, so the true market value likely "
        f"falls between ${prediction - mae:,.0f} and ${prediction + mae:,.0f}."
    )

    if investment >= 70:
        paragraphs.append(
            f"The investment profile is solid. With an investment score of "
            f"{investment}/100 and strong resale potential ({resale}/100), this "
            f"property is well-positioned for both personal use and long-term "
            f"appreciation. The risk level is {'low' if risk >= 70 else 'moderate'} "
            f"given the {nb_label} market dynamics."
        )
    elif investment >= 50:
        paragraphs.append(
            f"The investment profile is balanced. The investment score of "
            f"{investment}/100 suggests fair value at this price point. Rental "
            f"potential is {'strong' if rental >= 65 else 'moderate'} at "
            f"{rental}/100. For a buyer planning to live in the property for "
            f"5+ years, the risk-adjusted return is reasonable."
        )
    else:
        paragraphs.append(
            f"The investment score of {investment}/100 suggests this property "
            f"may not be the strongest financial play at ${prediction:,.0f}. "
            f"While it meets basic livability standards, buyers focused on "
            f"appreciation may find better options nearby."
        )

    if quality >= 7 and year >= 2000:
        paragraphs.append(
            "The combination of quality finishes and modern construction "
            "means lower near-term maintenance costs — an often-overlooked "
            "factor in total cost of ownership."
        )
    elif quality <= 4:
        paragraphs.append(
            "At this quality level, budget for potential upgrades. "
            "Kitchen and bathroom improvements typically offer the best "
            "return on investment for properties in this tier."
        )

    return " ".join(paragraphs)


# ============================================================
# NEXT ACTIONS
# ============================================================

def generate_next_actions() -> list:
    """Generate recommended next actions for the user."""
    return [
        {"title": "Download PDF Report", "desc": "Export the full valuation report with all sections.", "icon": "&#x1F4E5;"},
        {"title": "Predict Another Property", "desc": "Enter a different property to compare valuations.", "icon": "&#x1F504;"},
        {"title": "Explore Dataset Insights", "desc": "View market statistics and dataset benchmarks.", "icon": "&#x1F4CA;"},
        {"title": "View Prediction History", "desc": "Compare this prediction against previous analyses.", "icon": "&#x1F4CB;"},
    ]


# ============================================================
# PROPERTY CLASSIFICATION & IMAGE SELECTION
# ============================================================

import os
import hashlib

_HOUSE_IMAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "houses")

_HOUSE_CATEGORIES = [
    "starter", "cottage", "ranch", "townhouse", "suburban",
    "craftsman", "colonial", "modern", "luxury", "mansion",
]

# Dataset percentile thresholds (from train.csv)
_P25 = 129975
_P50 = 163000
_P75 = 214000
_P90 = 278000
_P95 = 326100


def classify_property(inputs: dict, prediction: float) -> str:
    """Deterministically classify a property into a house category.

    Classification rules (evaluated top-down, first match wins):
    1. mansion   — Qual>=9, price>P95, area>3000, garage>=3
    2. luxury    — Qual>=8, price>P75, area>2500
    3. modern    — Built>=2000, Qual>=7, area>1800
    4. craftsman — Qual>=6, built 1940-1969, area>1500
    5. colonial  — Qual>=6, built<1970, area>1800, beds>=4
    6. suburban  — Qual>=5, area>=1200, garage>=2, price P25-P75
    7. townhouse — Qual>=5, area<1800, garage<=2, beds<=3
    8. ranch     — Qual>=5, built<1970, area>=1200, beds<=3
    9. cottage   — Qual<=5, area<1200
    10. starter  — price<P25, Qual<=5
    11. suburban — fallback
    """
    oq = inputs.get("OverallQual", 5)
    gla = inputs.get("GrLivArea", 1500)
    gc = inputs.get("GarageCars", 2)
    yb = inputs.get("YearBuilt", 2005)
    beds = inputs.get("BedroomAbvGr", 3)

    if oq >= 9 and prediction > _P95 and gla > 3000 and gc >= 3:
        return "mansion"
    if oq >= 8 and prediction > _P75 and gla > 2500:
        return "luxury"
    if yb >= 2000 and oq >= 7 and gla > 1800:
        return "modern"
    if oq >= 6 and yb < 1970 and gla > 1800 and beds >= 4:
        return "colonial"
    if oq >= 6 and 1940 <= yb < 1970 and gla > 1500 and beds <= 3:
        return "craftsman"
    if oq >= 5 and gla >= 1200 and gc >= 2 and _P25 <= prediction <= _P75:
        return "suburban"
    if oq >= 5 and yb >= 1970 and gla < 1800 and gla >= 1100 and gc <= 2 and beds <= 3:
        return "townhouse"
    if oq >= 5 and yb < 1970 and gla >= 1200 and beds <= 3:
        return "ranch"
    if oq <= 5 and gla < 1100 and prediction < _P25:
        return "starter"
    if oq <= 5 and gla < 1100:
        return "cottage"
    return "suburban"


# Human-readable labels for each category
_CATEGORY_LABELS = {
    "starter": "Starter Residential Home",
    "cottage": "Charming Cottage Residence",
    "ranch": "Traditional Ranch Home",
    "townhouse": "Contemporary Townhouse",
    "suburban": "Suburban Family Home",
    "craftsman": "Craftsman-Style Residence",
    "colonial": "Colonial Family Home",
    "modern": "Modern Family Home",
    "luxury": "Luxury Executive Home",
    "mansion": "Premium Estate Property",
}

# Descriptive paragraph per category
_CATEGORY_DESCRIPTIONS = {
    "starter": "This property represents an accessible entry point into the market. With compact living space and standard finishes, it is well-suited for first-time buyers or investors seeking affordable rental opportunities.",
    "cottage": "A compact property with character charm. The modest living area and traditional construction offer a cozy living experience, ideal for downsizers or those seeking a low-maintenance lifestyle.",
    "ranch": "A single-story traditional home with practical layout. The ranch-style design provides accessible living on one level, well-suited for families who prefer horizontal space over multiple floors.",
    "townhouse": "A contemporary attached residence offering efficient use of space. The townhouse format provides modern living with reduced exterior maintenance, popular among young professionals and small families.",
    "suburban": "A well-proportioned family home in a suburban setting. With balanced living space, adequate garage capacity, and standard amenities, this property serves the needs of growing families seeking community-oriented living.",
    "craftsman": "A residence featuring traditional craftsmanship and architectural character. Built during an era of quality construction, this home offers solid bones and timeless design elements that appeal to buyers who value authenticity.",
    "colonial": "A classic colonial-style family home with generous bedroom count and traditional floor plan. The colonial architecture provides formal living spaces and a layout designed for family life across multiple generations.",
    "modern": "A contemporary residence built to modern standards. Recent construction ensures current building codes, energy efficiency, and updated systems, appealing to buyers who prioritize low maintenance and modern amenities.",
    "luxury": "An executive-level property with premium finishes and generous living space. High-quality construction materials and superior craftsmanship position this home in the upper market segment.",
    "mansion": "A premier estate property representing the top tier of the market. Exceptional construction quality, expansive living area, and premium features define this property as a distinguished residence.",
}

# Lifestyle tags per category
_CATEGORY_LIFESTYLES = {
    "starter": ["First-Time Buyers", "Rental Investors"],
    "cottage": ["Downsizers", "Young Professionals"],
    "ranch": ["Retirees", "Families"],
    "townhouse": ["Young Professionals", "Small Families"],
    "suburban": ["Growing Families", "Suburban Living"],
    "craftsman": ["Established Families", "Character Home Enthusiasts"],
    "colonial": ["Large Families", "Traditional Living"],
    "modern": ["Growing Families", "Low-Maintenance Seekers"],
    "luxury": ["Luxury Buyers", "Executive Living"],
    "mansion": ["High-Income Professionals", "Estate Buyers"],
}


def select_house_image(category: str, inputs: dict) -> str:
    """Select a deterministic house image from assets/houses/{category}/.

    Uses a hash of key property features so identical inputs always
    produce the same image. Returns a relative path suitable for
    Streamlit's st.image() or base64 encoding.

    Falls back to the first image in the category if hashing fails.
    Returns empty string if no images exist.
    """
    cat_dir = os.path.join(_HOUSE_IMAGE_DIR, category)
    if not os.path.isdir(cat_dir):
        return ""

    images = sorted([
        f for f in os.listdir(cat_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ])
    if not images:
        return ""

    # Deterministic selection: hash of key features
    key = f"{inputs.get('OverallQual',0)}-{inputs.get('GrLivArea',0)}-{inputs.get('GarageCars',0)}-{inputs.get('YearBuilt',0)}-{inputs.get('BedroomAbvGr',0)}-{inputs.get('FullBath',0)}"
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    idx = h % len(images)
    return os.path.join(cat_dir, images[idx])


def generate_property_profile(inputs: dict, prediction: float) -> dict:
    """Generate a complete property profile for the report.

    Returns a dictionary with all data needed for the property
    visualization section: classification, image path, label,
    description, lifestyle tags, highlights, drawbacks, and
    market segment info.
    """
    category = classify_property(inputs, prediction)
    image_path = select_house_image(category, inputs)

    oq = inputs.get("OverallQual", 5)
    gla = inputs.get("GrLivArea", 1500)
    gc = inputs.get("GarageCars", 2)
    yb = inputs.get("YearBuilt", 2005)
    beds = inputs.get("BedroomAbvGr", 3)
    baths = inputs.get("FullBath", 2)
    tbs = inputs.get("TotalBsmtSF", 800)
    age = CURRENT_YEAR - yb

    # Market segment based on prediction percentile
    if prediction >= _P90:
        market_segment = "Ultra Premium"
        segment_pct = 95
    elif prediction >= _P75:
        market_segment = "Premium"
        segment_pct = 80
    elif prediction >= _P50:
        market_segment = "Upper Mid Market"
        segment_pct = 60
    elif prediction >= _P25:
        market_segment = "Mid Market"
        segment_pct = 40
    else:
        market_segment = "Entry Level"
        segment_pct = 20

    # Property highlights (only applicable ones)
    highlights = []
    if oq >= 7:
        highlights.append({"icon": "&#x2B50;", "title": "Excellent Build Quality", "detail": f"Quality rating of {oq}/10 exceeds the dataset average."})
    elif oq >= 5:
        highlights.append({"icon": "&#x2713;", "title": "Solid Construction", "detail": f"Quality rating of {oq}/10 meets standard expectations."})

    if gla >= 2000:
        highlights.append({"icon": "&#x1F4D0;", "title": "Spacious Living Area", "detail": f"{gla:,} sq ft provides generous living space."})
    elif gla >= 1500:
        highlights.append({"icon": "&#x1F4D0;", "title": "Comfortable Living Area", "detail": f"{gla:,} sq ft offers balanced living space."})

    if yb >= 2005:
        highlights.append({"icon": "&#x1F3D7;", "title": "Modern Construction", "detail": f"Built in {yb} with current building standards."})
    elif yb >= 1990:
        highlights.append({"icon": "&#x1F3D7;", "title": "Updated Construction", "detail": f"Built in {yb} with relatively modern systems."})

    if gc >= 3:
        highlights.append({"icon": "&#x1F697;", "title": "Large Garage", "detail": f"{gc}-car garage exceeds typical expectations."})
    elif gc >= 2:
        highlights.append({"icon": "&#x1F697;", "title": "Adequate Garage", "detail": f"{gc}-car garage meets standard requirements."})

    if tbs >= 1000:
        highlights.append({"icon": "&#x1F4CF;", "title": "Finished Basement", "detail": f"{tbs:,} sq ft basement provides additional usable space."})

    if beds >= 4:
        highlights.append({"icon": "&#x1F6CF;", "title": "Family Friendly Layout", "detail": f"{beds} bedrooms accommodate larger households."})

    if baths >= 3:
        highlights.append({"icon": "&#x1F6BF;", "title": "Ample Bathrooms", "detail": f"{baths} full bathrooms support daily convenience."})

    # Potential drawbacks
    drawbacks = []
    if age >= 50:
        drawbacks.append({"icon": "&#x23F0;", "title": "Older Construction", "detail": f"Built {age} years ago — may require system updates."})
    if gc == 0:
        drawbacks.append({"icon": "&#x26A0;", "title": "No Garage", "detail": "Lack of covered parking reduces buyer appeal."})
    elif gc == 1:
        drawbacks.append({"icon": "&#x26A0;", "title": "Limited Garage", "detail": "Single-car garage may not meet family expectations."})
    if gla < 1000:
        drawbacks.append({"icon": "&#x1F4CF;", "title": "Compact Living Area", "detail": f"{gla:,} sq ft is below the dataset average."})
    if oq <= 4:
        drawbacks.append({"icon": "&#x1F527;", "title": "Lower Overall Quality", "detail": f"Quality rating of {oq}/10 may limit resale potential."})
    if tbs < 400 and tbs > 0:
        drawbacks.append({"icon": "&#x1F4CF;", "title": "Limited Basement", "detail": f"{tbs:,} sq ft basement offers minimal additional space."})

    # Lifestyle tags
    lifestyles = _CATEGORY_LIFESTYLES.get(category, ["Growing Families"])

    # Overall score (0-10)
    score = min(10.0, max(1.0, round(
        (oq / 10) * 3
        + min(gla / 4000, 1.0) * 2.5
        + min(gc / 4, 1.0) * 1.5
        + max(0, (1 - age / 80)) * 1.5
        + (1.5 if prediction > _P50 else 0.5)
    , 1)))

    # Recommendation
    if score >= 8:
        recommendation = "Highly Recommended"
    elif score >= 6:
        recommendation = "Recommended"
    elif score >= 4:
        recommendation = "Fair Option"
    else:
        recommendation = "Consider Alternatives"

    return {
        "category": category,
        "image_path": image_path,
        "label": _CATEGORY_LABELS.get(category, "Residential Property"),
        "description": _CATEGORY_DESCRIPTIONS.get(category, "A residential property with standard features."),
        "lifestyles": lifestyles,
        "market_segment": market_segment,
        "segment_pct": segment_pct,
        "highlights": highlights,
        "drawbacks": drawbacks,
        "score": score,
        "recommendation": recommendation,
    }
