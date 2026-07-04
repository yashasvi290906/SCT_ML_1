import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

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

QUALITY_THRESHOLDS = {
    "OverallQual": {"high": 7, "low": 4},
    "OverallCond": {"high": 7, "low": 4},
    "KitchenQual": {"high": 8, "low": 5},
}

NEIGHBORHOODS = [
    "Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr",
    "CollgCr", "Crawfor", "Edwards", "Gilbert", "IDOTRR",
    "MeadowV", "Mitchel", "Names", "NoRidge", "NPkVill",
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
    "Names": "North Ames",
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
# EXPLAINABLE AI
# ============================================================

def generate_explanations(inputs: dict, prediction: float) -> list[dict]:
    insights = []

    oq = inputs.get("OverallQual", 5)
    if oq >= 8:
        insights.append({
            "type": "positive",
            "icon": "+",
            "text": f"<strong>Outstanding Quality ({oq}/10)</strong> — Exceptional materials and finish significantly increased the property valuation.",
        })
    elif oq >= 6:
        insights.append({
            "type": "positive",
            "icon": "+",
            "text": f"<strong>Good Quality ({oq}/10)</strong> — Above-average construction quality contributed positively to the estimated price.",
        })
    elif oq <= 4:
        insights.append({
            "type": "negative",
            "icon": "-",
            "text": f"<strong>Below-Average Quality ({oq}/10)</strong> — Lower construction quality reduced the overall valuation.",
        })
    else:
        insights.append({
            "type": "neutral",
            "icon": "~",
            "text": f"<strong>Average Quality ({oq}/10)</strong> — Standard construction quality has a neutral impact on pricing.",
        })

    gla = inputs.get("GrLivArea", 1500)
    if gla >= 2500:
        insights.append({
            "type": "positive",
            "icon": "+",
            "text": f"<strong>{gla:,} sq ft Living Area</strong> — Spacious living space is a major driver of premium valuations.",
        })
    elif gla >= 1800:
        insights.append({
            "type": "positive",
            "icon": "+",
            "text": f"<strong>{gla:,} sq ft Living Area</strong> — Generous living area contributed positively to the price.",
        })
    elif gla < 1000:
        insights.append({
            "type": "negative",
            "icon": "-",
            "text": f"<strong>{gla:,} sq ft Living Area</strong> — Compact living space may limit resale potential.",
        })
    else:
        insights.append({
            "type": "neutral",
            "icon": "~",
            "text": f"<strong>{gla:,} sq ft Living Area</strong> — Moderate living area has a balanced effect on valuation.",
        })

    gc = inputs.get("GarageCars", 2)
    if gc >= 3:
        insights.append({
            "type": "positive",
            "icon": "+",
            "text": f"<strong>{gc}-Car Garage</strong> — Large garage capacity enhances convenience and adds tangible value.",
        })
    elif gc == 0:
        insights.append({
            "type": "negative",
            "icon": "-",
            "text": "<strong>No Garage</strong> — Absence of parking reduces overall property attractiveness.",
        })

    tbs = inputs.get("TotalBsmtSF", 800)
    if tbs >= 1500:
        insights.append({
            "type": "positive",
            "icon": "+",
            "text": f"<strong>{tbs:,} sq ft Basement</strong> — Expansive basement adds significant usable space and value.",
        })
    elif tbs == 0:
        insights.append({
            "type": "negative",
            "icon": "-",
            "text": "<strong>No Basement</strong> — Property lacks basement space, reducing overall square footage.",
        })

    fb = inputs.get("FullBath", 2)
    hb = inputs.get("HalfBath", 0)
    total_bath = fb + hb * 0.5
    if total_bath >= 3:
        insights.append({
            "type": "positive",
            "icon": "+",
            "text": f"<strong>{int(fb)} Full + {int(hb)} Half Baths</strong> — Ample bathroom count increases daily comfort and market appeal.",
        })

    yb = inputs.get("YearBuilt", 2005)
    age = 2026 - yb
    if age <= 10:
        insights.append({
            "type": "positive",
            "icon": "+",
            "text": f"<strong>Built in {yb}</strong> — Recent construction ({age} years old) commands a premium for modern standards.",
        })
    elif age >= 60:
        insights.append({
            "type": "neutral",
            "icon": "~",
            "text": f"<strong>Built in {yb}</strong> — Older construction ({age} years) may require updates but offers classic character.",
        })

    ba = inputs.get("BedroomAbvGr", 3)
    if ba >= 5:
        insights.append({
            "type": "positive",
            "icon": "+",
            "text": f"<strong>{ba} Bedrooms</strong> — Large bedroom count makes this property attractive for families.",
        })
    elif ba <= 1:
        insights.append({
            "type": "negative",
            "icon": "-",
            "text": f"<strong>{ba} Bedroom(s)</strong> — Limited bedroom count restricts target buyer pool.",
        })

    return insights


def generate_ai_summary(inputs: dict, prediction: float, quality_score: float) -> str:
    parts = []

    oq = inputs.get("OverallQual", 5)
    gla = inputs.get("GrLivArea", 1500)
    gc = inputs.get("GarageCars", 2)
    tbs = inputs.get("TotalBsmtSF", 800)
    yb = inputs.get("YearBuilt", 2005)
    fb = inputs.get("FullBath", 2)
    hb = inputs.get("HalfBath", 0)
    ba = inputs.get("BedroomAbvGr", 3)
    nb = inputs.get("Neighborhood", "CollgCr")
    age = 2026 - yb

    if prediction >= 350000:
        parts.append("This property is positioned in the premium market segment.")
    elif prediction >= 200000:
        parts.append("This property represents strong value in the current market.")
    else:
        parts.append("This property falls in the accessible price range.")

    if oq >= 7:
        parts.append(f"High construction quality ({oq}/10) is the primary value driver.")
    elif oq <= 4:
        parts.append(f"Below-average quality ({oq}/10) limits the valuation ceiling.")

    if gla >= 2000:
        parts.append(f"The {gla:,} sq ft living area provides above-average space.")
    if gc >= 3:
        parts.append(f"A {gc}-car garage adds practical convenience and resale appeal.")
    if age <= 15:
        parts.append(f"Built {age} years ago, the relatively modern construction is an asset.")
    elif age >= 50:
        parts.append(f"At {age} years old, the home has character but may need updates.")

    total_bath = fb + hb * 0.5
    if total_bath >= 3:
        parts.append(f"{int(fb)} full and {int(hb)} half bathrooms meet family needs.")

    neighborhood_display = NEIGHBORHOOD_LABELS.get(nb, nb)
    parts.append(f"Located in {neighborhood_display}, the property benefits from its neighborhood positioning.")

    return " ".join(parts)


# ============================================================
# CHARTS
# ============================================================

CHART_COLORS = {
    "primary": "#D7A24E",
    "secondary": "#F2BF67",
    "bg": "#1A0E0A",
    "card": "#261A14",
    "text": "#F5F0EB",
    "text_dim": "rgba(245,240,235,0.5)",
    "positive": "#48C78E",
    "negative": "#FF6363",
    "neutral": "#FFC837",
    "grid": "rgba(255,255,255,0.05)",
}

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=CHART_COLORS["text"], family="Inter, sans-serif", size=13),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor=CHART_COLORS["grid"], zeroline=False),
    yaxis=dict(gridcolor=CHART_COLORS["grid"], zeroline=False),
)


def create_feature_importance_chart() -> go.Figure:
    features = [
        "Overall Quality", "Living Area", "Garage Size",
        "Basement Area", "Year Built", "Bathrooms",
        "Bedrooms", "Kitchen Quality",
    ]
    importance = [0.30, 0.18, 0.12, 0.12, 0.10, 0.08, 0.05, 0.05]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=importance,
        y=features,
        orientation="h",
        marker=dict(
            color=importance,
            colorscale=[[0, "rgba(215,162,78,0.3)"], [1, "#D7A24E"]],
            cornerradius=6,
        ),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.0%}<extra></extra>",
    ))
    fig.update_layout(**CHART_LAYOUT, height=360, showlegend=False)
    fig.update_xaxes(title_text="Relative Importance", tickformat=".0%")
    return fig


def create_input_summary_chart(inputs: dict) -> go.Figure:
    labels = ["Quality", "Living Area", "Garage", "Basement", "Bathrooms", "Bedrooms", "Year Built"]
    max_vals = [10, 4000, 5, 3000, 5, 8, 2026]
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
        fillcolor="rgba(215,162,78,0.15)",
        line=dict(color="#D7A24E", width=2),
        marker=dict(size=8, color="#D7A24E"),
        hovertemplate="<b>%{theta}</b><br>Score: %{r:.0%}<extra></extra>",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        height=380,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=CHART_COLORS["grid"], tickformat=".0%"),
            angularaxis=dict(gridcolor=CHART_COLORS["grid"]),
        ),
        showlegend=False,
    )
    return fig


def create_price_comparison_chart(prediction: float) -> go.Figure:
    categories = ["Budget", "Value", "Average", "Your Home", "Premium", "Luxury"]
    avg_prices = [120000, 180000, 250000, prediction, 400000, 600000]
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
        marker=dict(
            color=colors,
            cornerradius=8,
        ),
        text=[f"${p:,.0f}" for p in avg_prices],
        textposition="outside",
        textfont=dict(color=CHART_COLORS["text"], size=12),
        hovertemplate="<b>%{x}</b><br>$%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(**CHART_LAYOUT, height=340)
    fig.update_yaxes(tickformat="$,.0f")
    fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
    return fig
