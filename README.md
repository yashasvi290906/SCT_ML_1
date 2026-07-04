<div align="center">

# HomeSense AI

**Premium AI-Powered Real Estate Valuation Platform**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

---

*Predict real estate values with machine learning, explainable AI, and professional-grade reporting.*

</div>

---

## Overview

HomeSense AI is a full-stack machine learning application that predicts residential property prices using Linear Regression trained on the Ames Housing Dataset. The platform delivers professional-grade valuations with full transparency — every prediction comes with an AI verdict, value drivers, investment analysis, strength/weakness assessment, and an exportable PDF report.

Built with a luxury dark chocolate and gold glassmorphism UI, the application is designed to feel like a commercial SaaS product rather than a student project.

---

## Features

| Feature | Description |
|---------|-------------|
| **AI Price Prediction** | Linear Regression model trained on 245 encoded property features |
| **Property Visualization** | Deterministic house image selection based on property characteristics |
| **Property Classification** | 10-category classification (Mansion, Luxury, Modern, Craftsman, Colonial, Suburban, Townhouse, Ranch, Cottage, Starter) |
| **Property Profile** | Premium property summary with lifestyle badges, highlights, and market segment |
| **AI Verdict** | Strong Value / Fair Value / Cautious Assessment with reasoning |
| **Value Drivers** | 3-5 concise factors explaining why the property is valued at its price |
| **Investment Interpretation** | Analyst-style paragraph on whether the property is a good investment |
| **Property Assessment** | Strengths and weaknesses compared to dataset averages |
| **Investment Scores** | Investment, Rental Potential, Resale Value, Luxury, and Risk scores |
| **Property Scorecard** | 7-category scoring with letter grade (A-D) |
| **Market Position** | Percentile ranking with visual tier indicator |
| **Interactive Charts** | Feature importance, radar profile, and price distribution (Plotly) |
| **AI Recommendations** | Targeted improvements with estimated value impact |
| **Improvement Simulator** | Dollar-value cost estimates and ROI for each improvement |
| **Neighborhood Insights** | Area-specific market data, demand, and buyer demographics |
| **Comparable Analysis** | Side-by-side comparison against dataset averages |
| **Benchmark Percentiles** | Where your property ranks across 7 key metrics |
| **Similar Properties** | Nearest-neighbor matching from the training dataset |
| **Appreciation Forecast** | 1, 3, and 5-year projected value growth |
| **Investment Risk Analysis** | Risk factors with severity levels and protection score |
| **Professional PDF Report** | 6-page exportable valuation report |
| **Prediction History** | Track and compare previous valuations |
| **Technical Details** | Collapsible panel with model metrics and methodology |
| **Responsive Design** | Desktop-first with tablet and mobile breakpoints |

---

## Screenshots

| Section | Description |
|---------|-------------|
| **Landing Page** | Animated hero with glassmorphism floating card and trust metrics |
| **Prediction Form** | 3-column input grid with sliders, number inputs, and neighborhood selector |
| **Valuation Hero** | Large price display with confidence interval and badge |
| **AI Verdict** | Color-coded verdict card with reasoning and recommendation |
| **Charts** | Feature importance bar chart, radar profile, and price distribution |
| **PDF Report** | 6-page professional report with cover, scores, and recommendations |

---

## Model Details

| Metric | Value |
|--------|-------|
| Algorithm | Linear Regression |
| Training Samples | 1,460 |
| Features | 245 (encoded) |
| Target Transform | log1p (inverse via expm1) |
| R² (Log Space) | 0.3819 |
| RMSE | $64,802 |
| MAE | $27,773 |
| MAPE | 21.7% |
| Median APE | 6.6% |
| Within 10% | 68.3% |
| Within 20% | 87.5% |

### Feature Engineering

- **Numeric features** (14): OverallQual, GrLivArea, GarageCars, TotalBsmtSF, FullBath, HalfBath, BedroomAbvGr, YearBuilt, LotFrontage, LotArea, OverallCond, YearRemodAdd, MasVnrArea, GarageArea
- **Categorical encoding**: One-hot encoding for Neighborhood (25 categories), resulting in 245 total feature columns
- **Missing values**: Filled with dataset medians for numeric features
- **Target**: SalePrice transformed via `log1p()` for training; predictions inverse-transformed via `np.expm1()`

### Explainable AI

Every prediction includes:

- **AI Verdict**: Deterministic scoring based on quality, area, year, investment score, and market percentile
- **Value Drivers**: Factors ranked by impact, compared against dataset averages
- **Feature Importance**: Standardized coefficients (|coef| × std) showing relative contribution
- **Investment Interpretation**: Natural language analysis of the investment profile
- **Transparency Steps**: 6-step explanation of how the prediction was computed

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit 1.35+ with Custom CSS (Glassmorphism) |
| ML Model | Linear Regression (Scikit-learn) |
| Visualization | Plotly (Interactive Charts) |
| PDF Generation | Matplotlib + PdfPages |
| Data Processing | Pandas, NumPy |
| Model Persistence | Joblib |
| Language | Python 3.10+ |
| Deployment | Streamlit Community Cloud |

---

## Dataset

**Ames Housing Dataset** — Compiled by Dean De Cock, containing 1,460 residential property records from Ames, Iowa with 80+ descriptive features including:

- Lot size, zoning, and configuration
- Overall quality and condition ratings (1-10 scale)
- Living area, bedrooms, and bathrooms
- Garage, basement, and exterior details
- Year built, remodel date, and sale information

The target variable is `SalePrice` (continuous, range: $34,900 — $755,000).

---

## Project Structure

```
HomeSense-AI/
├── app.py                      # Main Streamlit application (1,074 lines)
├── requirements.txt            # Python dependencies (7 packages)
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore rules
├── assets/
│   ├── style.css               # Premium luxury CSS theme (116 lines)
│   ├── hero_bg.jpg             # Hero section background image
│   └── houses/                 # 40 house images across 10 categories
│       ├── starter/            # 4 images
│       ├── cottage/            # 4 images
│       ├── ranch/              # 4 images
│       ├── townhouse/          # 4 images
│       ├── suburban/           # 6 images
│       ├── craftsman/          # 4 images
│       ├── colonial/           # 4 images
│       ├── modern/             # 4 images
│       ├── luxury/             # 4 images
│       └── mansion/            # 2 images
├── data/
│   ├── train.csv               # Training dataset (1,460 records)
│   ├── test.csv                # Test dataset
│   ├── sample_submission.csv   # Sample submission format
│   └── data_description.txt    # Feature documentation
├── model/
│   ├── house_price_model.pkl   # Trained Linear Regression model
│   └── model_columns.pkl       # 245 feature column names
├── utils/
│   ├── __init__.py             # All utility functions (1,930 lines)
│   └── pdf_report.py           # PDF report generation (263 lines)
├── notebooks/
│   ├── 01_Exploratory_Data_Analysis.ipynb
│   └── 02_Model_Training.ipynb
└── reports/
    └── .gitkeep
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip

### Steps

```bash
# Clone the repository
git clone https://github.com/yourusername/HomeSense-AI.git
cd HomeSense-AI

# Create virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Usage

1. **Landing Page** — Read the trust metrics and features overview
2. **Predict** — Enter property details (quality, area, garage, basement, year, neighborhood)
3. **View Report** — Review the AI verdict, value drivers, investment analysis, and scores
4. **Explore Charts** — Interact with feature importance, radar profile, and price distribution charts
5. **Review Recommendations** — See targeted improvements with estimated costs and ROI
6. **Download Report** — Export a professional 6-page PDF valuation report

---

## Deployment

### Streamlit Community Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set the main file path to `app.py`
5. Deploy

The app will be live at `https://your-app.streamlit.app`

### Important Notes

- The `data/train.csv` file (260 KB) is required at runtime for similar properties, feature importance, and dataset insights
- All model files are in `model/` — no external downloads needed
- No API keys or environment variables required

---

## Future Improvements

- [ ] Real-time MLS data integration
- [ ] Additional models (XGBoost, LightGBM ensemble)
- [ ] Historical price trend visualizations
- [ ] User authentication and saved predictions
- [ ] REST API for external integrations
- [ ] Advanced SHAP-based explainability
- [ ] Mobile-optimized responsive design

---

## License

This project is licensed under the MIT License.

---

## Developer

**Your Name**

- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Name](https://linkedin.com/in/yourusername)
- Portfolio: [yourportfolio.dev](https://yourportfolio.dev)

---

<div align="center">

**Built with Python, Machine Learning, and Streamlit.**

</div>
