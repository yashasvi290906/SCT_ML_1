<div align="center">

# HomeSense AI

**AI-Powered Luxury Real Estate Valuation Platform**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

---

*A premium, explainable AI platform that predicts house prices using machine learning with full transparency in every valuation.*

</div>

---

## Overview

HomeSense AI is a full-stack machine learning application that predicts residential property prices using a Ridge Regression model trained on the Ames Housing Dataset. The platform features a luxury-grade UI with glassmorphism design, interactive Plotly charts, and explainable AI that shows users exactly why a property is valued at a given price.

**Key differentiator:** This is not a student dashboard — it's designed to look and feel like a real AI startup product.

---

## Features

- **AI Price Prediction** — Ridge Regression model trained on 80+ property features
- **Explainable AI** — Human-readable insights explaining every prediction
- **Property Quality Score** — Computed score rating overall property condition
- **Interactive Analytics** — Plotly charts including feature importance, radar profiles, and market comparisons
- **Luxury UI** — Dark chocolate & gold glassmorphism theme with smooth animations
- **Responsive Design** — Desktop-first, laptop-friendly layout
- **Zero LLM Dependency** — All explanations are generated from feature values, not external APIs

---

## Screenshots

> Screenshots will be added after deployment. The platform includes:

| Section | Description |
|---------|-------------|
| Hero | Animated floating card with glassmorphism navbar |
| Trust Metrics | Count-up style stat cards |
| Prediction Form | Organized input grid with sliders and selects |
| Result Dashboard | Large price display with quality score and badges |
| Explainable AI | Color-coded insight cards (positive / negative / neutral) |
| Charts | Feature importance, radar profile, market comparison |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit + Custom CSS (Glassmorphism) |
| ML Model | Ridge Regression (Scikit-learn) |
| Visualization | Plotly (Interactive Charts) |
| Data | Ames Housing Dataset (1,460 records, 80+ features) |
| Language | Python 3.10+ |
| Deployment | Streamlit Community Cloud |

---

## Dataset

**Ames Housing Dataset** — Compiled by Dean De Cock, this dataset contains 1,460 residential property records from Ames, Iowa with 80+ descriptive features including:

- Lot size, zoning, and configuration
- Overall quality and condition ratings
- Living area, bedrooms, bathrooms
- Garage, basement, and exterior details
- Year built, remodel date, and sale information

The target variable is `SalePrice` (continuous).

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip

### Steps

```bash
# Clone the repository
git clone https://github.com/yourusername/homeSense-ai.git
cd homeSense-ai

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Project Structure

```
SCT_ML_1/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── assets/
│   ├── style.css          # Premium luxury CSS theme
│   ├── hero_bg.jpg        # Hero section background
│   └── house.png          # House illustration
├── model/
│   ├── house_price_model.pkl   # Trained Ridge Regression model
│   └── model_columns.pkl       # Feature column reference
├── utils/
│   └── __init__.py        # Explanation engine & chart generators
├── data/
│   ├── train.csv          # Training dataset
│   ├── test.csv           # Test dataset
│   ├── sample_submission.csv
│   └── data_description.txt
├── notebooks/
│   ├── 01_Exploratory_Data_Analysis.ipynb
│   └── 02_Model_Training.ipynb
└── .gitignore
```

---

## Deployment

### Streamlit Community Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set the main file path to `app.py`
5. Deploy

The app will be live at `https://your-app.streamlit.app`

---

## Future Improvements

- [ ] Add XGBoost / LightGBM ensemble models
- [ ] Real-time market data integration
- [ ] PDF report generation for valuations
- [ ] User authentication and saved predictions
- [ ] Historical price trend charts
- [ ] Neighborhood comparison analytics
- [ ] Mobile-optimized responsive design
- [ ] A/B testing for model comparison
- [ ] REST API for external integrations

---

## Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | Ridge Regression |
| Training Samples | 1,460 |
| Features Used | 80+ |
| R² Score | ~0.92 |
| Cross-Validation | 5-Fold |
| Regularization | Ridge (L2) |

---

## Author

**Your Name**

- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Name](https://linkedin.com/in/yourusername)

---

<div align="center">

**Built with passion using Python, Machine Learning, and Streamlit.**

</div>
