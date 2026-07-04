import io
import textwrap
from datetime import datetime
from utils import (
    get_quality_label, NEIGHBORHOOD_LABELS, DATASET_AVERAGES,
    compute_investment_scores, get_market_position, generate_recommendations,
    get_neighborhood_insights, compute_comparison,
    compute_property_report_scorecard, compute_improvement_simulator,
    generate_prediction_transparency, get_model_reliability_metrics,
)


def generate_pdf_report(inputs: dict, prediction: float, quality_score: float, ai_summary: str, scores: dict = None, forecast: dict = None, risk_analysis: dict = None) -> bytes:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        return b""

    buf = io.BytesIO()
    if scores is None:
        scores = compute_investment_scores(inputs, prediction)
    market = get_market_position(prediction)
    recs = generate_recommendations(inputs, prediction)
    comparison = compute_comparison(inputs)
    nb_insights = get_neighborhood_insights(inputs.get("Neighborhood", "CollgCr"))
    if forecast is None:
        from utils import compute_appreciation_forecast
        forecast = compute_appreciation_forecast(prediction, inputs.get("Neighborhood", "CollgCr"), inputs)
    if risk_analysis is None:
        from utils import compute_investment_risk
        risk_analysis = compute_investment_risk(inputs, prediction, scores)

    scorecard = compute_property_report_scorecard(inputs, prediction, scores)
    improvements = compute_improvement_simulator(inputs, prediction)
    transparency = generate_prediction_transparency(inputs, prediction)
    model_metrics = get_model_reliability_metrics()

    with PdfPages(buf) as pdf:
        # Page 1: Cover
        fig, ax = plt.subplots(figsize=(8.5, 11))
        fig.patch.set_facecolor("#0D0705")
        ax.set_facecolor("#0D0705")
        ax.axis("off")

        ax.text(0.5, 0.88, "HomeSense AI", fontsize=32, fontweight="bold",
                color="#D4A84E", ha="center", va="center", fontfamily="serif")
        ax.text(0.5, 0.82, "Premium Property Valuation Report", fontsize=14,
                color="#FAF2EB", ha="center", va="center", alpha=0.7)
        ax.axhline(y=0.78, xmin=0.3, xmax=0.7, color="#D4A84E", linewidth=0.5, alpha=0.4)

        ax.text(0.5, 0.70, f"${prediction:,.0f}", fontsize=42, fontweight="bold",
                color="#D4A84E", ha="center", va="center", fontfamily="serif")
        ax.text(0.5, 0.64, "Estimated Property Value", fontsize=12,
                color="#FAF2EB", ha="center", va="center", alpha=0.5)

        badge_text = "Premium" if prediction >= 350000 else "Strong Value" if prediction >= 200000 else "Accessible"
        ax.text(0.5, 0.58, badge_text, fontsize=11, fontweight="bold",
                color="#4ECB8D" if prediction >= 300000 else "#D4A84E",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#1E0C0A", edgecolor="#D4A84E", alpha=0.8))

        details = [
            f"Quality Score: {quality_score}/100",
            f"Investment Score: {scores['investment']}/100",
            f"Market Position: {market['tier']} ({market['percentile']}th percentile)",
            f"Neighborhood: {nb_insights['name']}",
            f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
        ]
        for i, detail in enumerate(details):
            ax.text(0.5, 0.46 - i * 0.04, detail, fontsize=10,
                    color="#FAF2EB", ha="center", va="center", alpha=0.6)

        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

        # Page 2: Summary & Scores
        fig, ax = plt.subplots(figsize=(8.5, 11))
        fig.patch.set_facecolor("#0D0705")
        ax.set_facecolor("#0D0705")
        ax.axis("off")

        ax.text(0.5, 0.94, "Executive Summary", fontsize=20, fontweight="bold",
                color="#D4A84E", ha="center", fontfamily="serif")

        wrapped = textwrap.fill(ai_summary, width=85)
        ax.text(0.08, 0.86, wrapped, fontsize=9.5, color="#FAF2EB",
                ha="left", va="top", alpha=0.75,
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#1E0C0A", edgecolor="none", alpha=0.5))

        ax.text(0.08, 0.52, "Investment Scores", fontsize=16, fontweight="bold",
                color="#D4A84E", fontfamily="serif", transform=ax.transAxes)

        score_labels = ["Investment", "Rental Potential", "Resale Value", "Luxury Score", "Market Safety"]
        score_vals = [scores["investment"], scores["rental"], scores["resale"], scores["luxury"], scores["risk"]]
        for i, (label, val) in enumerate(zip(score_labels, score_vals)):
            y = 0.46 - i * 0.06
            ax.text(0.08, y, f"{label}:", fontsize=10, color="#FAF2EB", alpha=0.7,
                    fontweight="bold", transform=ax.transAxes)
            ax.text(0.45, y, f"{val}/100", fontsize=10, color="#D4A84E",
                    fontweight="bold", transform=ax.transAxes)
            ax.barh(0.465 - i * 0.06, val / 100 * 0.45, height=0.018,
                    left=0.50, color="#D4A84E", alpha=0.6, transform=ax.transAxes)

        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

        # Page 3: Property Details & Comparison
        fig, ax = plt.subplots(figsize=(8.5, 11))
        fig.patch.set_facecolor("#0D0705")
        ax.set_facecolor("#0D0705")
        ax.axis("off")

        ax.text(0.5, 0.94, "Property Analysis", fontsize=20, fontweight="bold",
                color="#D4A84E", ha="center", fontfamily="serif")

        ax.text(0.08, 0.88, "Property Details", fontsize=14, fontweight="bold",
                color="#FAF2EB", fontfamily="serif", transform=ax.transAxes)
        prop_items = [
            ("Overall Quality", f"{inputs.get('OverallQual', 5)}/10 ({get_quality_label(inputs.get('OverallQual', 5))})"),
            ("Living Area", f"{inputs.get('GrLivArea', 1500):,} sq ft"),
            ("Garage", f"{inputs.get('GarageCars', 2)}-car capacity"),
            ("Basement", f"{inputs.get('TotalBsmtSF', 800):,} sq ft"),
            ("Bathrooms", f"{inputs.get('FullBath', 2)} full + {inputs.get('HalfBath', 0)} half"),
            ("Bedrooms", str(inputs.get("BedroomAbvGr", 3))),
            ("Year Built", str(inputs.get("YearBuilt", 2005))),
            ("Neighborhood", NEIGHBORHOOD_LABELS.get(inputs.get("Neighborhood", ""), inputs.get("Neighborhood", ""))),
        ]
        for i, (k, v) in enumerate(prop_items):
            y = 0.83 - i * 0.035
            ax.text(0.08, y, f"{k}:", fontsize=9.5, color="#FAF2EB", alpha=0.6, fontweight="bold", transform=ax.transAxes)
            ax.text(0.45, y, v, fontsize=9.5, color="#D4A84E", transform=ax.transAxes)

        ax.text(0.08, 0.52, "Dataset Comparison", fontsize=14, fontweight="bold",
                color="#FAF2EB", fontfamily="serif", transform=ax.transAxes)
        for i, comp in enumerate(comparison[:6]):
            y = 0.47 - i * 0.04
            diff_color = "#4ECB8D" if comp["better"] else "#FF6B6B"
            ax.text(0.08, y, comp["name"], fontsize=9, color="#FAF2EB", alpha=0.7, transform=ax.transAxes)
            ax.text(0.45, y, f"{comp['yours']}", fontsize=9, color="#D4A84E", fontweight="bold", transform=ax.transAxes)
            ax.text(0.60, y, f"vs {comp['average']}", fontsize=9, color="#FAF2EB", alpha=0.5, transform=ax.transAxes)
            ax.text(0.80, y, f"{'+' if comp['difference'] > 0 else ''}{comp['pct']:.1f}%", fontsize=9,
                    color=diff_color, fontweight="bold", transform=ax.transAxes)

        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

        # Page 4: Recommendations
        fig, ax = plt.subplots(figsize=(8.5, 11))
        fig.patch.set_facecolor("#0D0705")
        ax.set_facecolor("#0D0705")
        ax.axis("off")

        ax.text(0.5, 0.94, "Recommendations", fontsize=20, fontweight="bold",
                color="#D4A84E", ha="center", fontfamily="serif")

        for i, rec in enumerate(recs[:6]):
            y = 0.86 - i * 0.12
            ax.text(0.08, y, rec['title'], fontsize=12, fontweight="bold",
                    color="#FAF2EB", transform=ax.transAxes)
            ax.text(0.08, y - 0.035, textwrap.fill(rec["description"], width=75), fontsize=9.5, color="#FAF2EB",
                    alpha=0.6, transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#1E0C0A", edgecolor="none", alpha=0.4))
            ax.text(0.08, y - 0.075, rec["impact"], fontsize=9, color="#4ECB8D",
                    fontweight="bold", transform=ax.transAxes)

        ax.text(0.5, 0.06, "Generated by HomeSense AI  |  Powered by Machine Learning",
                fontsize=8, color="#FAF2EB", ha="center", alpha=0.3, transform=ax.transAxes)

        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

        # Page 5: Scorecard & Improvements
        fig, ax = plt.subplots(figsize=(8.5, 11))
        fig.patch.set_facecolor("#0D0705")
        ax.set_facecolor("#0D0705")
        ax.axis("off")

        ax.text(0.5, 0.94, "Property Scorecard", fontsize=20, fontweight="bold",
                color="#D4A84E", ha="center", fontfamily="serif")

        total_score = sum(sc["score"] for sc in scorecard)
        max_total = sum(sc["max"] for sc in scorecard)
        grade = "A" if total_score >= max_total * 0.8 else "B" if total_score >= max_total * 0.65 else "C" if total_score >= max_total * 0.5 else "D"

        ax.text(0.5, 0.86, f"Overall Grade: {grade} ({total_score}/{max_total})", fontsize=14,
                color="#D4A84E", ha="center", fontweight="bold", transform=ax.transAxes)

        for i, sc in enumerate(scorecard[:7]):
            y = 0.78 - i * 0.06
            bar_width = sc["score"] / sc["max"] * 0.35
            ax.text(0.08, y, f"{sc['category']}:", fontsize=10, color="#FAF2EB",
                    alpha=0.7, fontweight="bold", transform=ax.transAxes)
            ax.text(0.40, y, f"{sc['score']}/{sc['max']}", fontsize=10, color="#D4A84E",
                    fontweight="bold", transform=ax.transAxes)
            ax.barh(0.785 - i * 0.06, bar_width, height=0.018, left=0.48,
                    color="#D4A84E", alpha=0.6, transform=ax.transAxes)

        if improvements and improvements[0].get("feature") != "Well-Maintained":
            ax.text(0.08, 0.34, "Potential Improvements", fontsize=14, fontweight="bold",
                    color="#FAF2EB", fontfamily="serif", transform=ax.transAxes)
            for i, imp in enumerate(improvements[:4]):
                y = 0.28 - i * 0.055
                cost_text = f"${imp['cost_low']:,.0f}-${imp['cost_high']:,.0f}" if imp["cost_low"] > 0 else "N/A"
                ax.text(0.08, y, f"{imp['feature']}: {imp['description']}", fontsize=9,
                        color="#FAF2EB", alpha=0.7, transform=ax.transAxes)
                ax.text(0.08, y - 0.025, f"Cost: {cost_text}  |  Est. Value: +${imp['est_value']:,.0f}  |  ROI: {imp['roi']}%",
                        fontsize=8, color="#4ECB8D", fontweight="bold", transform=ax.transAxes)

        ax.text(0.5, 0.06, "Generated by HomeSense AI  |  Powered by Machine Learning",
                fontsize=8, color="#FAF2EB", ha="center", alpha=0.3, transform=ax.transAxes)

        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

        # Page 6: Forecast & Risk (renamed from Page 5)
        fig, ax = plt.subplots(figsize=(8.5, 11))
        fig.patch.set_facecolor("#0D0705")
        ax.set_facecolor("#0D0705")
        ax.axis("off")

        ax.text(0.5, 0.94, "Forecast & Risk Analysis", fontsize=20, fontweight="bold",
                color="#D4A84E", ha="center", fontfamily="serif")

        if forecast and "forecasts" in forecast:
            ax.text(0.08, 0.88, "Appreciation Forecast", fontsize=14, fontweight="bold",
                    color="#FAF2EB", fontfamily="serif", transform=ax.transAxes)
            for i, fc in enumerate(forecast["forecasts"]):
                y = 0.82 - i * 0.06
                ax.text(0.08, y, f"{fc['label']}:", fontsize=10, color="#FAF2EB",
                        alpha=0.7, fontweight="bold", transform=ax.transAxes)
                ax.text(0.35, y, f"${fc['value']:,.0f}", fontsize=10, color="#D4A84E",
                        fontweight="bold", transform=ax.transAxes)
                ax.text(0.60, y, f"+${fc['gain']:,.0f} (+{fc['gain_pct']}%)", fontsize=10,
                        color="#4ECB8D", fontweight="bold", transform=ax.transAxes)
            ax.text(0.08, 0.60, f"Annual Rate: {forecast.get('annual_rate', 'N/A')}%  |  Trend: {forecast.get('trend', 'N/A')}",
                    fontsize=9, color="#FAF2EB", alpha=0.5, transform=ax.transAxes)

        if risk_analysis:
            ax.text(0.08, 0.52, "Risk Assessment", fontsize=14, fontweight="bold",
                    color="#FAF2EB", fontfamily="serif", transform=ax.transAxes)
            risk_color = risk_analysis.get("risk_color", "#FFD060")
            ax.text(0.08, 0.46, f"Risk Level: {risk_analysis.get('risk_level', 'N/A')}  |  Score: {risk_analysis.get('risk_score', 0)}/100",
                    fontsize=10, color=risk_color, fontweight="bold", transform=ax.transAxes)
            for i, rf in enumerate(risk_analysis.get("factors", [])[:4]):
                y = 0.40 - i * 0.045
                ax.text(0.08, y, f"{rf['factor']}: {rf['detail']}",
                        fontsize=9, color="#FAF2EB", alpha=0.7, transform=ax.transAxes)
            if not risk_analysis.get("factors"):
                ax.text(0.08, 0.40, "No significant risk factors detected.", fontsize=10,
                        color="#4ECB8D", fontweight="bold", transform=ax.transAxes)

        ax.text(0.5, 0.06, "Generated by HomeSense AI  |  Powered by Machine Learning",
                fontsize=8, color="#FAF2EB", ha="center", alpha=0.3, transform=ax.transAxes)

        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

    return buf.getvalue()
