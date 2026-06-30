import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Crop Production Predictor", page_icon="🌾", layout="wide")
# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 10px 0;
    }
    div[data-testid="stMetricValue"] {
        color: #2d8f3f;
    }
    .footer {
        text-align: center;
        color: gray;
        font-size: 13px;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)
# ---------- LOAD MODEL & DATA ----------
model = joblib.load('crop_model.pkl')
le_state = joblib.load('le_state.pkl')
le_season = joblib.load('le_season.pkl')
le_crop = joblib.load('le_crop.pkl')

with open('unique_values.json') as f:
    options = json.load(f)

with open('model_comparison.json') as f:
    comparison_results = json.load(f)

history_df = pd.read_csv('app_data.csv')

# ---------- SIDEBAR ----------
with st.sidebar:
    st.title("🌾 About This Project")
    st.write(
        "This app predicts **agricultural crop production (in tonnes)** "
        "across Indian states using a Machine Learning model trained on "
        "real government crop data (1997–2015)."
    )
    st.markdown("---")
    st.subheader("📊 Model Performance")
    st.metric("R² Score", "0.958")
    st.metric("Mean Absolute Error", "8,638 tonnes")
    st.markdown("---")
    st.subheader("🏆 Best Model")
    st.write("Random Forest Regressor (best of 3 models tested)")
    st.markdown("---")
    st.caption("Built with Python, scikit-learn & Streamlit")
    st.markdown("---")
    st.markdown("👩‍💻 **Developed by:**")
    st.markdown("**Jagruti Jiralikar**")
    st.caption("Data Science & ML Intern @ UpSkill Campus")

# ---------- HEADER ----------
st.title("🌾 Crop Production Prediction")
st.write("A Machine Learning powered tool to predict agricultural crop production across Indian states.")
st.markdown("---")

# ---------- TABS ----------
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📈 Historical Trends", "🧪 Model Comparison"])

# ===================== TAB 1: PREDICT =====================
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        state = st.selectbox("📍 Select State", options['states'])
        season = st.selectbox("🌦️ Select Season", options['seasons'])
        crop = st.selectbox("🌱 Select Crop", options['crops'])

    with col2:
        year = st.number_input("📅 Crop Year", min_value=1997, max_value=2030, value=2020)
        area = st.number_input("📐 Area (in hectares)", min_value=0.1, value=100.0)

    predict_btn = st.button("🔮 Predict Production", use_container_width=True)

    if predict_btn:
        state_enc = le_state.transform([state])[0]
        season_enc = le_season.transform([season])[0]
        crop_enc = le_crop.transform([crop])[0]

        input_data = np.array([[state_enc, year, season_enc, crop_enc, area]])
        prediction = model.predict(input_data)[0]

        # Estimate a simple uncertainty range using the forest's individual trees
        tree_preds = np.array([t.predict(input_data)[0] for t in model.estimators_])
        low, high = np.percentile(tree_preds, [10, 90])

        st.success(f"### 🌾 Predicted Production: **{prediction:,.2f} tonnes**")
        st.caption(f"Estimated range: {low:,.2f} – {high:,.2f} tonnes (based on model's internal variation)")
        st.balloons()

        st.markdown("#### Input Summary")
        summary_df = pd.DataFrame({
            "Field": ["State", "Year", "Season", "Crop", "Area (hectares)", "Predicted Production (tonnes)"],
            "Value": [state, year, season, crop, area, f"{prediction:,.2f}"]
        })
        st.table(summary_df)

        # Download button
        csv_data = summary_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download Prediction Report",
            data=csv_data,
            file_name="crop_prediction_report.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Feature importance (always shown)
    st.markdown("---")
    st.subheader("📈 What Drives the Prediction?")
    st.write("Based on the trained model, here's how much each factor influences crop production:")

    feature_names = ['State', 'Year', 'Season', 'Crop', 'Area']
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    st.bar_chart(feat_imp)

# ===================== TAB 2: HISTORICAL TRENDS =====================
with tab2:
    st.subheader("📈 Explore Historical Crop Data")
    st.write("See how production has historically trended for a selected state and crop.")

    hist_col1, hist_col2 = st.columns(2)
    with hist_col1:
        hist_state = st.selectbox("Select State for History", options['states'], key="hist_state")
    with hist_col2:
        hist_crop = st.selectbox("Select Crop for History", options['crops'], key="hist_crop")

    filtered = history_df[
        (history_df['State_Name'] == hist_state) & (history_df['Crop'] == hist_crop)
    ]

    if filtered.empty:
        st.warning("No historical data available for this State + Crop combination. Try another combination.")
    else:
        yearly = filtered.groupby('Crop_Year')['Production'].sum().sort_index()
        st.line_chart(yearly)

        avg_production = filtered['Production'].mean()
        max_production = filtered['Production'].max()
        total_records = len(filtered)

        m1, m2, m3 = st.columns(3)
        m1.metric("Average Production", f"{avg_production:,.0f} tonnes")
        m2.metric("Highest Recorded", f"{max_production:,.0f} tonnes")
        m3.metric("Records Found", total_records)

# ===================== TAB 3: MODEL COMPARISON =====================
with tab3:
    st.subheader("🧪 Why Random Forest?")
    st.write(
        "Three different machine learning models were trained and compared on the same data. "
        "Random Forest was selected as the final model because it achieved the highest accuracy."
    )

    comp_df = pd.DataFrame(comparison_results).T
    comp_df.index.name = "Model"
    comp_df = comp_df.reset_index()

    c1, c2 = st.columns(2)
    with c1:
        st.write("**R² Score (higher is better)**")
        st.bar_chart(comp_df.set_index("Model")["R2 Score"])
    with c2:
        st.write("**Mean Absolute Error (lower is better)**")
        st.bar_chart(comp_df.set_index("Model")["MAE"])

    st.markdown("#### Detailed Results")
    st.table(comp_df)

    st.info(
        "💡 **Insight:** Linear Regression performed poorly because the relationship between "
        "Area and Production is non-linear. Random Forest, an ensemble of decision trees, "
        "captured these complex patterns far more effectively."
    )
    st.markdown("---")
st.markdown(
    "<div class='footer'>© 2026 Jagruti Jiralikar | Crop Production Prediction Project</div>",
    unsafe_allow_html=True
)