import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Wine Quality Classifier",
    page_icon="🍷",
    layout="wide"
)

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "wine_quality_model.pkl"

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
features = artifact["features"]
classes = artifact["classes"]

# ---------------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------------
TEST_ACCURACY = 0.70
TEST_MACRO_F1 = 0.54
CV_MACRO_F1_MEAN = 0.527
CV_MACRO_F1_STD = 0.032

# ---------------------------------------------------------
# PAGE STYLE
# ---------------------------------------------------------
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 25px;
    }

    .metric-box {
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #ddd;
    }

    .footer {
        text-align: center;
        margin-top: 50px;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown(
    '<div class="main-title">🍷 Wine Quality Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning powered wine quality classification using '
    'Balanced Random Forest'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# ---------------------------------------------------------
# INFORMATION
# ---------------------------------------------------------
st.info(
    "Enter the chemical properties of a wine below. "
    "The trained machine learning model will classify the wine "
    "as Low, Average, or High quality."
)

# ---------------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------------
st.header("📈 Model Performance")

m1, m2, m3 = st.columns(3)

with m1:
    st.metric("Test Accuracy", "70%")

with m2:
    st.metric("Test Macro F1", "0.54")

with m3:
    st.metric("5-Fold CV Macro F1", "0.527 ± 0.032")

st.caption(
    "These are overall model evaluation metrics and do not represent "
    "the certainty of any individual prediction."
)

st.divider()

# ---------------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------------
st.header("🧪 Wine Chemical Properties")

col1, col2, col3 = st.columns(3)

with col1:
    fixed_acidity = st.number_input(
        "Fixed Acidity",
        min_value=0.0,
        value=7.4,
        step=0.1
    )

    volatile_acidity = st.number_input(
        "Volatile Acidity",
        min_value=0.0,
        value=0.70,
        step=0.01
    )

    citric_acid = st.number_input(
        "Citric Acid",
        min_value=0.0,
        value=0.00,
        step=0.01
    )

    residual_sugar = st.number_input(
        "Residual Sugar",
        min_value=0.0,
        value=1.9,
        step=0.1
    )

with col2:
    chlorides = st.number_input(
        "Chlorides",
        min_value=0.0,
        value=0.076,
        step=0.001
    )

    free_sulfur_dioxide = st.number_input(
        "Free Sulfur Dioxide",
        min_value=0.0,
        value=11.0,
        step=1.0
    )

    total_sulfur_dioxide = st.number_input(
        "Total Sulfur Dioxide",
        min_value=0.0,
        value=34.0,
        step=1.0
    )

    density = st.number_input(
        "Density",
        min_value=0.0,
        value=0.9978,
        step=0.0001,
        format="%.4f"
    )

with col3:
    pH = st.number_input(
        "pH",
        min_value=0.0,
        max_value=14.0,
        value=3.51,
        step=0.01
    )

    sulphates = st.number_input(
        "Sulphates",
        min_value=0.0,
        value=0.56,
        step=0.01
    )

    alcohol = st.number_input(
        "Alcohol",
        min_value=0.0,
        value=9.4,
        step=0.1
    )

st.divider()

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
predict_button = st.button(
    "🔍 Predict Wine Quality",
    use_container_width=True
)

if predict_button:

    input_data = pd.DataFrame([[
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH,
        sulphates,
        alcohol
    ]], columns=features)

    with st.spinner("Analyzing wine properties..."):

        prediction = model.predict(input_data)[0]

    st.divider()
    st.header("🎯 Prediction Result")

    if prediction == "High":

        st.success("🍷 Predicted Wine Quality: HIGH")

        st.markdown(
            '<div class="result-box">'
            '<h2>High Quality Wine</h2>'
            '<p>The model classified this wine as High quality.</p>'
            '</div>',
            unsafe_allow_html=True
        )

    elif prediction == "Average":

        st.warning("🍷 Predicted Wine Quality: AVERAGE")

        st.markdown(
            '<div class="result-box">'
            '<h2>Average Quality Wine</h2>'
            '<p>The model classified this wine as Average quality.</p>'
            '</div>',
            unsafe_allow_html=True
        )

    else:

        st.error("🍷 Predicted Wine Quality: LOW")

        st.markdown(
            '<div class="result-box">'
            '<h2>Low Quality Wine</h2>'
            '<p>The model classified this wine as Low quality.</p>'
            '</div>',
            unsafe_allow_html=True
        )

# ---------------------------------------------------------
# QUALITY INFORMATION
# ---------------------------------------------------------
st.divider()

st.header("📊 Quality Categories")

info1, info2, info3 = st.columns(3)

with info1:
    st.error("🔴 LOW")
    st.write("Wine quality category associated with quality scores 3–4.")

with info2:
    st.warning("🟡 AVERAGE")
    st.write("Wine quality category associated with quality scores 5–6.")

with info3:
    st.success("🟢 HIGH")
    st.write("Wine quality category associated with quality scores 7–8.")

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown(
    '<div class="footer">'
    'Wine Quality Classification System | '
    'Balanced Random Forest Machine Learning'
    '</div>',
    unsafe_allow_html=True
)