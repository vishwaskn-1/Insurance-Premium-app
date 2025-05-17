# app.py
import streamlit as st
import pandas as pd
import joblib
import shap

# Load trained pipeline
model_pipeline = joblib.load("model_pipeline.pkl")

st.set_page_config(page_title="Prediction App", layout="wide")

st.title("🩺 Prediction App")

# Sidebar - Input
st.sidebar.header("Enter User Information")
age = st.sidebar.slider("Age", 18, 100, 30)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
bmi = st.sidebar.slider("BMI", 15.0, 45.0, 25.0)
children = st.sidebar.slider("Number of Children", 0, 5, 0)
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
region = st.sidebar.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

# Input Summary
st.subheader("📝 Input Summary:")
input_df = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker],
    "region": [region],
})
st.write(input_df)

# Predict
prediction = model_pipeline.predict(input_df)[0]
st.success(f"💰 Estimated Annual Insurance Premium: ${prediction:.2f}")

# SHAP Explanation
st.subheader("🔍 SHAP Explanation:")
explainer = shap.Explainer(model_pipeline.named_steps["regressor"], model_pipeline.named_steps["preprocessor"].transform(input_df))
shap_values = explainer(model_pipeline.named_steps["preprocessor"].transform(input_df))

st.set_option('deprecation.showPyplotGlobalUse', False)
shap.plots.waterfall(shap_values[0], show=False)
import matplotlib.pyplot as plt
st.pyplot(plt.gcf())
