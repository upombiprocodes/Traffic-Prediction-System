import pandas as pd
import shap
import joblib
import streamlit as st

def explain_model(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values

def dashboard():
    st.title("Traffic Prediction Dashboard")
    df = pd.read_csv("sample_data.csv")
    model = joblib.load("traffic_model.pkl")
    st.write("Sample Data", df.head())
    preds = model.predict(df[["hour", "day_of_week", "weather", "lat", "lon", "event", "wind", "precip", "visibility", "pollution"]])
    df['prediction'] = preds
    st.write("Predictions", df[['traffic_volume', 'prediction']])
    shap_values = explain_model(model, df[["hour", "day_of_week", "weather", "lat", "lon", "event", "wind", "precip", "visibility", "pollution"]])
    st.write("Feature Importance")
    shap.summary_plot(shap_values, df[["hour", "day_of_week", "weather", "lat", "lon", "event", "wind", "precip", "visibility", "pollution"]], show=False)

if __name__ == "__main__":
    dashboard()
