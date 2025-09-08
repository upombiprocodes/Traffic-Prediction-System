# Advanced Traffic Prediction System

## Features
- ML Model: Random Forest with time, weather, geospatial, and event data
- Incident Integration: Real-time event API (demo uses sample incidents)
- Geospatial Analysis: Hotspot detection and interactive maps
- Anomaly Detection: Time-series traffic anomaly flags
- Predictive Routing: Travel time recommendations
- Explainability & Dashboard: SHAP explainability, Streamlit dashboard
- Continuous Learning: Retraining with new data
- Weather/Environmental Impact: Extended weather features

## Setup

```bash
pip install pandas scikit-learn joblib requests folium shap streamlit
```

## Usage

1. Train the model:
    ```bash
    python traffic_predictor.py
    ```
2. Incident features:
    ```bash
    python incident_integration.py
    ```
3. Geospatial hotspot mapping:
    ```bash
    python geospatial_analysis.py
    ```
4. Anomaly detection:
    ```bash
    python anomaly_detection.py
    ```
5. Travel recommendations:
    ```bash
    python recommendation.py
    ```
6. Dashboard (browser):
    ```bash
    streamlit run explain_dashboard.py
    ```

## Example Output

- Model Validation MAE: ~50 vehicles
- Hotspot map: `hotspots_map.html`
- Dashboard: Interactive predictions and feature importances

## Extending
- Plug in real APIs for incidents, weather, and sensor data.
- Add more features to `sample_data.csv` for richer modeling.
- Connect to live traffic feeds for real-time prediction.

---

All modules are functional and ready for use.  
For real incident/event feeds, update the source in `incident_integration.py`.
