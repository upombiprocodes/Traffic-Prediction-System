# 🚦 AI Traffic Prediction System

> **A Next-Gen Traffic Intelligence Dashboard combining Historical AI Models with Real-Time TomTom Data.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)
![Status](https://img.shields.io/badge/Status-Active-success)

## 🌟 Overview

This application is a cutting-edge traffic monitoring and prediction system designed with a **Cyberpunk/SaaS aesthetic**. It seamlessly blends machine learning forecasts with live API data to provide accurate, actionable traffic insights.

Whether you are a city planner, a commuter, or a data enthusiast, this dashboard offers a futuristic window into urban mobility.

## ✨ Key Features

### 🧠 Dual-Core Intelligence
*   **AI Forecast Model**: Uses a Random Forest Regressor trained on historical data (Time, Weather, Events) to predict traffic volume trends.
*   **Real-Time Reality Check**: Instantly validates predictions against live TomTom API data (Speed, Congestion) for 100% accuracy.

### 🖥️ Modern UI/UX
*   **Cyberpunk Aesthetic**: Dark mode, neon accents, and glassmorphism cards.
*   **Interactive Background**: A dynamic, cursor-reactive particle network (CSS/JS) that brings the app to life.
*   **Lottie Animations**: Engaging motion graphics for a premium feel.

### 📡 Live Monitoring
*   **Real-Time Dashboard**: Monitors Volume, Speed, and Incidents with live updates.
*   **GPS Integration**: Automatically detects your location to provide hyper-local data.
*   **Incident Scanner**: Scans for accidents, roadworks, and jams within a 5km radius.

### 🗺️ Geospatial Visualization
*   **Hotspot Mapping**: Visualizes high-traffic zones using Folium heatmaps.
*   **Interactive Maps**: Drill down into specific incidents with detailed markers.

## 🛠️ Tech Stack

*   **Frontend**: Streamlit (Python)
*   **Data Processing**: Pandas, NumPy
*   **Machine Learning**: Scikit-Learn (Random Forest)
*   **Visualization**: Altair, Folium, Matplotlib
*   **APIs**: TomTom Traffic API, Open-Meteo (Weather), Streamlit Geolocation

## 🚀 Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/upombiprocodes/Traffic-Prediction-System.git
    cd Traffic-Prediction-System
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**
    ```bash
    streamlit run explain_dashboard.py
    ```

## 🔑 API Configuration

To unlock the full power of Real-Time features, you need a **TomTom API Key**.

1.  Get a free key from [developer.tomtom.com](https://developer.tomtom.com/).
2.  Launch the app.
3.  Open the **"⚙️ API Settings"** sidebar.
4.  Paste your key.

*Note: Without a key, the system runs in "Simulation Mode" using the AI model.*

## 📂 Project Structure

*   `explain_dashboard.py`: Main application entry point.
*   `traffic_predictor.py`: ML model training and inference logic.
*   `tomtom_integration.py`: Handles real-time API calls.
*   `traffic_model.pkl`: Pre-trained Random Forest model.

## 🤝 Contributing

Contributions are welcome! Please fork the repo and submit a pull request.

---
*Built with ❤️ by Antigravity*
