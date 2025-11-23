import streamlit as st
import pandas as pd
import joblib
import shap
import folium
import streamlit.components.v1 as components
import matplotlib.pyplot as plt

# Local imports
from traffic_predictor import TrafficPredictor
from incident_integration import fetch_incidents, add_incident_feature
from geospatial_analysis import find_hotspots, visualize_hotspots

st.set_page_config(page_title="Traffic Prediction System", layout="wide")

# --- Helper Functions ---
@st.cache_resource
def load_model_and_predictor():
    tp = TrafficPredictor()
    # Ensure model is trained/loaded
    if not tp.is_trained:
        try:
            tp.load("traffic_model.pkl")
        except:
            # If load fails, try to train on sample data
            try:
                X, y = tp.load_data("sample_data.csv")
                tp.train(X, y)
            except Exception as e:
                st.error(f"Failed to initialize model: {e}")
    return tp

@st.cache_data
def load_data():
    try:
        return pd.read_csv("sample_data.csv")
    except:
        return pd.DataFrame()

# --- Main App ---
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Traffic Forecast", "Live Map", "Incidents", "Live Monitor", "Model Analysis"])

    # Global API Key Management
    if "tomtom_key" not in st.session_state:
        st.session_state.tomtom_key = "SPcuC0jdg8et6Fjy6LKqgnUOwmPanb9z"
    
    # Allow user to update key in sidebar
    with st.sidebar.expander("⚙️ Settings"):
        st.session_state.tomtom_key = st.text_input("TomTom API Key", value=st.session_state.tomtom_key, type="password")

    tp = load_model_and_predictor()
    data = load_data()
    
    # Imports for real-time data
    from weather_integration import fetch_current_weather
    from tomtom_integration import fetch_real_time_incidents

    if page == "Traffic Forecast":
        st.title("🚦 Real-time Traffic Forecast")
        st.markdown("Predict traffic volume based on current conditions.")

        col1, col2 = st.columns(2)
        
        with col1:
            hour = st.slider("Hour of Day", 0, 23, 12)
            day = st.selectbox("Day of Week", [0, 1, 2, 3, 4, 5, 6], format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])
            weather = st.selectbox("Weather Condition", [1, 2, 3, 4], format_func=lambda x: {1:"Clear", 2:"Cloudy", 3:"Rain", 4:"Snow"}.get(x, "Unknown"))
            event = st.checkbox("Major Event Nearby?", value=False)

        with col2:
            lat = st.number_input("Latitude", value=40.7128, format="%.4f")
            lon = st.number_input("Longitude", value=-74.0060, format="%.4f")
            
            # Real-time Weather Button
            if st.button("☁️ Fetch Real Weather"):
                real_weather = fetch_current_weather(lat, lon)
                if real_weather:
                    st.session_state.rw = real_weather
                    st.success("Fetched real-time weather!")
                else:
                    st.error("Could not fetch weather.")
            
            # Use fetched values if available
            rw = st.session_state.get("rw", {})
            
            wind = st.slider("Wind Speed (km/h)", 0, 50, int(rw.get("wind_speed", 10)))
            precip = st.slider("Precipitation (mm)", 0.0, 20.0, float(rw.get("precipitation", 0.0)))
            visibility = st.slider("Visibility (km)", 0, 20, int(rw.get("visibility", 10)))
            pollution = st.slider("Pollution Index", 0, 100, 20)
            
            # Update weather dropdown if real data fetched
            if rw:
                weather = rw.get("weather_condition", 1)

        if st.button("Predict Traffic Volume", type="primary"):
            # Prepare input
            input_data = pd.DataFrame({
                "hour": [hour],
                "day_of_week": [day],
                "weather": [weather],
                "lat": [lat],
                "lon": [lon],
                "event": [1 if event else 0],
                "wind": [wind],
                "precip": [precip],
                "visibility": [visibility],
                "pollution": [pollution]
            })
            
            try:
                prediction = tp.predict(input_data)[0]
                st.success(f"Predicted Traffic Volume: **{int(prediction)} vehicles/hr**")
                
                # Contextual interpretation
                if prediction < 1000:
                    st.info("Traffic is light. Good time to travel!")
                elif prediction < 2000:
                    st.warning("Traffic is moderate. Expect some delays.")
                else:
                    st.error("Traffic is heavy. Avoid this route if possible.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    elif page == "Live Map":
        st.title("🗺️ Live Traffic Hotspots")
        st.markdown("Visualizing traffic hotspots and congestion areas.")
        
        # Search for location
        search_lat = st.number_input("Center Latitude", value=40.7128, format="%.4f")
        search_lon = st.number_input("Center Longitude", value=-74.0060, format="%.4f")
        
        if st.button("Update Map"):
            # Fetch real incidents
            real_incidents = fetch_real_time_incidents(st.session_state.tomtom_key, search_lat, search_lon)
            
            # Create map centered on search
            m = folium.Map(location=[search_lat, search_lon], zoom_start=13)
            
            # Plot incidents
            if real_incidents:
                for inc in real_incidents:
                    folium.Marker(
                        [inc['lat'], inc['lon']],
                        popup=f"{inc['type']}: {inc['description']}",
                        icon=folium.Icon(color='red', icon='info-sign')
                    ).add_to(m)
                st.success(f"Found {len(real_incidents)} real-time incidents!")
            else:
                st.info("No active incidents found in this area.")
                
            # Render map
            map_html = m._repr_html_()
            components.html(map_html, height=600)
        else:
            # Default view
            if not data.empty:
                df_hotspots = find_hotspots(data.copy())
                m = visualize_hotspots(df_hotspots)
                map_html = m._repr_html_()
                components.html(map_html, height=600)

    elif page == "Incidents":
        st.title("🚨 Active Incidents")
        st.markdown("Real-time reports of accidents, roadworks, and hazards.")
        
        col_lat, col_lon = st.columns(2)
        i_lat = col_lat.number_input("Latitude", value=40.7128, format="%.4f", key="inc_lat")
        i_lon = col_lon.number_input("Longitude", value=-74.0060, format="%.4f", key="inc_lon")
        
        if st.button("Scan for Incidents"):
            incidents = fetch_real_time_incidents(st.session_state.tomtom_key, i_lat, i_lon)
            if incidents:
                for inc in incidents:
                    with st.expander(f"{inc['type']} - {inc['severity']} Severity"):
                        st.write(f"**Description:** {inc['description']}")
                        st.write(f"**Location:** {inc['lat']:.4f}, {inc['lon']:.4f}")
                        st.map(pd.DataFrame([inc]))
            else:
                st.info("No active incidents reported in this area.")
        else:
             st.info("Click 'Scan' to find real incidents near the coordinates.")

    elif page == "Live Monitor":
        st.title("📡 Live Traffic Monitor")
        st.markdown("Real-time traffic data feed.")
        
        # Use global key
        api_key = st.session_state.tomtom_key
        
        if not api_key:
            st.info("No key provided. Running in **Simulation Mode**.")
        else:
            st.success("Connected to TomTom Network. Fetching **Real-World Data**.")
            
        # Location Input for Real Data
        col_lat, col_lon = st.columns(2)
        monitor_lat = col_lat.number_input("Monitor Latitude", value=40.7128, format="%.4f", key="mon_lat")
        monitor_lon = col_lon.number_input("Monitor Longitude", value=-74.0060, format="%.4f", key="mon_lon")

        # Dashboard metrics
        col1, col2, col3, col4 = st.columns(4)
        metric_vol = col1.empty()
        metric_speed = col2.empty()
        metric_incidents = col3.empty()
        metric_status = col4.empty()
        
        # Chart placeholder
        chart_placeholder = st.empty()
        
        # Simulation loop
        import time
        import numpy as np
        from tomtom_integration import fetch_real_time_traffic
        
        if "live_data" not in st.session_state:
            st.session_state.live_data = []

        if st.button("Start Monitoring"):
            st.info("Monitoring started... (Press Stop to end)")
            stop_btn = st.button("Stop")
            
            while not stop_btn:
                # Default simulated values
                new_vol = int(np.random.normal(1500, 300))
                new_speed = int(np.random.normal(45, 10))
                active_incidents = int(np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1]))
                status = "Normal"
                
                # Fetch Real Data if Key is present
                if api_key:
                    real_data = fetch_real_time_traffic(api_key, monitor_lat, monitor_lon)
                    if real_data:
                        new_speed = real_data['current_speed']
                        # Estimate volume from congestion (inverse relationship approx)
                        congestion = real_data['congestion_level']
                        new_vol = int(500 + (congestion * 20)) # Rough proxy
                        
                        if congestion > 50: status = "Congested"
                        elif congestion < 10: status = "Free Flow"
                        else: status = "Normal"
                        
                        # Incidents not in this specific endpoint, keep simulated or 0
                        active_incidents = 0 
                else:
                    # Simulation logic
                    if new_vol > 2000: status = "Congested"
                    elif new_vol < 500: status = "Free Flow"

                # Update metrics
                metric_vol.metric("Volume (Est)", f"{new_vol} veh/hr", delta=f"{np.random.randint(-50, 50)}")
                metric_speed.metric("Avg Speed", f"{new_speed} km/h", delta=f"{np.random.randint(-5, 5)}")
                metric_incidents.metric("Active Incidents", f"{active_incidents}", delta_color="inverse")
                
                if status == "Congested":
                    metric_status.error(status)
                else:
                    metric_status.success(status)
                
                # Update chart
                st.session_state.live_data.append(new_vol)
                if len(st.session_state.live_data) > 50:
                    st.session_state.live_data.pop(0)
                    
                chart_placeholder.line_chart(st.session_state.live_data)
                
                time.sleep(2) # Slower update for API rate limits
                
    elif page == "Model Analysis":
        st.title("📊 Model Explainability")
        st.markdown("Understanding how the model makes predictions.")
        
        if tp.is_trained and not data.empty:
            st.subheader("Feature Importance (SHAP)")
            st.write("This plot shows which features have the biggest impact on traffic predictions.")
            
            # Calculate SHAP values (using a subset for speed)
            X_sample = data[tp.features].sample(min(100, len(data)))
            explainer = shap.TreeExplainer(tp.model)
            shap_values = explainer.shap_values(X_sample)
            
            # Plot
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_sample, show=False)
            st.pyplot(plt.gcf())
            plt.clf() # Clear figure
        else:
            st.warning("Model not trained or no data available for analysis.")

if __name__ == "__main__":
    main()
