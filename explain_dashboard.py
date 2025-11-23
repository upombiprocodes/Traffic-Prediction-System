import streamlit as st
import pandas as pd
import joblib
import shap
import folium
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import altair as alt

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

    # --- Global Location Management (Sidebar) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("📍 Location Settings")

    # Initialize Global Location State
    if "global_lat" not in st.session_state: st.session_state.global_lat = 40.7128
    if "global_lon" not in st.session_state: st.session_state.global_lon = -74.0060

    # GPS Tracker in Sidebar
    from streamlit_geolocation import streamlit_geolocation
    location = streamlit_geolocation()
    if location and location['latitude'] is not None:
        st.session_state.global_lat = location['latitude']
        st.session_state.global_lon = location['longitude']
        st.sidebar.success("Updated via GPS!")

    # Manual Input (Linked to Global State)
    # We use callbacks or just let the input update the state directly
    st.session_state.global_lat = st.sidebar.number_input("Latitude", value=st.session_state.global_lat, format="%.4f")
    st.session_state.global_lon = st.sidebar.number_input("Longitude", value=st.session_state.global_lon, format="%.4f")

    # Global API Key Management
    if "tomtom_key" not in st.session_state:
        st.session_state.tomtom_key = "SPcuC0jdg8et6Fjy6LKqgnUOwmPanb9z"
    
    with st.sidebar.expander("⚙️ API Settings"):
        st.session_state.tomtom_key = st.text_input("TomTom API Key", value=st.session_state.tomtom_key, type="password")

    tp = load_model_and_predictor()
    data = load_data()
    
    # Imports for real-time data
    from weather_integration import fetch_current_weather
    from tomtom_integration import fetch_real_time_incidents
    from datetime import datetime

    # Use global location variables for convenience
    lat = st.session_state.global_lat
    lon = st.session_state.global_lon

    if page == "Traffic Forecast":
        st.title("🚦 Real-time Traffic Forecast")
        st.markdown(f"Predicting for **{lat:.4f}, {lon:.4f}**")

        col1, col2 = st.columns(2)
        
        # Initialize session state for inputs if not set
        if "f_hour" not in st.session_state: st.session_state.f_hour = 12
        if "f_day" not in st.session_state: st.session_state.f_day = 0
        if "f_weather" not in st.session_state: st.session_state.f_weather = 1
        if "f_wind" not in st.session_state: st.session_state.f_wind = 10
        if "f_precip" not in st.session_state: st.session_state.f_precip = 0.0
        if "f_vis" not in st.session_state: st.session_state.f_vis = 10
        
        with col2:
            # Real-time Weather & Time Button
            if st.button("☁️ Fetch Real Weather & Time"):
                # 1. Fetch Weather
                real_weather = fetch_current_weather(lat, lon)
                if real_weather:
                    st.session_state.f_weather = real_weather.get("weather_condition", 1)
                    st.session_state.f_wind = int(real_weather.get("wind_speed", 10))
                    st.session_state.f_precip = float(real_weather.get("precipitation", 0.0))
                    # Clamp visibility to slider max (20) to avoid errors
                    st.session_state.f_vis = min(int(real_weather.get("visibility", 10)), 20)
                    st.success("Fetched real-time weather!")
                else:
                    st.error("Could not fetch weather.")
                
                # 2. Set Time to Now
                now = datetime.now()
                st.session_state.f_hour = now.hour
                st.session_state.f_day = now.weekday() # 0=Mon, 6=Sun
            
            wind = st.slider("Wind Speed (km/h)", 0, 50, key="f_wind")
            precip = st.slider("Precipitation (mm)", 0.0, 20.0, key="f_precip")
            visibility = st.slider("Visibility (km)", 0, 20, key="f_vis")
            pollution = st.slider("Pollution Index", 0, 100, 20)

        with col1:
            hour = st.slider("Hour of Day", 0, 23, key="f_hour")
            day = st.selectbox("Day of Week", [0, 1, 2, 3, 4, 5, 6], format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x], key="f_day")
            weather = st.selectbox("Weather Condition", [1, 2, 3, 4], format_func=lambda x: {1:"Clear", 2:"Cloudy", 3:"Rain", 4:"Snow"}.get(x, "Unknown"), key="f_weather")
            event = st.checkbox("Major Event Nearby?", value=False)

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
        st.markdown(f"Visualizing traffic hotspots near **{lat:.4f}, {lon:.4f}**")
        
        if st.button("Update Map"):
            # Fetch real incidents
            real_incidents = fetch_real_time_incidents(st.session_state.tomtom_key, lat, lon)
            
            # Create map centered on search
            m = folium.Map(location=[lat, lon], zoom_start=13)
            
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
        st.markdown(f"Real-time reports near **{lat:.4f}, {lon:.4f}**")
        
        if st.button("Scan for Incidents"):
            incidents = fetch_real_time_incidents(st.session_state.tomtom_key, lat, lon)
            if incidents:
                for inc in incidents:
                    with st.expander(f"{inc['type']} - {inc['severity']} Severity"):
                        st.write(f"**Description:** {inc['description']}")
                        st.write(f"**Location:** {inc['lat']:.4f}, {inc['lon']:.4f}")
                        
                        # Use Folium instead of st.map for better stability
                        m = folium.Map(location=[inc['lat'], inc['lon']], zoom_start=15)
                        folium.Marker(
                            [inc['lat'], inc['lon']],
                            popup=inc['description'],
                            icon=folium.Icon(color='red', icon='warning-sign')
                        ).add_to(m)
                        map_html = m._repr_html_()
                        components.html(map_html, height=300)
            else:
                st.info("No active incidents reported in this area.")
        else:
             st.info("Click 'Scan' to find real incidents near the coordinates.")

    elif page == "Live Monitor":
        st.title("📡 Live Traffic Monitor")
        st.markdown(f"Monitoring traffic at **{lat:.4f}, {lon:.4f}**")
        
        # Use global key
        api_key = st.session_state.tomtom_key
        
        if not api_key:
            st.info("No key provided. Running in **Simulation Mode**.")
        else:
            st.success("Connected to TomTom Network. Fetching **Real-World Data**.")
            
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
                    real_data = fetch_real_time_traffic(api_key, lat, lon)
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
                
                # Update chart data
                current_time = datetime.now().strftime("%H:%M:%S")
                st.session_state.live_data.append({
                    "Time": current_time,
                    "Volume": new_vol,
                    "Speed": new_speed
                })
                
                if len(st.session_state.live_data) > 30:
                    st.session_state.live_data.pop(0)
                
                # Create DataFrame
                df_chart = pd.DataFrame(st.session_state.live_data)
                
                # Altair Dual-Axis Chart
                base = alt.Chart(df_chart).encode(x='Time')

                line_vol = base.mark_area(opacity=0.3, color='red').encode(
                    y=alt.Y('Volume', axis=alt.Axis(title='Traffic Volume', titleColor='red'))
                )

                line_speed = base.mark_line(stroke='blue', interpolate='monotone').encode(
                    y=alt.Y('Speed', axis=alt.Axis(title='Speed (km/h)', titleColor='blue'))
                )

                c = alt.layer(line_vol, line_speed).resolve_scale(
                    y='independent'
                ).properties(
                    title="Real-time Traffic Volume vs Speed"
                )
                
                chart_placeholder.altair_chart(c, use_container_width=True)
                
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
