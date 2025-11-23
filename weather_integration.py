import requests

def fetch_current_weather(lat, lon):
    """
    Fetches current weather data from Open-Meteo API.
    Returns a dictionary with temperature, wind, precip, visibility, etc.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,precipitation,rain,showers,snowfall,weather_code,wind_speed_10m",
        "hourly": "visibility",
        "forecast_days": 1
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        current = data.get("current", {})
        hourly = data.get("hourly", {})
        
        # Map WMO weather codes to our simple 1-4 scale
        # 1=Clear, 2=Cloudy, 3=Rain, 4=Snow
        wmo_code = current.get("weather_code", 0)
        weather_condition = 1 # Default Clear
        
        if wmo_code in [0, 1]: weather_condition = 1 # Clear
        elif wmo_code in [2, 3, 45, 48]: weather_condition = 2 # Cloudy
        elif wmo_code in [51, 53, 55, 61, 63, 65, 80, 81, 82]: weather_condition = 3 # Rain
        elif wmo_code in [71, 73, 75, 77, 85, 86]: weather_condition = 4 # Snow
        
        # Visibility is hourly, take the first one (current hour approx)
        vis_km = 10 # Default
        if "visibility" in hourly and hourly["visibility"]:
            vis_km = hourly["visibility"][0] / 1000.0
            
        return {
            "temperature": current.get("temperature_2m", 20),
            "wind_speed": current.get("wind_speed_10m", 10),
            "precipitation": current.get("precipitation", 0.0),
            "weather_condition": weather_condition,
            "visibility": vis_km,
            "wmo_code": wmo_code
        }
        
    except Exception as e:
        print(f"Open-Meteo API Error: {e}")
        return None

if __name__ == "__main__":
    # Test for NYC
    print(fetch_current_weather(40.7128, -74.0060))
