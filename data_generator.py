import pandas as pd
import numpy as np
import random

def generate_traffic_data(n_samples=10000):
    print(f"Generating {n_samples} samples of synthetic traffic data...")
    
    # Base data
    hours = np.random.randint(0, 24, n_samples)
    days = np.random.randint(0, 7, n_samples)
    
    # Weather: 1=Clear, 2=Cloudy, 3=Rain, 4=Snow
    # Weighted probabilities: mostly clear/cloudy
    weather = np.random.choice([1, 2, 3, 4], n_samples, p=[0.5, 0.3, 0.15, 0.05])
    
    # Location (NYC area approx)
    lat_base = 40.7128
    lon_base = -74.0060
    lats = lat_base + np.random.normal(0, 0.05, n_samples)
    lons = lon_base + np.random.normal(0, 0.05, n_samples)
    
    # Events: 0=None, 1=Event
    events = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    
    # Environmental factors
    wind = np.random.randint(0, 40, n_samples)
    precip = np.random.uniform(0, 10, n_samples)
    visibility = np.random.randint(5, 20, n_samples)
    pollution = np.random.randint(10, 100, n_samples)
    
    # Calculate Traffic Volume based on rules
    traffic_volume = []
    
    for i in range(n_samples):
        base_vol = 500
        
        # Time of day effects (Rush hours)
        h = hours[i]
        if 7 <= h <= 9: # Morning rush
            base_vol += 1500
        elif 16 <= h <= 19: # Evening rush
            base_vol += 1800
        elif 10 <= h <= 15: # Mid-day
            base_vol += 800
        elif 0 <= h <= 5: # Late night
            base_vol -= 300
            
        # Weekend reduction
        if days[i] >= 5: # Sat/Sun
            base_vol *= 0.6
            
        # Weather impact
        w = weather[i]
        if w == 3: # Rain
            base_vol *= 0.9 # Slower/less traffic? Or more congestion? Let's say less volume but slower. 
                            # Actually volume might drop if people stay home.
        elif w == 4: # Snow
            base_vol *= 0.7
            
        # Event impact
        if events[i] == 1:
            base_vol += 1000
            
        # Random noise
        noise = np.random.normal(0, 200)
        final_vol = max(0, int(base_vol + noise))
        traffic_volume.append(final_vol)
        
    df = pd.DataFrame({
        "hour": hours,
        "day_of_week": days,
        "weather": weather,
        "lat": lats,
        "lon": lons,
        "event": events,
        "wind": wind,
        "precip": precip,
        "visibility": visibility,
        "pollution": pollution,
        "traffic_volume": traffic_volume
    })
    
    return df

if __name__ == "__main__":
    df = generate_traffic_data(15000)
    df.to_csv("traffic_data_large.csv", index=False)
    print("Saved to traffic_data_large.csv")
