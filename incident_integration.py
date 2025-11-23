import pandas as pd
import random

def fetch_incidents():
    """
    Mock function to simulate fetching real-time incidents from an API.
    Returns a list of incident dictionaries.
    """
    # In a real app, this would call an external API (e.g., Google Maps, Waze)
    # For now, we generate some realistic-looking mock data around NYC coordinates
    incidents = [
        {"lat": 40.7128 + random.uniform(-0.01, 0.01), "lon": -74.0060 + random.uniform(-0.01, 0.01), "type": "Accident", "severity": "High", "description": "Multi-vehicle collision"},
        {"lat": 40.7150 + random.uniform(-0.01, 0.01), "lon": -74.0070 + random.uniform(-0.01, 0.01), "type": "Roadwork", "severity": "Medium", "description": "Lane closure for maintenance"},
        {"lat": 40.7180 + random.uniform(-0.01, 0.01), "lon": -74.0080 + random.uniform(-0.01, 0.01), "type": "Congestion", "severity": "Low", "description": "Heavy traffic due to rush hour"},
        {"lat": 40.7200 + random.uniform(-0.01, 0.01), "lon": -74.0100 + random.uniform(-0.01, 0.01), "type": "Hazard", "severity": "High", "description": "Debris on road"}
    ]
    return incidents

def add_incident_feature(df, incidents):
    """
    Adds an 'event' feature to the dataframe based on proximity to incidents.
    """
    df['event'] = 0
    if not incidents:
        return df
        
    for inc in incidents:
        # Simple proximity check (approx 1km radius)
        near = ((abs(df['lat'] - inc['lat']) < 0.01) & (abs(df['lon'] - inc['lon']) < 0.01))
        df.loc[near, 'event'] = 1
    return df

if __name__ == "__main__":
    try:
        df = pd.read_csv("sample_data.csv")
        incidents = fetch_incidents()
        print(f"Fetched {len(incidents)} incidents.")
        df = add_incident_feature(df, incidents)
        print(df.head())
    except FileNotFoundError:
        print("sample_data.csv not found. Skipping demo.")
