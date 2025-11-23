import pandas as pd
from sklearn.cluster import DBSCAN
import folium

def find_hotspots(df):
    if df.empty:
        return df
    
    coords = df[["lat", "lon"]].values
    # Check if we have enough data for DBSCAN
    if len(coords) < 5:
        df['hotspot'] = -1
        return df

    try:
        clustering = DBSCAN(eps=0.01, min_samples=2).fit(coords)
        df['hotspot'] = clustering.labels_
    except Exception as e:
        print(f"Clustering failed: {e}")
        df['hotspot'] = -1
        
    return df

def visualize_hotspots(df):
    """
    Generates a Folium map with hotspots marked.
    Returns the map object.
    """
    if df.empty:
        return folium.Map(location=[40.7128, -74.0060], zoom_start=13)

    center_lat = df["lat"].mean()
    center_lon = df["lon"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
    
    for _, row in df.iterrows():
        # Only plot if we have hotspot info, default to blue if not a hotspot
        is_hotspot = row.get('hotspot', -1) != -1
        color = "red" if is_hotspot else "blue"
        
        popup_text = f"Traffic: {row.get('traffic_volume', 'N/A')}"
        
        folium.CircleMarker(
            location=[row['lat'], row['lon']], 
            radius=5, 
            color=color, 
            fill=True,
            fill_color=color,
            popup=popup_text
        ).add_to(m)
    return m

if __name__ == "__main__":
    try:
        df = pd.read_csv("sample_data.csv")
        df = find_hotspots(df)
        m = visualize_hotspots(df)
        m.save("hotspots_map.html")
        print("Hotspot map saved to hotspots_map.html")
    except FileNotFoundError:
        print("sample_data.csv not found.")
