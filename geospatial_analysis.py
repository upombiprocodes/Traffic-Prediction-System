import pandas as pd
from sklearn.cluster import DBSCAN
import folium

def find_hotspots(df):
    coords = df[["lat", "lon"]].values
    clustering = DBSCAN(eps=0.01, min_samples=2).fit(coords)
    df['hotspot'] = clustering.labels_
    return df

def visualize_hotspots(df, filename="hotspots_map.html"):
    m = folium.Map(location=[df["lat"].mean(), df["lon"].mean()], zoom_start=13)
    for _, row in df.iterrows():
        color = "red" if row['hotspot'] != -1 else "blue"
        folium.CircleMarker(location=[row['lat'], row['lon']], radius=5, color=color).add_to(m)
    m.save(filename)
    print(f"Hotspot map saved to {filename}")

if __name__ == "__main__":
    df = pd.read_csv("sample_data.csv")
    df = find_hotspots(df)
    visualize_hotspots(df)
