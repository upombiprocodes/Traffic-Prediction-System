import pandas as pd

def suggest_travel_times(df, threshold=0.5):
    good_times = df[df["traffic_volume"] < df["traffic_volume"].quantile(threshold)]
    return good_times[["hour", "day_of_week", "lat", "lon", "traffic_volume"]]

if __name__ == "__main__":
    df = pd.read_csv("sample_data.csv")
    tips = suggest_travel_times(df)
    print("Recommended travel times/routes:")
    print(tips)
