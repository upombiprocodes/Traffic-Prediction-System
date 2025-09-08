import pandas as pd
from sklearn.ensemble import IsolationForest

def detect_anomalies(df):
    clf = IsolationForest(contamination=0.1, random_state=42)
    df['anomaly'] = clf.fit_predict(df[["traffic_volume"]])
    df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})
    return df

if __name__ == "__main__":
    df = pd.read_csv("sample_data.csv")
    df = detect_anomalies(df)
    print(df[df['anomaly']==1])
