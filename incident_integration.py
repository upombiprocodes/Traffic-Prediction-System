import pandas as pd

def add_incident_feature(df, incidents):
    df['event'] = 0
    for inc in incidents:
        near = ((abs(df['lat']-inc['lat']) < 0.01) & (abs(df['lon']-inc['lon']) < 0.01))
        df.loc[near, 'event'] = 1
    return df

if __name__ == "__main__":
    df = pd.read_csv("sample_data.csv")
    # Example incident data
    incidents = [{"lat":40.7150, "lon":-74.0070, "type":"accident"}, {"lat":40.7160, "lon":-74.0075, "type":"roadwork"}]
    df = add_incident_feature(df, incidents)
    print(df.head())
