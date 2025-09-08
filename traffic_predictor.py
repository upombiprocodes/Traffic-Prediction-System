import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

class TrafficPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False

    def load_data(self, filepath):
        df = pd.read_csv(filepath)
        features = ["hour", "day_of_week", "weather", "lat", "lon", "event", "wind", "precip", "visibility", "pollution"]
        X = df[features]
        y = df["traffic_volume"]
        return X, y

    def train(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        self.is_trained = True
        print(f"Model Validation MAE: {mae:.2f}")

    def predict(self, X):
        if not self.is_trained:
            raise Exception("Model is not trained yet.")
        return self.model.predict(X)

    def save(self, path="traffic_model.pkl"):
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def load(self, path="traffic_model.pkl"):
        self.model = joblib.load(path)
        self.is_trained = True

if __name__ == "__main__":
    tp = TrafficPredictor()
    X, y = tp.load_data("sample_data.csv")
    tp.train(X, y)
    predictions = tp.predict(X.head())
    print("Sample predictions:", predictions)
    tp.save()
 
