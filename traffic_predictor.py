import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os

class TrafficPredictor:
    def __init__(self, model_path="traffic_model.pkl"):
        self.model_path = model_path
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        self.features = ["hour", "day_of_week", "weather", "lat", "lon", "event", "wind", "precip", "visibility", "pollution"]
        
        if os.path.exists(self.model_path):
            try:
                self.load(self.model_path)
            except Exception as e:
                print(f"Could not load existing model: {e}")

    def load_data(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        df = pd.read_csv(filepath)
        # Ensure all required features exist
        missing_features = [f for f in self.features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features in data: {missing_features}")
            
        X = df[self.features]
        y = df["traffic_volume"]
        return X, y

    def train(self, X, y):
        print("Training model...")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        self.is_trained = True
        print(f"Model Validation MAE: {mae:.2f}")
        self.save()
        return mae

    def predict(self, X):
        if not self.is_trained:
            raise Exception("Model is not trained yet. Please train the model first.")
        
        # Ensure input has correct columns if it's a DataFrame
        if isinstance(X, pd.DataFrame):
            missing = [f for f in self.features if f not in X.columns]
            if missing:
                # If missing columns, try to fill with defaults or error? 
                # For now, let's error to be safe, or just select the ones we need if extra are present.
                pass 
            X = X[self.features]
            
        return self.model.predict(X)

    def save(self):
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")

    def load(self, path):
        self.model = joblib.load(path)
        self.is_trained = True
        print(f"Model loaded from {path}")

if __name__ == "__main__":
    tp = TrafficPredictor()
    try:
        if os.path.exists("sample_data.csv"):
            X, y = tp.load_data("sample_data.csv")
            tp.train(X, y)
            predictions = tp.predict(X.head())
            print("Sample predictions:", predictions)
        else:
            print("sample_data.csv not found. Skipping training demo.")
    except Exception as e:
        print(f"An error occurred: {e}")
 
