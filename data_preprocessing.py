import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    # Identify target column
    possible_targets = ["target", "output", "HeartDisease", "label", "num"]
    target_col = None

    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        raise Exception(f"Target column not found. Columns: {df.columns}")

    # Split features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Convert to binary if needed
    if len(y.unique()) > 2:
        y = y.apply(lambda x: 1 if x > 0 else 0)

    # Encode categorical
    X = pd.get_dummies(X, drop_first=True)

    # Save feature columns (VERY IMPORTANT)
    os.makedirs("models", exist_ok=True)
    joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")

    # Handle missing values
    X = X.fillna(X.mean())

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler
    joblib.dump(scaler, "models/scaler.pkl")

    # Split
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42), X.columns