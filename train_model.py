from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_models(X_train, y_train):
    models = {}

    # Baseline model
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    models["logistic_regression"] = lr

    # Advanced model
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    models["random_forest"] = rf

    return models