from src.data_preprocessing import load_data, preprocess
from src.train_model import train_models
from src.evaluate_model import evaluate, calculate_improvement
from src.utils import save_metrics

import joblib
import os
import matplotlib.pyplot as plt
import numpy as np

# Ensure folders exist
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)

# Load data
df = load_data("data/raw/heart.csv")

# Preprocess (UPDATED: returns feature names too)
(X_train, X_test, y_train, y_test), feature_names = preprocess(df)

# Train models
models = train_models(X_train, y_train)

# Evaluate models
results = evaluate(models, X_test, y_test)

# Calculate improvement
improvement = calculate_improvement(results)

# Print results
for model, res in results.items():
    print(f"\nModel: {model}")
    print("Accuracy:", round(res["accuracy"], 4))
    print("ROC-AUC:", round(res["roc_auc"], 4) if res["roc_auc"] else "N/A")

print(f"\nImprovement over baseline: {improvement}%")

# Save best model
best_model = models["random_forest"]
joblib.dump(best_model, "models/model.pkl")

# Save metrics
save_metrics(results, improvement)

# -------------------------------
# FEATURE IMPORTANCE (FINAL FIXED)
# -------------------------------

if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_

    # Get top 10 features
    indices = np.argsort(importances)[::-1][:10]

    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    # Plot
    plt.figure()
    plt.title("Top 10 Feature Importances")
    plt.bar(range(len(indices)), top_importances)
    plt.xticks(range(len(indices)), top_features, rotation=45, ha="right")
    plt.tight_layout()

    # Save plot
    plt.savefig("outputs/plots/feature_importance.png")
    plt.close()

    print("\nFeature importance plot saved to outputs/plots/")