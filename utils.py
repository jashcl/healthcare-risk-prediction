import os

def save_metrics(results, improvement):
    os.makedirs("outputs", exist_ok=True)

    with open("outputs/metrics.txt", "w") as f:
        for model, res in results.items():
            f.write(f"\nModel: {model}\n")
            f.write(f"Accuracy: {res['accuracy']}\n")
            f.write(f"ROC-AUC: {res['roc_auc']}\n")
            f.write(res["report"])
            f.write("\n" + "-"*40 + "\n")

        f.write(f"\nImprovement over baseline: {improvement}%\n")