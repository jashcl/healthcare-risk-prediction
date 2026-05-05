from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

def evaluate(models, X_test, y_test):
    results = {}

    for name, model in models.items():
        preds = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
            roc = roc_auc_score(y_test, probs)
        else:
            roc = None

        acc = accuracy_score(y_test, preds)

        results[name] = {
            "accuracy": acc,
            "roc_auc": roc,
            "report": classification_report(y_test, preds)
        }

    return results


def calculate_improvement(results):
    base = results["logistic_regression"]["accuracy"]
    advanced = results["random_forest"]["accuracy"]

    improvement = ((advanced - base) / base) * 100
    return round(improvement, 2)