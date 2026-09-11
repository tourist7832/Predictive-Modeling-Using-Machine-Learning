import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def evaluate_regression_model():
    rng = np.random.RandomState(42)
    X = rng.rand(300, 1) * 10
    y = 3 * X[:, 0] + 5 + rng.normal(0, 1, size=300)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2_score = model.score(X_test, y_test)

    plt.figure(figsize=(8, 5))
    plt.scatter(X_test, y_test, color="blue", label="Actual")
    plt.scatter(X_test, preds, color="red", alpha=0.6, label="Predicted")
    plt.title("Linear Regression Prediction")
    plt.xlabel("Input feature")
    plt.ylabel("Target")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "linear_regression_plot.png", dpi=150)
    plt.close()

    return {
        "Model": "Linear Regression",
        "R2 Score": round(float(r2_score), 4),
    }


def evaluate_classification_models():
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    metric_rows = []
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    roc_fig, roc_ax = plt.subplots(figsize=(8, 6))

    for idx, (name, model) in enumerate(models.items()):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(
            y_test,
            y_pred,
            target_names=["Malignant", "Benign"],
            output_dict=True,
        )

        metric_rows.append(
            {
                "Model": name,
                "Accuracy": round(float(accuracy), 4),
                "Precision": round(float(report["weighted avg"]["precision"]), 4),
                "Recall": round(float(report["weighted avg"]["recall"]), 4),
                "F1 Score": round(float(report["weighted avg"]["f1-score"]), 4),
                "Confusion Matrix": cm.tolist(),
            }
        )

        labels = ["Malignant", "Benign"]
        cm_display = confusion_matrix(y_test, y_pred)
        axes[idx].imshow(cm_display, interpolation="nearest", cmap="Blues")
        axes[idx].set_title(f"{name}\nConfusion Matrix")
        axes[idx].set_xticks([0, 1])
        axes[idx].set_yticks([0, 1])
        axes[idx].set_xticklabels(labels)
        axes[idx].set_yticklabels(labels)
        axes[idx].set_xlabel("Predicted label")
        axes[idx].set_ylabel("True label")

        for row in range(cm_display.shape[0]):
            for col in range(cm_display.shape[1]):
                axes[idx].text(
                    col,
                    row,
                    cm_display[row, col],
                    ha="center",
                    va="center",
                    color="white" if cm_display[row, col] > cm_display.max() / 2 else "black",
                )

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = roc_auc_score(y_test, y_prob)
            roc_ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
            metric_rows[-1]["ROC AUC"] = round(float(roc_auc), 4)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrices.png", dpi=150)
    plt.close(fig)

    roc_ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    roc_ax.set_title("ROC Curves for Classification Models")
    roc_ax.set_xlabel("False Positive Rate")
    roc_ax.set_ylabel("True Positive Rate")
    roc_ax.legend(loc="lower right")
    roc_ax.grid(alpha=0.3)
    roc_fig.tight_layout()
    roc_fig.savefig(RESULTS_DIR / "roc_curve.png", dpi=150)
    plt.close(roc_fig)

    return metric_rows


def main():
    regression_result = evaluate_regression_model()
    classification_results = evaluate_classification_models()

    print("\nRegression evaluation:")
    print(pd.DataFrame([regression_result]))

    print("\nClassification evaluation:")
    metrics_df = pd.DataFrame(classification_results)
    print(metrics_df)

    metrics_df.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)

    print(f"\nSaved plots to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
