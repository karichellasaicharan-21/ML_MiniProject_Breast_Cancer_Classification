# BREAST CANCER CLASSIFICATION
# Comparison of Logistic Regression, Decision Tree and Random Forest

# 1. Install dependencies using: pip install -r requirements.txt

# 2. Import libraries
from pathlib import Path
import json
import os
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "breast_cancer_classification_matplotlib_cache")
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# 3. Set parameters
RANDOM_STATE = 42
TEST_SIZE = 0.20

# 4. Create results folder
output_dir = Path("results")
output_dir.mkdir(parents=True, exist_ok=True)

# 5. Load dataset
dataset = load_breast_cancer(as_frame=True)
features = dataset.data
original_target = dataset.target

# 6. Convert target values
malignant_positive = (original_target == 0).astype(int)
target_names = {0: "Benign", 1: "Malignant"}

# 7. Count classes
class_counts = malignant_positive.map(target_names).value_counts().reindex(
    ["Benign", "Malignant"], fill_value=0
)

# 8. Dataset summary
summary = {
    "dataset": "Wisconsin Diagnostic Breast Cancer",
    "rows": int(features.shape[0]),
    "feature_count": int(features.shape[1]),
    "feature_names": list(features.columns),
    "target_mapping": {
        "0": "Benign (negative)",
        "1": "Malignant (positive)"
    },
    "class_counts": {
        name: int(count) for name, count in class_counts.items()
    },
    "missing_values_total": int(features.isna().sum().sum())
}

with (output_dir / "dataset_summary.json").open("w") as file:
    json.dump(summary, file, indent=2)

# 9. Save dataset preview
preview = features.head(10).copy()
preview["diagnosis"] = malignant_positive.head(10).map(target_names)
preview.to_csv(output_dir / "dataset_preview.csv", index=False)

# 10. Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    features,
    malignant_positive,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=malignant_positive
)

# 11. Standardize data for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 12. Create machine learning models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=5000, random_state=RANDOM_STATE
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE
    )
}

model_inputs = {
    "Logistic Regression": (X_train_scaled, X_test_scaled),
    "Decision Tree": (X_train, X_test),
    "Random Forest": (X_train, X_test)
}

# 13. Initialize result storage
metric_rows = []
reports = []
matrices = {}
roc_data = {}

# 14. Train and evaluate models
for model_name, model in models.items():

    train_features, test_features = model_inputs[model_name]

    model.fit(train_features, y_train)

    predictions = model.predict(test_features)
    positive_probabilities = model.predict_proba(test_features)[:, 1]

    metric_rows.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, pos_label=1),
        "Recall": recall_score(y_test, predictions, pos_label=1),
        "F1 Score": f1_score(y_test, predictions, pos_label=1),
        "ROC-AUC": roc_auc_score(y_test, positive_probabilities)
    })

    matrices[model_name] = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    )

    reports.append(
        f"{model_name}\n"
        f"{'=' * len(model_name)}\n"
        + classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Benign", "Malignant"],
            digits=4,
            zero_division=0
        )
    )

    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_test, positive_probabilities, pos_label=1
    )

    roc_data[model_name] = (
        false_positive_rate, true_positive_rate
    )

# 15. Create metrics table
metrics = pd.DataFrame(metric_rows)

metrics.to_csv(
    output_dir / "model_metrics.csv",
    index=False,
    float_format="%.6f"
)

# 16. Save classification reports
with (output_dir / "classification_reports.txt").open("w") as file:
    file.write(
        "Positive class: Malignant (1). Negative class: Benign (0).\n"
        f"Split: 80% training / 20% testing; stratified; "
        f"random_state={RANDOM_STATE}.\n\n"
        + "\n\n".join(reports)
    )

# 17. Plot class counts
plt.figure(figsize=(7, 4.5))

bars = plt.bar(
    class_counts.index,
    class_counts.values,
    color=["#4C78A8", "#E45756"]
)

plt.title("Class Counts in the Full Dataset")
plt.ylabel("Number of rows")
plt.xlabel("Diagnosis")
plt.bar_label(bars)
plt.tight_layout()
plt.savefig(output_dir / "class_counts.png", dpi=180)
plt.close()

# 18. Plot confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
display_labels = ["Benign", "Malignant"]

for axis, (model_name, matrix) in zip(axes, matrices.items()):

    axis.imshow(matrix, cmap="Blues")
    axis.set_title(model_name)
    axis.set_xticks([0, 1], labels=display_labels, rotation=25, ha="right")
    axis.set_yticks([0, 1], labels=display_labels)
    axis.set_xlabel("Predicted diagnosis")
    axis.set_ylabel("Actual diagnosis")

    for row in range(2):
        for column in range(2):
            axis.text(
                column, row, str(matrix[row, column]),
                ha="center", va="center"
            )

fig.suptitle("Confusion Matrices")
fig.tight_layout()
fig.savefig(
    output_dir / "confusion_matrices.png",
    dpi=180,
    bbox_inches="tight"
)
plt.close(fig)

# 19. Plot metric comparison
metric_columns = [
    "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"
]

metrics.set_index("Model")[metric_columns].plot(
    kind="bar",
    figsize=(10, 5),
    ylim=(0, 1.05),
    color=["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B"]
)

plt.title("Test-Set Metric Comparison")
plt.ylabel("Score (0 to 1)")
plt.xlabel("Model")
plt.xticks(rotation=0)
plt.legend(loc="lower right", ncol=2)
plt.tight_layout()
plt.savefig(output_dir / "metrics_comparison.png", dpi=180)
plt.close()

# 20. Plot ROC curves
plt.figure(figsize=(7, 5))

for model_name, (fpr, tpr) in roc_data.items():

    auc = metrics.loc[
        metrics["Model"] == model_name, "ROC-AUC"
    ].iloc[0]

    plt.plot(
        fpr, tpr,
        label=f"{model_name} (AUC = {auc:.3f})"
    )

plt.plot(
    [0, 1], [0, 1],
    linestyle="--", color="gray",
    label="Chance baseline"
)

plt.title("ROC Curves on the Test Set")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Malignant Recall)")
plt.legend(loc="lower right")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(output_dir / "roc_curves.png", dpi=180)
plt.close()

# 21. Display final output
print("BREAST CANCER CLASSIFICATION RESULTS")
print("------------------------------------")

print("\nDataset overview")
print("Rows:", features.shape[0])
print("Features:", features.shape[1])
print("Class counts:", class_counts.to_dict())
print("Missing values:", summary["missing_values_total"])

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nMODEL PERFORMANCE")
print(metrics.to_string(
    index=False,
    float_format=lambda value: f"{value:.4f}"
))

print("\nAll results and charts saved in:", output_dir.resolve())
