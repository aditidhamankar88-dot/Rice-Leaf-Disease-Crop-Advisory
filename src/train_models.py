
"""
Rice Leaf Disease & Crop Advisory System
Train and compare four ML models, save validation confusion matrices
for every model, and evaluate the best model on the test set.

Run from the project root:
    python -m src.train_models
"""

from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

from src.feature_extraction import extract_features


warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = ROOT_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = ROOT_DIR / "models"


TRAIN_CSV = METRICS_DIR / "train.csv"
VALIDATION_CSV = METRICS_DIR / "validation.csv"
TEST_CSV = METRICS_DIR / "test.csv"


for directory in [METRICS_DIR, PLOTS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load dataset split CSV files
# --------------------------------------------------

def find_column(df, possible_names, description):
    """Find a CSV column using a list of possible column names."""

    column_lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for name in possible_names:
        if name.lower() in column_lookup:
            return column_lookup[name.lower()]

    raise ValueError(
        f"Could not find the {description} column.\n"
        f"Available columns: {list(df.columns)}\n"
        f"Expected one of: {possible_names}"
    )


def resolve_image_path(value):
    """Resolve a dataset image path to an absolute path."""

    path = Path(str(value))

    if path.is_absolute() and path.exists():
        return path

    candidates = [
        ROOT_DIR / path,
        ROOT_DIR / "dataset" / path,
        ROOT_DIR / "dataset" / "Rice Leaf Disease Images" / path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Image not found: {value}\n"
        "Check the image paths in your train, validation, and test CSV files."
    )


def load_split(csv_path):
    """Load image paths and class labels from a split CSV."""

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Cannot find {csv_path}\n"
            "Run src/data_preprocessing.py first to generate the split CSV files."
        )

    df = pd.read_csv(csv_path)

    path_column = find_column(
        df,
        [
            "filepath",
            "file_path",
            "image_path",
            "path",
            "filename",
            "image",
        ],
        "image path",
    )

    label_column = find_column(
        df,
        [
            "label",
            "class",
            "class_name",
            "disease",
            "category",
        ],
        "class label",
    )

    df = df[[path_column, label_column]].dropna().copy()
    df.columns = ["filepath", "label"]

    df["filepath"] = df["filepath"].apply(resolve_image_path)
    df["label"] = df["label"].astype(str)

    return df.reset_index(drop=True)


# --------------------------------------------------
# 3. Extract or load image features
# --------------------------------------------------

def get_features(df, cache_path, split_name):
    """
    Extract features using the existing feature_extraction.py module.
    A cache is reused only when its ordered image paths match the CSV.
    """

    image_paths = [str(path) for path in df["filepath"]]

    if cache_path.exists():
        try:
            cached = joblib.load(cache_path)

            if (
                isinstance(cached, dict)
                and cached.get("paths") == image_paths
                and "features" in cached
            ):
                features = np.asarray(cached["features"])

                if len(features) == len(df):
                    print(f"Loaded cached {split_name} features.")
                    return features
        except Exception:
            pass

    print(f"Extracting features for {split_name} images...")

    feature_list = []

    for index, image_path in enumerate(df["filepath"], start=1):
        try:
            features = extract_features(str(image_path))
            feature_list.append(np.asarray(features, dtype=np.float32).ravel())
        except Exception as exc:
            raise RuntimeError(
                f"Feature extraction failed for image: {image_path}"
            ) from exc

        if index % 200 == 0 or index == len(df):
            print(f"  Processed {index}/{len(df)} images")

    X = np.vstack(feature_list).astype(np.float32)

    joblib.dump(
        {
            "paths": image_paths,
            "features": X,
        },
        cache_path,
    )

    print(f"Saved feature cache: {cache_path}")
    return X


# --------------------------------------------------
# 4. Calculate evaluation metrics
# --------------------------------------------------

def calculate_metrics(y_true, predictions):
    """Calculate weighted classification metrics."""

    return {
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_true,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
    }


# --------------------------------------------------
# 5. Save confusion matrix
# --------------------------------------------------

def save_confusion_matrix(
    y_true,
    predictions,
    class_labels,
    filename,
    title,
):
    """Create and save a confusion matrix image."""

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=class_labels,
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_labels,
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False,
        xticks_rotation=30,
    )

    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    fig.tight_layout()

    output_path = PLOTS_DIR / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved confusion matrix: {output_path}")


# --------------------------------------------------
# 6. Plot validation model comparison
# --------------------------------------------------

def save_model_comparison(results_df):
    """Plot validation accuracy and weighted F1-score."""

    model_names = results_df["model"].tolist()
    accuracy_values = results_df["accuracy"].tolist()
    f1_values = results_df["f1_score"].tolist()

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        x - width / 2,
        accuracy_values,
        width,
        label="Accuracy",
    )

    ax.bar(
        x + width / 2,
        f1_values,
        width,
        label="Weighted F1-score",
    )

    ax.set_title("Model Comparison on Validation Set")
    ax.set_xlabel("Machine Learning Model")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()

    output_path = PLOTS_DIR / "model_comparison_validation.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved model comparison plot: {output_path}")


# --------------------------------------------------
# 7. Main training function
# --------------------------------------------------

def main():
    print("=" * 65)
    print("RICE LEAF DISEASE CLASSIFICATION - MODEL TRAINING")
    print("=" * 65)

    # Load the three data splits.
    train_df = load_split(TRAIN_CSV)
    val_df = load_split(VALIDATION_CSV)
    test_df = load_split(TEST_CSV)

    print(f"\nTraining images:   {len(train_df)}")
    print(f"Validation images: {len(val_df)}")
    print(f"Test images:       {len(test_df)}")

    X_train = get_features(
        train_df,
        MODELS_DIR / "X_train_features.joblib",
        "training",
    )

    X_val = get_features(
        val_df,
        MODELS_DIR / "X_validation_features.joblib",
        "validation",
    )

    X_test = get_features(
        test_df,
        MODELS_DIR / "X_test_features.joblib",
        "test",
    )

    y_train = train_df["label"].to_numpy()
    y_val = val_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    class_labels = sorted(
        set(y_train) | set(y_val) | set(y_test)
    )

    if set(y_train) != set(class_labels):
        raise ValueError(
            "Training split must contain every class present in the dataset."
        )

    # Four models to compare.
    models = {
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5)),
        ]),

        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=3,
                    class_weight="balanced",
                ),
            ),
        ]),

        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=25,
            random_state=42,
        ),

        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]),
    }

    validation_results = []
    fitted_models = {}

    # --------------------------------------------------
    # Train every model and save its validation confusion matrix.
    # --------------------------------------------------

    for name, model in models.items():
        print("\n" + "-" * 65)
        print(f"Training model: {name}")
        print("-" * 65)

        model.fit(X_train, y_train)

        fitted_models[name] = model

        # Evaluate this model on the validation set.
        val_predictions = model.predict(X_val)
        metrics = calculate_metrics(y_val, val_predictions)

        result = {
            "model": name,
            **metrics,
        }
        validation_results.append(result)

        print(f"Validation accuracy:  {metrics['accuracy']:.4f}")
        print(f"Validation precision: {metrics['precision']:.4f}")
        print(f"Validation recall:    {metrics['recall']:.4f}")
        print(f"Validation F1-score:  {metrics['f1_score']:.4f}")

        # Create a separate confusion matrix for every model.
        safe_name = name.lower().replace(" ", "_")

        save_confusion_matrix(
            y_true=y_val,
            predictions=val_predictions,
            class_labels=class_labels,
            filename=f"confusion_matrix_{safe_name}_validation.png",
            title=f"Validation Confusion Matrix - {name}",
        )

    # --------------------------------------------------
    # 8. Compare models and select the best validation F1-score.
    # --------------------------------------------------

    results_df = pd.DataFrame(validation_results)
    results_df = results_df.sort_values(
        by="f1_score",
        ascending=False,
    ).reset_index(drop=True)

    comparison_path = (
        METRICS_DIR / "validation_model_comparison.csv"
    )
    results_df.to_csv(comparison_path, index=False)

    save_model_comparison(results_df)

    print("\n" + "=" * 65)
    print("VALIDATION MODEL COMPARISON")
    print("=" * 65)
    print(results_df.to_string(index=False, float_format="%.4f"))

    best_model_name = results_df.iloc[0]["model"]
    best_model = fitted_models[best_model_name]

    print(f"\nSelected model: {best_model_name}")
    print("Selection criterion: highest validation weighted F1-score")

    # --------------------------------------------------
    # 9. Evaluate only the selected model on the test set.
    # --------------------------------------------------

    test_predictions = best_model.predict(X_test)
    test_metrics = calculate_metrics(y_test, test_predictions)

    print("\n" + "=" * 65)
    print("FINAL TEST RESULTS")
    print("=" * 65)

    for metric_name, value in test_metrics.items():
        print(f"{metric_name}: {value:.4f}")

    # Save the selected model's test confusion matrix.
    save_confusion_matrix(
        y_true=y_test,
        predictions=test_predictions,
        class_labels=class_labels,
        filename="best_model_confusion_matrix.png",
        title=f"Test Confusion Matrix - {best_model_name}",
    )

    # Save per-class classification report.
    report = classification_report(
        y_test,
        test_predictions,
        labels=class_labels,
        target_names=class_labels,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()

    # Retain only individual disease classes.
    per_class_report = report_df.loc[class_labels]

    report_path = METRICS_DIR / "classification_report.csv"
    per_class_report.to_csv(report_path)

    print("\nPer-class test classification report:")
    print(per_class_report.to_string(float_format="%.4f"))

    # Save final test metrics as JSON.
    final_results = {
        "selected_model": best_model_name,
        "selection_metric": "validation_weighted_f1_score",
        "test_metrics": test_metrics,
        "test_samples": int(len(y_test)),
        "class_labels": class_labels,
    }

    metrics_json_path = METRICS_DIR / "final_test_metrics.json"

    with open(metrics_json_path, "w", encoding="utf-8") as file:
        json.dump(final_results, file, indent=4)

    # Save the model in the format expected by app.py.
    model_bundle = {
        "model": best_model,
        "feature_type": "RGB histogram + HOG",
        "class_labels": class_labels,
        "image_size": (128, 128),
        "selected_model": best_model_name,
    }

    model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(model_bundle, model_path)

    # --------------------------------------------------
    # 10. Final output locations
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("TRAINING COMPLETED")
    print("=" * 65)

    print("\nValidation confusion matrices:")
    for name in models:
        safe_name = name.lower().replace(" ", "_")
        print(
            PLOTS_DIR
            / f"confusion_matrix_{safe_name}_validation.png"
        )

    print("\nOther saved files:")
    print(f"Best model:             {model_path}")
    print(f"Validation comparison:  {comparison_path}")
    print(f"Validation plot:        {PLOTS_DIR / 'model_comparison_validation.png'}")
    print(f"Test confusion matrix:  {PLOTS_DIR / 'best_model_confusion_matrix.png'}")
    print(f"Classification report:  {report_path}")
    print(f"Final test metrics:     {metrics_json_path}")


if __name__ == "__main__":
    main()

