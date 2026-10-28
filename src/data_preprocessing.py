
from pathlib import Path
from collections import Counter
import hashlib

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Configuration
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR / "dataset" / "Rice Leaf Disease Images"
RESULTS_DIR = PROJECT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
METRICS_DIR = RESULTS_DIR / "metrics"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RANDOM_STATE = 42

for folder in (PLOTS_DIR, METRICS_DIR):
    folder.mkdir(parents=True, exist_ok=True)


def sha256_file(path):
    """Return the SHA-256 hash of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def inspect_dataset():
    """Validate images and create an audit report."""
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found:\n{DATASET_DIR}\n"
            "Check the dataset path in this script."
        )

    class_folders = sorted(
        path for path in DATASET_DIR.iterdir() if path.is_dir()
    )

    if len(class_folders) < 2:
        raise ValueError(
            "Expected at least two class folders inside the dataset directory."
        )

    records = []
    invalid_files = []
    seen_hashes = {}

    for class_folder in class_folders:
        label = class_folder.name

        for path in sorted(class_folder.rglob("*")):
            if not path.is_file():
                continue

            if path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            try:
                # Verify that the file is a readable image.
                with Image.open(path) as image:
                    width, height = image.size
                    image.verify()

                if width <= 0 or height <= 0:
                    raise ValueError("Invalid image dimensions")

                file_hash = sha256_file(path)

            except (
                UnidentifiedImageError,
                OSError,
                ValueError,
            ) as error:
                invalid_files.append({
                    "path": str(path.relative_to(PROJECT_DIR)),
                    "error": str(error),
                })
                continue

            records.append({
                "path": str(path.relative_to(PROJECT_DIR)),
                "label": label,
                "width": width,
                "height": height,
                "sha256": file_hash,
            })

            seen_hashes.setdefault(file_hash, []).append({
                "path": str(path.relative_to(PROJECT_DIR)),
                "label": label,
            })

    if not records:
        raise ValueError(
            "No readable images found. Check the dataset folder and extensions."
        )

    # Save the image audit, including dimensions and hashes.
    audit_df = pd.DataFrame(records)
    audit_df.to_csv(METRICS_DIR / "image_audit.csv", index=False)

    pd.DataFrame(
        invalid_files, columns=["path", "error"]
    ).to_csv(METRICS_DIR / "invalid_images.csv", index=False)

    # Identify exact duplicate files.
    duplicate_rows = []
    excluded_hashes = set()

    for file_hash, items in seen_hashes.items():
        if len(items) <= 1:
            continue

        labels = {item["label"] for item in items}

        # Same pixels assigned to different classes: exclude all copies.
        if len(labels) > 1:
            excluded_hashes.add(file_hash)
            reason = "Duplicate image appears in multiple classes"
        else:
            # Same pixels and same class: retain one copy.
            reason = "Exact duplicate; retain first copy"

        for item in items:
            duplicate_rows.append({
                "path": item["path"],
                "label": item["label"],
                "sha256": file_hash,
                "reason": reason,
            })

    pd.DataFrame(
        duplicate_rows,
        columns=["path", "label", "sha256", "reason"],
    ).to_csv(METRICS_DIR / "duplicate_images.csv", index=False)

    # Keep one copy of each duplicate within a class.
    # Exclude hashes that have conflicting class labels.
    retained = []
    retained_hashes = set()

    for record in records:
        file_hash = record["sha256"]

        if file_hash in excluded_hashes:
            continue

        if file_hash in retained_hashes:
            continue

        retained_hashes.add(file_hash)
        retained.append(record)

    cleaned_df = pd.DataFrame(retained)

    # Summarize the classes after cleaning.
    summary = (
        cleaned_df.groupby("label")
        .size()
        .rename("images")
        .reset_index()
        .rename(columns={"label": "class"})
    )

    summary.to_csv(
        METRICS_DIR / "dataset_summary.csv", index=False
    )

    # Plot class distribution.
    plt.figure(figsize=(9, 5))
    plt.bar(summary["class"], summary["images"])
    plt.title("Rice Leaf Dataset: Class Distribution")
    plt.xlabel("Disease Class")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "class_distribution.png", dpi=200)
    plt.close()

    return audit_df, cleaned_df, invalid_files, duplicate_rows, summary


def create_splits(cleaned_df):
    """Create reproducible 70/15/15 stratified data manifests."""
    class_counts = cleaned_df["label"].value_counts()

    if len(class_counts) < 2 or class_counts.min() < 3:
        raise ValueError(
            "Each class needs at least three images for stratified splitting."
        )

    # 70% training; 30% held out for validation and testing.
    train_df, holdout_df = train_test_split(
        cleaned_df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=cleaned_df["label"],
    )

    # Divide the remaining 30% equally: 15% validation, 15% test.
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=holdout_df["label"],
    )

    columns = ["path", "label", "width", "height"]

    splits = {
        "train.csv": train_df,
        "validation.csv": val_df,
        "test.csv": test_df,
    }

    for filename, frame in splits.items():
        frame[columns].to_csv(
            METRICS_DIR / filename, index=False
        )

    # Save counts per class and split for the report.
    split_summary = pd.concat([
        frame.assign(split=name.removesuffix(".csv"))
        for name, frame in splits.items()
    ])

    split_counts = pd.crosstab(
        split_summary["label"], split_summary["split"]
    )

    split_counts.to_csv(METRICS_DIR / "split_class_counts.csv")

    print("\nSplit sizes:")
    for filename, frame in splits.items():
        print(f"{filename}: {len(frame)} images")

    print("\nClass counts by split:")
    print(split_counts.to_string())

    print("\nSaved split manifests in:", METRICS_DIR)


if __name__ == "__main__":
    audit, cleaned, invalid, duplicates, summary = inspect_dataset()

    print("\n========== DATASET AUDIT ==========")
    print("Readable images:", len(audit))
    print("Invalid/corrupted images:", len(invalid))
    print("Files involved in exact duplicates:", len(duplicates))
    print("Images retained after duplicate handling:", len(cleaned))

    print("\nClass distribution after cleaning:")
    print(summary.to_string(index=False))

    create_splits(cleaned)

    print("\nClass-distribution plot saved to:")
    print(PLOTS_DIR / "class_distribution.png")
    print("\nData preprocessing completed.")
