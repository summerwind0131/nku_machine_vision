import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report


DEFAULT_CLASSES = ["A", "B", "C", "Five", "Point", "V"]
LABEL_PATTERNS = [
    (re.compile(r"^test_A", re.IGNORECASE), "A"),
    (re.compile(r"^test_B", re.IGNORECASE), "B"),
    (re.compile(r"^test_C", re.IGNORECASE), "C"),
    (re.compile(r"^test_Sign\s*5", re.IGNORECASE), "Five"),
    (re.compile(r"^test_Five", re.IGNORECASE), "Five"),
    (re.compile(r"^test_Point", re.IGNORECASE), "Point"),
    (re.compile(r"^test_V", re.IGNORECASE), "V"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze prediction output and create a confusion matrix."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/result.txt"),
        help="Prediction result file from test.py."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/evaluation"),
        help="Folder for generated evaluation reports."
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        help="Class order for reports and confusion matrix."
    )
    return parser.parse_args()


def infer_label(filename):
    for pattern, label in LABEL_PATTERNS:
        if pattern.search(filename):
            return label
    return None


def read_predictions(path):
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                filename, prediction = line.rsplit(maxsplit=1)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid prediction line {line_number}: {line!r}"
                ) from exc

            true_label = infer_label(filename)
            rows.append({
                "filename": filename,
                "true_label": true_label,
                "prediction": prediction,
            })
    return rows


def save_confusion_csv(rows, classes, output_path):
    counts = Counter(
        (row["true_label"], row["prediction"])
        for row in rows
        if row["true_label"] is not None
    )

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true_label", *classes])
        for true_label in classes:
            writer.writerow([
                true_label,
                *[counts[(true_label, prediction)] for prediction in classes],
            ])


def save_confusion_plot(y_true, y_pred, classes, output_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=classes,
        display_labels=classes,
        cmap="Blues",
        xticks_rotation=45,
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Prediction Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_summary(rows, classes, predictions_path, output_path):
    labeled_rows = [
        row for row in rows
        if row["true_label"] is not None
    ]
    y_true = [row["true_label"] for row in labeled_rows]
    y_pred = [row["prediction"] for row in labeled_rows]

    correct = sum(true == pred for true, pred in zip(y_true, y_pred))
    total = len(y_true)
    accuracy = correct / total if total else 0.0

    report = classification_report(
        y_true,
        y_pred,
        labels=classes,
        zero_division=0,
    )

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(f"Prediction file: {predictions_path}\n")
        file.write(f"Labeled images: {total}\n")
        file.write(f"Correct: {correct}\n")
        file.write(f"Accuracy: {accuracy:.4f}\n\n")
        file.write(report)
        file.write("\n")

    return y_true, y_pred, accuracy


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.predictions.exists():
        raise FileNotFoundError(f"Prediction file not found: {args.predictions}")

    rows = read_predictions(args.predictions)
    labeled_rows = [
        row for row in rows
        if row["true_label"] is not None
    ]

    if not labeled_rows:
        raise ValueError(
            "No labels could be inferred from filenames. "
            "Expected names like test_A1.png, test_C1.png, or test_Sign 5 (1).png."
        )

    y_true, y_pred, accuracy = save_summary(
        rows,
        args.classes,
        args.predictions,
        args.output_dir / "summary.txt",
    )
    save_confusion_csv(
        rows,
        args.classes,
        args.output_dir / "confusion_matrix.csv",
    )
    save_confusion_plot(
        y_true,
        y_pred,
        args.classes,
        args.output_dir / "confusion_matrix.png",
    )

    print(f"Analyzed {len(y_true)} labeled predictions.")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Saved evaluation reports to {args.output_dir}")


if __name__ == "__main__":
    main()
