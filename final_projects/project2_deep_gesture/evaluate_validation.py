import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.resnet_model import build_resnet18


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the trained checkpoint on the validation split."
    )
    parser.add_argument(
        "--val_dir",
        type=Path,
        default=Path("data/val"),
        help="Validation dataset folder in ImageFolder format."
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path("checkpoints/best_model.pth"),
        help="Trained checkpoint path."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/validation_evaluation"),
        help="Folder for validation evaluation reports."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for validation inference."
    )
    return parser.parse_args()


def get_validation_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_checkpoint_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint.get("classes")

    if classes is None and "class_to_idx" in checkpoint:
        class_to_idx = checkpoint["class_to_idx"]
        classes = [
            class_name
            for class_name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])
        ]

    if classes is None:
        raise ValueError("Checkpoint must include 'classes' or 'class_to_idx'.")

    model = build_resnet18(num_classes=len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, classes


def run_validation(model, dataloader, device):
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1).cpu()

            y_true.extend(labels.tolist())
            y_pred.extend(predictions.tolist())

    return y_true, y_pred


def save_confusion_matrix(y_true, y_pred, classes, output_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        display_labels=classes,
        cmap="Blues",
        xticks_rotation=45,
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Validation Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_per_class_accuracy(y_true, y_pred, classes, output_path):
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
    )
    totals = matrix.sum(axis=1)
    correct = matrix.diagonal()
    accuracies = [
        (correct[index] / totals[index]) if totals[index] else 0.0
        for index in range(len(classes))
    ]

    fig, ax = plt.subplots(figsize=(9, 5.6))
    bars = ax.bar(classes, accuracies, color="#2f7ebc")
    ax.set_ylim(0, 1.18)
    ax.set_title("Validation Per-Class Accuracy", pad=18)
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy")
    ax.grid(axis="y", alpha=0.25)

    for bar, accuracy, total in zip(bars, accuracies, totals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(accuracy + 0.025, 1.08),
            f"{accuracy:.3f}\n(n={int(total)})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return correct, totals, accuracies


def save_confusion_csv(y_true, y_pred, classes, output_path):
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
    )

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true_label", *classes])
        for class_name, row in zip(classes, matrix):
            writer.writerow([class_name, *row.tolist()])


def save_summary(correct, totals, accuracies, classes, output_path):
    total_correct = int(sum(correct))
    total = int(sum(totals))
    overall_accuracy = total_correct / total if total else 0.0

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(f"Validation images: {total}\n")
        file.write(f"Correct: {total_correct}\n")
        file.write(f"Accuracy: {overall_accuracy:.4f}\n\n")
        file.write("Per-class accuracy:\n")
        for class_name, class_correct, class_total, accuracy in zip(
            classes,
            correct,
            totals,
            accuracies,
        ):
            file.write(
                f"{class_name}: {int(class_correct)}/{int(class_total)} "
                f"({accuracy:.4f})\n"
            )

    return overall_accuracy


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = datasets.ImageFolder(args.val_dir, transform=get_validation_transform())
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model, checkpoint_classes = load_checkpoint_model(args.model_path, device)
    if list(dataset.classes) != list(checkpoint_classes):
        raise ValueError(
            "Validation dataset classes do not match checkpoint classes. "
            f"Dataset: {dataset.classes}; checkpoint: {checkpoint_classes}"
        )

    y_true, y_pred = run_validation(model, dataloader, device)
    save_confusion_matrix(
        y_true,
        y_pred,
        dataset.classes,
        args.output_dir / "validation_confusion_matrix.png",
    )
    correct, totals, accuracies = save_per_class_accuracy(
        y_true,
        y_pred,
        dataset.classes,
        args.output_dir / "validation_per_class_accuracy.png",
    )
    save_confusion_csv(
        y_true,
        y_pred,
        dataset.classes,
        args.output_dir / "validation_confusion_matrix.csv",
    )
    overall_accuracy = save_summary(
        correct,
        totals,
        accuracies,
        dataset.classes,
        args.output_dir / "validation_summary.txt",
    )

    print(f"Validation accuracy: {overall_accuracy:.4f}")
    print(f"Saved validation evaluation reports to {args.output_dir}")


if __name__ == "__main__":
    main()
