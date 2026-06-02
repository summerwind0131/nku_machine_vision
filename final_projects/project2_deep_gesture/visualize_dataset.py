import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create dataset distribution and sample visualizations."
    )
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=Path("data/raw"),
        help="Raw dataset folder."
    )
    parser.add_argument(
        "--train_dir",
        type=Path,
        default=Path("data/train"),
        help="Training split folder."
    )
    parser.add_argument(
        "--val_dir",
        type=Path,
        default=Path("data/val"),
        help="Validation split folder."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/visualizations"),
        help="Folder for generated charts."
    )
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=5,
        help="Number of sample images to show for each class."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection."
    )
    return parser.parse_args()


def list_classes(dataset_dir):
    if not dataset_dir.exists():
        return []
    return sorted(path.name for path in dataset_dir.iterdir() if path.is_dir())


def list_images(class_dir):
    if not class_dir.exists():
        return []
    return sorted(
        path for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def count_dataset(dataset_dir):
    counts = {}
    for class_name in list_classes(dataset_dir):
        counts[class_name] = len(list_images(dataset_dir / class_name))
    return counts


def save_counts_csv(dataset_counts, output_path):
    all_classes = sorted({
        class_name
        for counts in dataset_counts.values()
        for class_name in counts
    })

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["class", *dataset_counts.keys()])
        for class_name in all_classes:
            writer.writerow([
                class_name,
                *[
                    dataset_counts[split_name].get(class_name, 0)
                    for split_name in dataset_counts
                ],
            ])


def plot_distribution(dataset_counts, output_path):
    all_classes = sorted({
        class_name
        for counts in dataset_counts.values()
        for class_name in counts
    })

    if not all_classes:
        raise ValueError("No class folders found for distribution plot.")

    split_names = list(dataset_counts)
    x_positions = list(range(len(all_classes)))
    bar_width = 0.8 / max(1, len(split_names))

    fig, ax = plt.subplots(figsize=(11, 6))
    for split_index, split_name in enumerate(split_names):
        values = [
            dataset_counts[split_name].get(class_name, 0)
            for class_name in all_classes
        ]
        offsets = [
            x + (split_index - (len(split_names) - 1) / 2) * bar_width
            for x in x_positions
        ]
        ax.bar(offsets, values, width=bar_width, label=split_name)

    ax.set_title("Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Image Count")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(all_classes)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_samples(dataset_dir, output_path, samples_per_class, seed):
    class_names = list_classes(dataset_dir)
    if not class_names:
        return False

    rng = random.Random(seed)
    rows = len(class_names)
    cols = max(1, samples_per_class)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 2.1))

    if rows == 1:
        axes = [axes]

    for row_index, class_name in enumerate(class_names):
        images = list_images(dataset_dir / class_name)
        selected = rng.sample(images, min(cols, len(images))) if images else []

        for col_index in range(cols):
            axis = axes[row_index][col_index] if cols > 1 else axes[row_index]
            axis.axis("off")

            if col_index >= len(selected):
                continue

            with Image.open(selected[col_index]) as image:
                axis.imshow(image.convert("RGB"))

            if col_index == 0:
                axis.set_ylabel(class_name, rotation=0, labelpad=30, va="center")

    fig.suptitle(f"Samples: {dataset_dir}", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dirs = {
        "raw": args.raw_dir,
        "train": args.train_dir,
        "val": args.val_dir,
    }
    dataset_counts = {
        split_name: count_dataset(dataset_dir)
        for split_name, dataset_dir in dataset_dirs.items()
        if dataset_dir.exists()
    }

    save_counts_csv(dataset_counts, args.output_dir / "class_counts.csv")
    plot_distribution(dataset_counts, args.output_dir / "class_distribution.png")

    for split_name, dataset_dir in dataset_dirs.items():
        if not dataset_dir.exists():
            continue
        created = plot_samples(
            dataset_dir,
            args.output_dir / f"{split_name}_samples.png",
            args.samples_per_class,
            args.seed,
        )
        if created:
            print(f"Saved {split_name} sample grid.")

    print(f"Saved visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()
