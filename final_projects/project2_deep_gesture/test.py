# Task type: classify all 6 hand postures.
# Usage:
# python test.py --image_dir ./test_images --model_path ./checkpoints/best_model.pth

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from models.resnet_model import build_resnet18


def get_test_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_idx_to_class(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Checkpoint must include class metadata: 'class_to_idx' or 'classes'."
        )

    if "class_to_idx" in checkpoint:
        idx_to_class = {
            int(index): class_name
            for class_name, index in checkpoint["class_to_idx"].items()
        }
    elif "classes" in checkpoint:
        idx_to_class = {
            index: class_name
            for index, class_name in enumerate(checkpoint["classes"])
        }
    else:
        raise ValueError(
            "Checkpoint must include class metadata: 'class_to_idx' or 'classes'."
        )

    expected_indices = set(range(len(idx_to_class)))
    actual_indices = set(idx_to_class)
    if actual_indices != expected_indices:
        raise ValueError(
            f"Class indices must be contiguous from 0 to {len(idx_to_class) - 1}. "
            f"Got: {sorted(actual_indices)}"
        )

    return idx_to_class


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    idx_to_class = get_idx_to_class(checkpoint)

    model = build_resnet18(num_classes=len(idx_to_class), pretrained=False)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model, idx_to_class


def predict_image(model, image_path, transform, idx_to_class, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)

    pred_idx = pred.item()
    pred_label = idx_to_class[pred_idx]

    return pred_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Path to folder containing PNG images."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to trained model."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/result.txt",
        help="Path to save prediction results."
    )

    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    model_path = args.model_path
    output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, idx_to_class = load_model(model_path, device)
    transform = get_test_transform()

    image_paths = sorted(list(image_dir.glob("*.png")))

    if len(image_paths) == 0:
        print("No PNG images found in:", image_dir)
        return

    results = []

    for image_path in image_paths:
        pred_label = predict_image(model, image_path, transform, idx_to_class, device)
        line = f"{image_path.name} {pred_label}"
        print(line)
        results.append(line)

    with open(output_path, "w", encoding="utf-8") as f:
        for line in results:
            f.write(line + "\n")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
