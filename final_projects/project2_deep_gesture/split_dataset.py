# 将data自动划分为train与raw
import os
import random
import shutil
import pathlib as Path

def split_dataset(
        raw_dir="data/raw",
        train_dir="data/train",
        val_dir="data/val",
        val_ratio=0.2,
        seed=42
):
    random.seed(42)
    classes = ["A", "B", "C", "Five", "Point", "V"]

    raw_dir = Path(raw_dir)
    train_dir = Path(train_dir)
    val_dir = Path(val_dir)

    for cls in classes:
        src_cls_dir = raw_dir / cls
        train_cls_dir = train_dir / cls
        val_cls_dir = val_dir / cls

        train_cls_dir.mkdir(parents=True, exist_ok=True)
        val_cls_dir.mkdir(parents=True, exist_ok=True)

        images = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
            images.extend(list(src_cls_dir.glob(ext)))

        random.shuffle(images)

        val_count = int(len(images) * val_ratio)
        val_images = images[:val_count]
        train_images = images[val_count:]

        for img_path in train_images:
            shutil.copy(img_path, train_cls_dir / img_path.name)

        for img_path in val_images:
            shutil.copy(img_path, val_cls_dir / img_path.name)

        print(f"{cls}: train={len(train_images)}, val={len(val_images)}")


if __name__ == "__main__":
    split_dataset()