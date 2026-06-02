import os
import copy
import csv
import subprocess
import sys
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.resnet_model import build_resnet18

def get_transforms():
    train_transforms=transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224,scale=(0.75,1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.25,
            hue=0.05
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.08,0.08),
            scale=(0.9,1.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    val_tranforms=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    return train_transforms,val_tranforms

def train_one_epoch(model,dataloader,criterion,optimizer,device):
    model.train()

    running_loss=0.0
    running_corrects=0
    total=0

    for images, labels in tqdm(dataloader,desc="Train",leave=False):
        images=images.to(device)
        labels=labels.to(device)

        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        _,preds=torch.max(outputs,1)
        running_loss+=loss.item()*images.size(0)
        running_corrects+=torch.sum(preds==labels).item()
        total+=labels.size(0)
    
    epoch_loss=running_loss/total
    epoch_acc=running_corrects/total

    return epoch_loss,epoch_acc

def validate(model,dataloader,criterion,device):
    model.eval()

    running_loss=0.0
    running_corrects=0
    total=0
    with torch.no_grad():
        for images,labels in tqdm(dataloader,desc="Val",leave=False):
            images=images.to(device)
            labels=labels.to(device)

            outputs=model(images)
            loss=criterion(outputs,labels)

            _,preds=torch.max(outputs,1)

            running_loss+=loss.item()*images.size(0)
            running_corrects+=torch.sum(preds==labels).item()
            total+=labels.size(0)
        epoch_loss=running_loss/total
        epoch_acc=running_corrects/total
        return epoch_loss,epoch_acc

def get_learning_rate(optimizer):
    return optimizer.param_groups[0]["lr"]

def save_training_history(history, output_path):
    fieldnames = [
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "learning_rate",
        "is_best",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

def update_training_visualizations(history_path, output_dir, combined_output):
    subprocess.run(
        [
            sys.executable,
            "visualize_training_history.py",
            "--history",
            history_path,
            "--output_dir",
            output_dir,
            "--combined_output",
            combined_output,
        ],
        check=False,
    )
    
def main():
    train_dir = "data/train"
    val_dir = "data/val"
    checkpoint_dir = "checkpoints"
    output_dir = "outputs"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    history_path = os.path.join(output_dir, "training_history.csv")
    curves_path = os.path.join(output_dir, "training_curves.png")
    training_visualization_dir = os.path.join(output_dir, "training_visualizations")

    batch_size = 32
    num_epochs = 30
    learning_rate = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_transform, val_transform = get_transforms()

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    print("Class to idx:", train_dataset.class_to_idx)

    if val_dataset.class_to_idx != train_dataset.class_to_idx:
        raise ValueError(
            "Train and validation class_to_idx mappings do not match. "
            f"Train: {train_dataset.class_to_idx}; "
            f"Val: {val_dataset.class_to_idx}"
        )

    expected_classes = ["A", "B", "C", "Five", "Point", "V"]
    expected_class_to_idx = {
        class_name: index
        for index, class_name in enumerate(expected_classes)
    }

    if train_dataset.class_to_idx != expected_class_to_idx:
        print("Warning: class_to_idx is not the expected order!")
        print("Expected:", expected_class_to_idx)
        print("Actual:", train_dataset.class_to_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    model = build_resnet18(num_classes=len(train_dataset.classes), pretrained=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs
    )

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    history = []

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            device
        )

        current_lr = get_learning_rate(optimizer)
        scheduler.step()

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        is_best = False
        if val_acc > best_acc:
            is_best = True
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

            save_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save({
                "model_state_dict": best_model_wts,
                "class_to_idx": train_dataset.class_to_idx,
                "classes": train_dataset.classes,
                "best_acc": best_acc
            }, save_path)

            print(f"Best model saved. Val Acc: {best_acc:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "learning_rate": current_lr,
            "is_best": is_best,
        })
        save_training_history(history, history_path)
        update_training_visualizations(
            history_path,
            training_visualization_dir,
            curves_path,
        )

    print("\nTraining finished.")
    print(f"Best Val Acc: {best_acc:.4f}")
    print(f"Training history saved to {history_path}")
    print(f"Training curves saved to {curves_path}")


if __name__ == "__main__":
    main()



