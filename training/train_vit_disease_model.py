"""
AgroSage - ViT Disease Classifier Training

Trains a Vision Transformer (ViT) classifier for plant disease detection and saves
it in transformers format under models/vit.
"""

import argparse
import os
import random
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoImageProcessor, ViTForImageClassification


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
VIT_DIR = os.path.join(MODEL_DIR, "vit")

SEED = 42


def resolve_dataset_dirs() -> Tuple[str, str, str, str]:
    """
    Resolve disease dataset paths across supported layouts.

    Supported examples:
    - data/plant_disease_dataset/train + valid
    - data/Plant Village Dataset/Train + Val
    """
    candidates = [
        {
            "dataset": "Plant Village Dataset",
            "train": "Train",
            "valid": "Val",
            "test": "Test",
        },
        {
            "dataset": "Plant Village Dataset",
            "train": "train",
            "valid": "val",
            "test": "test",
        },
        {
            "dataset": "plant_disease_dataset",
            "train": "train",
            "valid": "valid",
            "test": "test",
        },
    ]

    for item in candidates:
        dataset_dir = os.path.join(DATA_ROOT, item["dataset"])
        train_dir = os.path.join(dataset_dir, item["train"])
        valid_dir = os.path.join(dataset_dir, item["valid"])
        test_dir = os.path.join(dataset_dir, item["test"])
        if os.path.isdir(train_dir) and os.path.isdir(valid_dir):
            return dataset_dir, train_dir, valid_dir, test_dir

    checked_paths = "\n".join(
        [
            os.path.join(DATA_ROOT, "plant_disease_dataset", "train") + " + valid",
            os.path.join(DATA_ROOT, "Plant Village Dataset", "Train") + " + Val",
            os.path.join(DATA_ROOT, "Plant Village Dataset", "train") + " + val",
        ]
    )
    raise FileNotFoundError(
        "Could not find a supported disease dataset split. Checked:\n"
        f"{checked_paths}"
    )


def set_reproducibility(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms(processor: AutoImageProcessor):
    size = processor.size.get("height", 224)
    mean = processor.image_mean
    std = processor.image_std

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    valid_tf = transforms.Compose(
        [
            transforms.Resize(size + 32),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return train_tf, valid_tf


def build_dataloaders(train_dir: str, valid_dir: str, processor: AutoImageProcessor, batch_size: int):
    train_tf, valid_tf = build_transforms(processor)

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    valid_ds = datasets.ImageFolder(valid_dir, transform=valid_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, valid_loader, train_ds.class_to_idx


def evaluate(model, data_loader, device, max_batches: int | None = None):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = images.to(device)
            labels = labels.to(device)

            logits = model(pixel_values=images).logits
            loss = loss_fn(logits, labels)
            total_loss += float(loss.item()) * labels.size(0)

            preds = torch.argmax(logits, dim=1)
            correct += int((preds == labels).sum().item())
            total += labels.size(0)

    avg_loss = total_loss / max(1, total)
    accuracy = correct / max(1, total)
    return avg_loss, accuracy


def train(
    model,
    train_loader,
    valid_loader,
    device,
    epochs: int,
    lr: float,
    max_train_batches: int | None = None,
    max_valid_batches: int | None = None,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(pixel_values=images).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * labels.size(0)
            total += labels.size(0)

        train_loss = total_loss / max(1, total)
        val_loss, val_acc = evaluate(model, valid_loader, device, max_batches=max_valid_batches)
        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a ViT disease classifier.")
    parser.add_argument("--model", default="google/vit-base-patch16-224-in21k")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--fast", action="store_true", help="Enable fast training mode.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze ViT encoder weights.")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-valid-batches", type=int, default=None)
    args = parser.parse_args()

    set_reproducibility(SEED)
    _, train_dir, valid_dir, _ = resolve_dataset_dirs()

    processor = AutoImageProcessor.from_pretrained(args.model)
    train_loader, valid_loader, class_to_idx = build_dataloaders(
        train_dir,
        valid_dir,
        processor,
        batch_size=args.batch_size,
    )

    idx_to_class = {idx: label for label, idx in class_to_idx.items()}

    model = ViTForImageClassification.from_pretrained(
        args.model,
        num_labels=len(class_to_idx),
        id2label=idx_to_class,
        label2id=class_to_idx,
        ignore_mismatched_sizes=True,
    )

    if args.fast:
        args.freeze_backbone = True if not args.freeze_backbone else args.freeze_backbone
        if args.max_train_batches is None:
            args.max_train_batches = 200
        if args.max_valid_batches is None:
            args.max_valid_batches = 50

    if args.freeze_backbone:
        for param in model.vit.parameters():
            param.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Training ViT on {len(class_to_idx)} classes")
    train(
        model,
        train_loader,
        valid_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        max_train_batches=args.max_train_batches,
        max_valid_batches=args.max_valid_batches,
    )

    os.makedirs(VIT_DIR, exist_ok=True)
    model.save_pretrained(VIT_DIR)
    processor.save_pretrained(VIT_DIR)
    print(f"Saved ViT model to: {VIT_DIR}")


if __name__ == "__main__":
    main()
