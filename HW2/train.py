import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms


DATASET_DIR = "dataset"
MODEL_PATH = "face_model.pth"
BATCH_SIZE = 16
NUM_EPOCHS = 5
LR = 1e-4
VAL_SPLIT = 0.2
RANDOM_SEED = 42
NUM_WORKERS = 2  # adjust if needed


def get_transforms(img_size: int = 224):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return train_tf, val_tf


def make_dataloaders() -> Tuple[DataLoader, DataLoader, Dict[int, str]]:
    train_tf, val_tf = get_transforms()

    # First create dataset with train transforms (we'll clone for val)
    full_dataset = datasets.ImageFolder(DATASET_DIR, transform=train_tf)
    if len(full_dataset.classes) == 0:
        raise RuntimeError(f"No classes found in {DATASET_DIR}")

    # Split into train / val
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    gen = torch.Generator().manual_seed(RANDOM_SEED)
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=gen)

    # Override transform on val subset
    val_ds.dataset = datasets.ImageFolder(DATASET_DIR, transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    idx_to_class = {v: k for k, v in full_dataset.class_to_idx.items()}
    return train_loader, val_loader, idx_to_class


class EmbeddingNet(nn.Module):
    """ResNet-50 backbone with an embedding layer and classifier head."""

    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()  # remove original classifier
        self.backbone = backbone
        self.embedding = nn.Linear(in_features, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        feats = self.backbone(x)              # [B, in_features]
        emb = self.embedding(feats)           # [B, embed_dim]
        emb = nn.functional.normalize(emb, p=2, dim=1)  # L2-normalized embeddings
        logits = self.classifier(emb)         # [B, num_classes]
        return emb, logits


def build_model(num_classes: int, embed_dim: int = 128) -> nn.Module:
    return EmbeddingNet(embed_dim, num_classes)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        emb, logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(logits, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            emb, logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, idx_to_class = make_dataloaders()
    num_classes = len(idx_to_class)
    print(f"Found {num_classes} classes.")

    embed_dim = 128
    model = build_model(num_classes, embed_dim=embed_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = eval_one_epoch(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "model_state_dict": model.state_dict(),
                "idx_to_class": idx_to_class,
                "embed_dim": embed_dim,
            }

    if best_state is not None:
        torch.save(best_state, MODEL_PATH)
        print(f"Saved best model with val acc {best_val_acc:.3f} to {MODEL_PATH}")
    else:
        print("No model was saved (check your data).")


if __name__ == "__main__":
    main()