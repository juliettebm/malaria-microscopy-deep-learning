"""Reusable data, modelling, and interpretation helpers for the notebooks."""
from __future__ import annotations

import hashlib
import io
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import Dataset

LABEL_TO_IDX = {"Uninfected": 0, "Parasitized": 1}


def audit_image(project_root: Path, relative_path: str) -> dict:
    """Return integrity, identity, and shape metadata for one image."""
    raw_bytes = (project_root / relative_path).read_bytes()
    corrupted = False
    width = height = np.nan
    mode = None
    try:
        image = Image.open(io.BytesIO(raw_bytes))
        image.verify()
        image = Image.open(io.BytesIO(raw_bytes))
        width, height = image.size
        mode = image.mode
    except Exception:
        corrupted = True
    return {
        "path": relative_path,
        "md5": hashlib.md5(raw_bytes).hexdigest(),
        "corrupted": corrupted,
        "width": width,
        "height": height,
        "mode": mode,
    }


class MalariaCellDataset(Dataset):
    """Single-cell microscopy images loaded from a manifest DataFrame."""

    def __init__(self, manifest_df, root_dir: Path, transform=None, label_to_idx=None):
        self.df = manifest_df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.label_to_idx = label_to_idx or LABEL_TO_IDX

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = Image.open(self.root_dir / row["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.label_to_idx[row["label"]]


class SimpleCNN(nn.Module):
    """Three convolution blocks followed by a compact binary classifier."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs):
        return self.classifier(self.features(inputs))


def build_resnet18_transfer(num_classes: int = 2, *, pretrained: bool = True):
    """Build a ResNet-18 feature extractor with a trainable classification head."""
    from torchvision import models

    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    if pretrained:
        for parameter in model.parameters():
            parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def keep_frozen_batchnorm_in_eval(model):
    """Prevent frozen BatchNorm layers from updating running statistics."""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            parameters = list(module.parameters(recurse=False))
            if parameters and not any(parameter.requires_grad for parameter in parameters):
                module.eval()


def evaluate_model(model, loader, device, positive_index: int = 1):
    """Compute loss, accuracy, and positive-class F1 on a data loader."""
    model.eval()
    total_loss, predictions, labels = 0.0, [], []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, targets).item() * inputs.size(0)
            predictions.extend(outputs.argmax(dim=1).cpu().numpy())
            labels.extend(targets.cpu().numpy())
    return (
        total_loss / len(loader.dataset),
        accuracy_score(labels, predictions),
        f1_score(labels, predictions, pos_label=positive_index),
    )


def train_model(model, train_loader, val_loader, epochs, lr, name, device, positive_index=1):
    """Train, select on validation F1, and restore the best model state."""
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_state = deepcopy(model.state_dict())
    best_epoch, best_val_f1 = 0, -np.inf
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        keep_frozen_batchnorm_in_eval(model)
        running_loss, predictions, labels = 0.0, [], []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            predictions.extend(outputs.argmax(dim=1).detach().cpu().numpy())
            labels.extend(targets.cpu().numpy())
        val_loss, val_acc, val_f1 = evaluate_model(model, val_loader, device, positive_index)
        history["train_loss"].append(running_loss / len(train_loader.dataset))
        history["train_acc"].append(accuracy_score(labels, predictions))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        if val_f1 > best_val_f1:
            best_epoch, best_val_f1 = epoch, val_f1
            best_state = deepcopy(model.state_dict())
        print(f"[{name}] epoch {epoch}/{epochs} - val_loss {val_loss:.4f} val_acc {val_acc:.4f} val_f1 {val_f1:.4f}")
    elapsed = time.time() - started
    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["best_val_f1"] = best_val_f1
    return history, elapsed


def predict_probabilities(model, loader, device, positive_index: int = 1):
    """Return labels and positive-class probabilities in loader order."""
    model.eval()
    labels, probabilities = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            logits = model(inputs.to(device))
            probabilities.extend(torch.softmax(logits, dim=1)[:, positive_index].cpu().numpy())
            labels.extend(targets.numpy())
    return np.asarray(labels), np.asarray(probabilities)


def select_threshold_for_sensitivity(y_true, y_prob, target_sensitivity: float = 0.98):
    """Maximise specificity while meeting a predeclared validation sensitivity."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    candidates = np.unique(np.r_[0.0, y_prob, 1.0])
    feasible = []
    for threshold in candidates:
        prediction = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if tp + fn else np.nan
        specificity = tn / (tn + fp) if tn + fp else np.nan
        if sensitivity >= target_sensitivity:
            feasible.append((specificity, threshold, sensitivity))
    if not feasible:
        raise ValueError("No threshold satisfies the requested sensitivity")
    specificity, threshold, sensitivity = max(feasible)
    return {"threshold": float(threshold), "sensitivity": float(sensitivity), "specificity": float(specificity)}


def ppv_npv_at_prevalence(sensitivity, specificity, prevalence):
    """Translate sensitivity and specificity to predictive values at a prevalence."""
    ppv = sensitivity * prevalence / (sensitivity * prevalence + (1 - specificity) * (1 - prevalence))
    npv = specificity * (1 - prevalence) / (specificity * (1 - prevalence) + (1 - sensitivity) * prevalence)
    return ppv, npv


def denormalize(tensor_image, mean, std):
    """Reverse channel-wise normalization for display."""
    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)
    return (tensor_image * std_tensor + mean_tensor).clamp(0, 1)


def compute_saliency(model, image_tensor, target_class, device):
    """Compute a maximum-channel input-gradient saliency map."""
    model.eval()
    image_tensor = image_tensor.clone().unsqueeze(0).to(device).requires_grad_(True)
    score = model(image_tensor)[0, target_class]
    model.zero_grad()
    score.backward()
    return image_tensor.grad.data.abs().squeeze(0).max(dim=0).values.cpu().numpy()
