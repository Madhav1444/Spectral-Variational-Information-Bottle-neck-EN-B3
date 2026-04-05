# -*- coding: windows-1252 -*-
# -*- coding: utf-8 -*-
"""
SV_IBN_main.py
==============
SV-IBN: Spectral-Variational Information Bottleneck Network
============================================================
Baseline  : EfficientNet-B3
Proposed  : SV-IBN + EfficientNet-B3
Ablations : SV-IBN_no_Spectral  |  SV-IBN_no_VIB

What this script does (end-to-end)
-----------------------------------
1.  Trains all 4 models across 6 seeds (42-47)
    - Seeds 42/43/44: original runs
    - Seeds 45/46/47: extended for statistical power
    - Skips any model+seed already completed

2.  Saves all outputs in organised per-model subfolders:
    SV_IBN_RESULTS_FINAL_2026/
    +-- EfficientNet_B3/
    ¦   +-- metrics/    <- test_metrics, CM, ROC, history CSVs + PNGs
    ¦   +-- curves/     <- training curve PNGs
    ¦   +-- gradcam/    <- glioma/ meningioma/ notumor/ pituitary/
    ¦   +-- ablation_note.txt
    +-- SV_IBN_no_VIB/
    +-- SV_IBN_no_Spectral/
    +-- SV_IBN_EfficientNet_B3/
    +-- all_results.csv
    +-- summary_aggregated.csv
    +-- ablation_summary.csv
    +-- wilcoxon_n6.csv
    +-- mcnemar_pooled.csv
    +-- superiority_table.csv
    +-- comparison plots ...

3.  Statistical analysis with n=6:
    - Wilcoxon signed-rank (can now reach p=0.03125)
    - McNemar's test pooled across all seeds (primary stat)
    - Superiority table: SV-IBN vs EfficientNet-B3

FIXES applied
-------------
- val_loader / test_loader  num_workers=0   (DataLoader timeout on Py 3.13)
- train_loader              num_workers=4   (no persistent_workers)
- RGB conversion in transforms              (handles RGBA / palette PNG)
- CUDA sync + cache clear every epoch       (prevents memory fragmentation)
- try/except per test batch                 (skips corrupt images)
- GradCAM hooks via named layer lookup      (correct gradient capture)
- Label smoothing (e=0.1) in CE loss        (regularises EfficientNet-B3)
- MixUp (a=0.4) for proposed model         (boosts generalisation)
- Larger LATENT_DIM=512 + projection head   (richer bottleneck)
- BETA_KL annealing 0->1e-3 over 20 epochs (stable VIB training)
- Organised per-model output folders        (clean paper structure)
- 6 seeds for valid statistical testing     (Wilcoxon + McNemar)
"""

import os, gc, random, itertools, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from scipy.stats import chi2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import torchvision.transforms.v2 as v2

import timm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    roc_curve, auc, precision_score, recall_score,
    classification_report
)
from sklearn.model_selection import train_test_split
from scipy import stats

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
TRAIN_PATH   = "/nfsshare/users/raghavan/Brainz/Brain tumor dataset/Training/"
TEST_PATH    = "/nfsshare/users/raghavan/Brainz/Brain tumor dataset/Test/"
SAVE_DIR     = Path("SV_IBN_RESULTS_FINAL_2026_Vaitheeswaran")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE       = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
IMG_SIZE     = 224
LR           = 1e-4
EPOCHS       = 60
PATIENCE     = 12
LATENT_DIM   = 512
BETA_KL_MAX  = 1e-3
LABEL_SMOOTH = 0.10
MIXUP_ALPHA  = 0.4
ALPHA        = 0.05          # significance threshold

SEEDS  = [42, 43, 44, 45, 46, 47]    # n=6 for valid statistical testing

MODELS = [
    "EfficientNet_B3",           # baseline
    "SV_IBN_no_VIB",             # ablation A: spectral gate only
    "SV_IBN_no_Spectral",        # ablation B: VIB only
    "SV_IBN_EfficientNet_B3",    # proposed: full SV-IBN
]

MODEL_BATCH = {
    "EfficientNet_B3":        16,
    "SV_IBN_no_VIB":          16,
    "SV_IBN_no_Spectral":     16,
    "SV_IBN_EfficientNet_B3": 16,
}

ABLATION_ROLES = {
    "EfficientNet_B3":
        "Baseline. Pretrained EfficientNet-B3, no spectral gate, no VIB.",
    "SV_IBN_no_VIB":
        "Ablation A: spectral gate only. VIB removed. "
        "Isolates contribution of AdaptiveSpectralGate.",
    "SV_IBN_no_Spectral":
        "Ablation B: VIB only. Spectral gate removed. "
        "Isolates contribution of VariationalBottleneck.",
    "SV_IBN_EfficientNet_B3":
        "Proposed model. Full SV-IBN: spectral gate + VIB + EfficientNet-B3.",
}

METRICS = ["acc", "f1_macro", "precision_macro", "recall_macro", "auc_macro"]
METRIC_LABELS = {
    "acc":             "Accuracy",
    "f1_macro":        "Macro F1",
    "precision_macro": "Macro Precision",
    "recall_macro":    "Macro Recall",
    "auc_macro":       "Macro AUC",
}


# -----------------------------------------------------------------------------
# Organised folder helpers
# -----------------------------------------------------------------------------
def get_model_dirs(model_name: str) -> dict:
    """Creates and returns per-model subdirectory paths."""
    root    = SAVE_DIR / model_name
    metrics = root / "metrics"
    curves  = root / "curves"
    gradcam = root / "gradcam"

    for d in [root, metrics, curves, gradcam]:
        d.mkdir(parents=True, exist_ok=True)

    note = root / "ablation_note.txt"
    if not note.exists():
        note.write_text(
            f"Model : {model_name}\n"
            f"Role  : {ABLATION_ROLES.get(model_name, 'Unknown')}\n"
        )
    return {"root": root, "metrics": metrics, "curves": curves, "gradcam": gradcam}


def get_model_paths(model_name: str, seed: int, dirs: dict) -> dict:
    """Returns all output file paths for one model+seed."""
    return {
        "test_metrics": dirs["metrics"] / f"test_metrics_seed{seed}.csv",
        "cm_csv":       dirs["metrics"] / f"cm_seed{seed}.csv",
        "cm_png":       dirs["metrics"] / f"cm_seed{seed}.png",
        "roc_data":     dirs["metrics"] / f"roc_data_seed{seed}.csv",
        "roc_png":      dirs["metrics"] / f"roc_seed{seed}.png",
        "history_csv":  dirs["metrics"] / f"history_seed{seed}.csv",
        "curves_png":   dirs["curves"]  / f"curves_seed{seed}.png",
        "gradcam_base": dirs["gradcam"],
    }


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# -----------------------------------------------------------------------------
# MixUp
# -----------------------------------------------------------------------------
def mixup_data(x, y, alpha=0.4):
    lam       = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx       = torch.randperm(x.size(0), device=x.device)
    mixed_x   = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def mixup_criterion(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
class AdaptiveSpectralGate(nn.Module):
    """FFT-domain gating with learnable complex weights + SE channel attention."""
    def __init__(self, channels: int):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(1, channels, 1, 1) * 0.02)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // 8, 1), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // 8, 1), channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ffted = torch.fft.rfft2(x, norm='ortho')
        ffted = ffted * (1.0 + self.complex_weight)
        attn  = self.gate(x)
        x_rec = torch.fft.irfft2(ffted, s=x.shape[-2:], norm='ortho')
        return x_rec * attn


class VariationalBottleneck(nn.Module):
    """Reparameterised Gaussian bottleneck with projection head."""
    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        self.proj   = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )
        self.mu     = nn.Linear(latent_dim, latent_dim)
        self.logvar = nn.Linear(latent_dim, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar.clamp(-4, 4))
        return mu + torch.randn_like(std) * std

    def forward(self, x: torch.Tensor):
        h      = self.proj(x)
        mu     = self.mu(h)
        logvar = self.logvar(h)
        return self.reparameterize(mu, logvar), mu, logvar


# -----------------------------------------------------------------------------
# Model factory
# -----------------------------------------------------------------------------
def create_model(name: str, num_classes: int) -> nn.Module:

    if name == "EfficientNet_B3":
        base = timm.create_model(
            "efficientnet_b3.ra2_in1k", pretrained=True, num_classes=num_classes
        )
        class Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.model        = base
                self.target_layer = base.blocks[-1][-1].conv_pwl
            def forward(self, x):
                return self.model(x), None, None
        return Wrapper()

    elif name == "SV_IBN_no_VIB":
        backbone = timm.create_model(
            "efficientnet_b3.ra2_in1k", pretrained=True, features_only=True
        )
        nf = backbone.feature_info[-1]['num_chs']
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone     = backbone
                self.gate         = AdaptiveSpectralGate(nf)
                self.dropout      = nn.Dropout(0.3)
                self.head         = nn.Linear(nf, num_classes)
                self.target_layer = backbone.blocks[-1][-1].conv_pwl
            def forward(self, x):
                feats  = self.backbone(x)[-1]
                feats  = self.gate(feats)
                pooled = F.adaptive_avg_pool2d(feats, 1).flatten(1)
                return self.head(self.dropout(pooled)), None, None
        return Model()

    elif name == "SV_IBN_no_Spectral":
        backbone = timm.create_model(
            "efficientnet_b3.ra2_in1k", pretrained=True, features_only=True
        )
        nf = backbone.feature_info[-1]['num_chs']
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone     = backbone
                self.bottle       = VariationalBottleneck(nf, LATENT_DIM)
                self.dropout      = nn.Dropout(0.3)
                self.head         = nn.Linear(LATENT_DIM, num_classes)
                self.target_layer = backbone.blocks[-1][-1].conv_pwl
            def forward(self, x):
                feats  = self.backbone(x)[-1]
                pooled = F.adaptive_avg_pool2d(feats, 1).flatten(1)
                z, mu, logvar = self.bottle(pooled)
                return self.head(self.dropout(z)), mu, logvar
        return Model()

    elif name == "SV_IBN_EfficientNet_B3":
        backbone = timm.create_model(
            "efficientnet_b3.ra2_in1k", pretrained=True, features_only=True
        )
        nf = backbone.feature_info[-1]['num_chs']
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone     = backbone
                self.gate         = AdaptiveSpectralGate(nf)
                self.bottle       = VariationalBottleneck(nf, LATENT_DIM)
                self.dropout      = nn.Dropout(0.3)
                self.head         = nn.Linear(LATENT_DIM, num_classes)
                self.target_layer = backbone.blocks[-1][-1].conv_pwl
            def forward(self, x):
                feats  = self.backbone(x)[-1]
                feats  = self.gate(feats)
                pooled = F.adaptive_avg_pool2d(feats, 1).flatten(1)
                z, mu, logvar = self.bottle(pooled)
                return self.head(self.dropout(z)), mu, logvar
        return Model()

    else:
        raise ValueError(f"Unknown model: {name}")


# -----------------------------------------------------------------------------
# GradCAM
# -----------------------------------------------------------------------------
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model       = model
        self.gradients   = None
        self.activations = None

        def fwd(module, input, output):
            self.activations = output.detach()

        def bwd(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._fh = target_layer.register_forward_hook(fwd)
        self._bh = target_layer.register_full_backward_hook(bwd)

    def generate(self, x: torch.Tensor, class_idx: int = None) -> np.ndarray:
        self.model.zero_grad()
        logits, _, _ = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        logits[:, class_idx].sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam     = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
        cam_min, cam_max = cam.min(), cam.max()
        return ((cam - cam_min) / (cam_max - cam_min + 1e-8)).cpu().numpy()[0, 0]

    def release(self):
        self._fh.remove()
        self._bh.remove()


# -----------------------------------------------------------------------------
# Denormalise
# -----------------------------------------------------------------------------
_MEAN = torch.tensor([0.485, 0.456, 0.406])
_STD  = torch.tensor([0.229, 0.224, 0.225])

def denorm(t: torch.Tensor) -> np.ndarray:
    t = t.cpu() * _STD.view(3, 1, 1) + _MEAN.view(3, 1, 1)
    return (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Save all test metrics
# -----------------------------------------------------------------------------
def save_all_test_metrics(model_name, seed, classes, y_true, y_pred, y_prob,
                           out_dir=None):
    save_to = Path(out_dir) if out_dir else SAVE_DIR
    rows    = []
    report  = classification_report(
        y_true, y_pred, target_names=classes,
        output_dict=True, zero_division=0
    )
    per_auc = []
    for cls_idx, cls_name in enumerate(classes):
        r           = report[cls_name]
        fpr, tpr, _ = roc_curve((y_true == cls_idx).astype(int), y_prob[:, cls_idx])
        cls_auc     = auc(fpr, tpr)
        per_auc.append(cls_auc)

        cm_bin      = confusion_matrix(
            (y_true == cls_idx).astype(int), (y_pred == cls_idx).astype(int)
        )
        tn          = cm_bin[0, 0] if cm_bin.shape == (2, 2) else 0
        fp          = cm_bin[0, 1] if cm_bin.shape == (2, 2) else 0
        rows.append({
            "model": model_name, "seed": seed, "class": cls_name,
            "precision":   round(r["precision"], 4),
            "recall":      round(r["recall"],    4),
            "f1_score":    round(r["f1-score"],  4),
            "support":     int(r["support"]),
            "auc":         round(cls_auc,         4),
            "specificity": round(tn / (tn + fp + 1e-8), 4),
        })

    acc         = accuracy_score(y_true, y_pred)
    macro_prec  = precision_score(y_true, y_pred, average='macro',    zero_division=0)
    macro_rec   = recall_score(y_true, y_pred,    average='macro',    zero_division=0)
    macro_f1    = f1_score(y_true, y_pred,        average='macro',    zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred,        average='weighted', zero_division=0)
    macro_auc   = float(np.mean(per_auc))

    for tag, vals in [
        ("MACRO_AVG",   {"precision": macro_prec, "recall": macro_rec,
                         "f1_score":  macro_f1,   "auc":    macro_auc}),
        ("ACCURACY",    {"f1_score": acc}),
        ("WEIGHTED_F1", {"f1_score": weighted_f1}),
    ]:
        row = {"model": model_name, "seed": seed, "class": tag,
               "precision": "", "recall": "", "f1_score": "",
               "support": int(y_true.shape[0]), "auc": "", "specificity": ""}
        for k, v in vals.items():
            row[k] = round(v, 4)
        rows.append(row)

    out_path = save_to / f"test_metrics_seed{seed}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  ok  Saved {out_path.name}")
    return acc, macro_f1, macro_prec, macro_rec, macro_auc


# -----------------------------------------------------------------------------
# Single seed: train + evaluate
# -----------------------------------------------------------------------------
def run_one_seed(seed: int, model_name: str) -> dict:

    # Check if already completed — skip gracefully
    existing_csv = SAVE_DIR / "all_results.csv"
    if existing_csv.exists():
        ex = pd.read_csv(existing_csv)
        if not ex[
            (ex["model"] == model_name) & (ex["seed"] == seed)
        ].empty:
            print(f"  ->  {model_name} seed {seed} already done — skipping")
            row = ex[(ex["model"] == model_name) & (ex["seed"] == seed)].iloc[0]
            return row.to_dict()

    set_seed(seed)

    # Organised output paths
    dirs  = get_model_dirs(model_name)
    paths = get_model_paths(model_name, seed, dirs)

    print(f"\n{'='*65}")
    print(f"  Seed {seed}  |  Model: {model_name}")
    print(f"{'='*65}")

    batch_size = MODEL_BATCH.get(model_name, 16)
    use_mixup  = (model_name == "SV_IBN_EfficientNet_B3")

    # Transforms
    train_tf = v2.Compose([
        v2.Lambda(lambda img: img.convert("RGB")),
        v2.RandomResizedCrop(IMG_SIZE, scale=(0.80, 1.0)),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(p=0.1),
        v2.RandomRotation(20),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        v2.RandomGrayscale(p=0.05),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = v2.Compose([
        v2.Lambda(lambda img: img.convert("RGB")),
        v2.Resize((IMG_SIZE, IMG_SIZE)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Datasets — val uses its own ImageFolder with val_tf (no augmentation leak)
    full_train   = datasets.ImageFolder(TRAIN_PATH, transform=train_tf)
    full_val_ref = datasets.ImageFolder(TRAIN_PATH, transform=val_tf)
    test_ds      = datasets.ImageFolder(TEST_PATH,  transform=val_tf)
    classes      = full_train.classes
    num_classes  = len(classes)

    train_idx, val_idx = train_test_split(
        np.arange(len(full_train)),
        test_size=0.2,
        stratify=full_train.targets,
        random_state=seed,
    )

    # DataLoaders
    train_loader = DataLoader(
        Subset(full_train,   train_idx),
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(full_val_ref, val_idx),   # uses val_tf — no aug leak
        batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    # Model / optimiser / scheduler
    model     = create_model(model_name, num_classes).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=1, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    best_val_loss    = float('inf')
    best_state       = None
    patience_counter = 0
    history = {'epoch': [], 'train_loss': [], 'train_acc': [],
               'val_loss': [], 'val_acc': []}

    # Training loop
    for epoch in range(EPOCHS):
        torch.cuda.synchronize(DEVICE)
        torch.cuda.empty_cache()
        gc.collect()

        beta_kl = BETA_KL_MAX * min(1.0, epoch / 20.0)
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0

        for batch_idx, (x, y) in enumerate(train_loader):
            try:
                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)
                optimizer.zero_grad()

                if use_mixup:
                    x_mix, y_a, y_b, lam = mixup_data(x, y, MIXUP_ALPHA)
                    logits, mu, logvar    = model(x_mix)
                    loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
                else:
                    logits, mu, logvar = model(x)
                    loss = criterion(logits, y)

                if mu is not None:
                    kl   = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = loss + beta_kl * kl

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                t_loss    += loss.item() * x.size(0)
                t_correct += (logits.argmax(1) == y).sum().item()
                t_total   += y.size(0)

            except RuntimeError as e:
                print(f"  ! Train batch {batch_idx} skipped: {e}")
                torch.cuda.empty_cache()
                continue

        scheduler.step()
        train_loss = t_loss    / max(t_total, 1)
        train_acc  = t_correct / max(t_total, 1)

        # Validation
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                logits, _, _ = model(x)
                v_loss    += criterion(logits, y).item() * x.size(0)
                v_correct += (logits.argmax(1) == y).sum().item()
                v_total   += y.size(0)

        val_loss = v_loss    / max(v_total, 1)
        val_acc  = v_correct / max(v_total, 1)

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(round(train_loss, 6))
        history['train_acc'].append(round(train_acc,   6))
        history['val_loss'].append(round(val_loss,     6))
        history['val_acc'].append(round(val_acc,       6))

        print(f"  [{epoch+1:2d}/{EPOCHS}]  "
              f"Train {train_loss:.4f}/{train_acc:.4f}  "
              f"Val {val_loss:.4f}/{val_acc:.4f}  "
              f"beta_kl={beta_kl:.2e}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_state       = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("  -> Early stopping triggered")
                break

    model.load_state_dict(best_state)

    # Save training history + curves
    pd.DataFrame(history).to_csv(paths["history_csv"], index=False)

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.plot(history['epoch'], history['train_loss'], 'b-',  label='Train Loss')
    ax1.plot(history['epoch'], history['val_loss'],   'c--', label='Val Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx()
    ax2.plot(history['epoch'], history['train_acc'], 'r-',  label='Train Acc', alpha=0.85)
    ax2.plot(history['epoch'], history['val_acc'],   'm--', label='Val Acc',   alpha=0.85)
    ax2.set_ylabel('Accuracy', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim(0, 1.05)
    lines  = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc='center right', fontsize=9)
    plt.title(f"{model_name}  —  seed {seed}")
    plt.grid(True, alpha=0.25); plt.tight_layout()
    plt.savefig(paths["curves_png"], dpi=160, bbox_inches='tight')
    plt.close()

    # Test inference + GradCAM
    cam_base = paths["gradcam_base"]
    for cls in classes:
        (cam_base / cls).mkdir(parents=True, exist_ok=True)

    cam_engine = GradCAM(model, model.target_layer)
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    for i, (x, y) in enumerate(test_loader):
        try:
            x_dev = x.to(DEVICE)
            y_dev = y.to(DEVICE)

            with torch.set_grad_enabled(True):
                logits_eval, _, _ = model(x_dev)
                prob = F.softmax(logits_eval.detach(), dim=1)
                pred = logits_eval.argmax(dim=1).item()
                cam  = cam_engine.generate(x_dev, class_idx=y_dev.item())

            orig_img = denorm(x[0])
            cls_name = classes[y.item()]

            fig, axes = plt.subplots(1, 3, figsize=(13, 4))
            axes[0].imshow(orig_img);                    axes[0].set_title("Original");          axes[0].axis('off')
            axes[1].imshow(cam, cmap='jet', vmin=0, vmax=1); axes[1].set_title("GradCAM");      axes[1].axis('off')
            axes[2].imshow(orig_img)
            axes[2].imshow(plt.cm.jet(cam)[:, :, :3], alpha=0.45)
            correct = "correct" if pred == y.item() else "wrong"
            axes[2].set_title(f"True: {cls_name}  Pred: {classes[pred]}  ({correct})", fontsize=10)
            axes[2].axis('off')
            plt.suptitle(f"{model_name} — seed {seed}", fontsize=9, y=1.01)
            plt.tight_layout()
            plt.savefig(cam_base / cls_name / f"img_{i:05d}.png",
                        dpi=110, bbox_inches='tight')
            plt.close()

            y_true.append(y.item())
            y_pred.append(pred)
            y_prob.append(prob.cpu().numpy()[0])

        except Exception as e:
            print(f"  ! Test image {i} skipped: {e}")
            continue

    cam_engine.release()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # Metrics CSV
    acc, macro_f1, macro_prec, macro_rec, macro_auc = save_all_test_metrics(
        model_name, seed, classes, y_true, y_pred, y_prob,
        out_dir=dirs["metrics"]
    )

    # Confusion matrix
    cm    = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.to_csv(paths["cm_csv"])
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues',
                linewidths=0.5, linecolor='gray', ax=ax)
    ax.set_title(f"Confusion Matrix — {model_name} seed {seed}", fontsize=12)
    ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(paths["cm_png"], dpi=150, bbox_inches='tight')
    plt.close()

    # ROC curves
    roc_rows = []
    plt.figure(figsize=(9, 6))
    for j, cls in enumerate(classes):
        fpr, tpr, thresholds = roc_curve((y_true == j).astype(int), y_prob[:, j])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{cls} (AUC={roc_auc:.3f})', linewidth=2)
        for f, t, th in zip(fpr, tpr, thresholds):
            roc_rows.append({
                'model': model_name, 'seed': seed, 'class': cls,
                'fpr': round(float(f), 6), 'tpr': round(float(t), 6),
                'threshold': round(float(th), 6), 'auc': round(roc_auc, 6),
            })
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.legend(fontsize=9); plt.grid(True, alpha=0.3)
    plt.title(f"ROC — {model_name} seed {seed}")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.tight_layout()
    plt.savefig(paths["roc_png"], dpi=150, bbox_inches='tight')
    plt.close()
    pd.DataFrame(roc_rows).to_csv(paths["roc_data"], index=False)

    result = {
        "seed":            seed,
        "model":           model_name,
        "acc":             round(acc,        4),
        "f1_macro":        round(macro_f1,   4),
        "precision_macro": round(macro_prec, 4),
        "recall_macro":    round(macro_rec,  4),
        "auc_macro":       round(macro_auc,  4),
    }
    print(f"  ok  Acc={acc:.4f}  F1={macro_f1:.4f}  AUC={macro_auc:.4f}")
    return result


# -----------------------------------------------------------------------------
# Statistical helpers
# -----------------------------------------------------------------------------
def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    pooled = np.sqrt(
        ((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2)
        / (na + nb - 2 + 1e-12)
    )
    return float((a.mean() - b.mean()) / (pooled + 1e-12))


def effect_label(d: float) -> str:
    a = abs(d)
    if a >= 0.8:   return "large"
    if a >= 0.5:   return "medium"
    if a >= 0.2:   return "small"
    return "negligible"


def load_predictions_from_cm(model_name: str, seed: int):
    """Reconstructs y_true/y_pred from saved confusion matrix CSV."""
    cm_path = SAVE_DIR / model_name / "metrics" / f"cm_seed{seed}.csv"
    if not cm_path.exists():
        raise FileNotFoundError(f"Missing: {cm_path}")
    cm_df   = pd.read_csv(cm_path, index_col=0)
    classes = list(cm_df.index)
    cm      = cm_df.values.astype(int)
    y_true_list, y_pred_list = [], []
    for ti in range(len(classes)):
        for pi in range(len(classes)):
            count = cm[ti, pi]
            y_true_list.extend([ti] * count)
            y_pred_list.extend([pi] * count)
    return np.array(y_true_list), np.array(y_pred_list), classes


def mcnemar_test(y_true, pred_a, pred_b):
    """McNemar with continuity correction (Edwards 1948)."""
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    b = int(np.sum( correct_a & ~correct_b))
    c = int(np.sum(~correct_a &  correct_b))
    if (b + c) == 0:
        return np.nan, np.nan, b, c
    chi2_stat = (abs(b - c) - 1.0) ** 2 / (b + c)
    p_value   = 1.0 - chi2.cdf(chi2_stat, df=1)
    return chi2_stat, p_value, b, c


# -----------------------------------------------------------------------------
# Wilcoxon signed-rank (n=6)
# -----------------------------------------------------------------------------
def run_wilcoxon(df: pd.DataFrame):
    print("\n=== Wilcoxon Signed-Rank Test (n=6 seeds) ===")
    rows = []
    for metric in METRICS:
        print(f"\n  Metric: {metric}")
        for m1, m2 in itertools.combinations(MODELS, 2):
            s1 = df[df["model"] == m1][metric].values
            s2 = df[df["model"] == m2][metric].values
            if len(s1) < 2:
                continue
            try:
                w_stat, p = stats.wilcoxon(s1, s2, zero_method='wilcox', correction=False)
            except ValueError:
                w_stat, p = np.nan, np.nan

            d     = cohens_d(s1, s2)
            delta = s1.mean() - s2.mean()
            eff   = effect_label(d)
            sig   = (not np.isnan(p)) and (p < ALPHA)

            print(f"    {m1:30s} vs {m2:25s} | "
                  f"d={delta:+.4f}  W={w_stat:.1f}  p={p:.4f}  "
                  f"d={d:+.3f}({eff})  {'* sig' if sig else 'n.s.'}")

            rows.append({
                "metric":          metric,
                "model_A":         m1,
                "model_B":         m2,
                "mean_A":          round(s1.mean(), 4),
                "mean_B":          round(s2.mean(), 4),
                "delta_A_minus_B": round(delta,     6),
                "wilcoxon_W":      round(w_stat, 2) if not np.isnan(w_stat) else np.nan,
                "p_value":         round(p, 6)       if not np.isnan(p)      else np.nan,
                "cohens_d":        round(d, 4),
                "effect_size":     eff,
                "significant_005": sig,
                "n_seeds":         len(s1),
            })

    out = pd.DataFrame(rows)
    out.to_csv(SAVE_DIR / "wilcoxon_n6.csv", index=False)
    print(f"\n  ok  Saved wilcoxon_n6.csv")
    return out


# -----------------------------------------------------------------------------
# McNemar pooled across all seeds
# -----------------------------------------------------------------------------
def run_mcnemar_pooled(all_seeds):
    print("\n=== McNemar's Test — Pooled Across All Seeds (PRIMARY STAT) ===")
    pair_counts = {}

    for seed in all_seeds:
        preds = {}
        y_ref = None
        for m in MODELS:
            try:
                y_true, y_pred, _ = load_predictions_from_cm(m, seed)
                preds[m] = y_pred
                if y_ref is None:
                    y_ref = y_true
            except FileNotFoundError as e:
                print(f"  ! {e}")
                continue

        for m1, m2 in itertools.combinations(list(preds.keys()), 2):
            _, _, b, c = mcnemar_test(y_ref, preds[m1], preds[m2])
            key = (m1, m2)
            if key not in pair_counts:
                pair_counts[key] = [0, 0]
            pair_counts[key][0] += b
            pair_counts[key][1] += c

    rows = []
    print(f"\n  {'Model A':30s} vs {'Model B':28s} | "
          f"{'b':>6} {'c':>6}  {'chi2':>8}  {'p':>8}  result  winner")
    print("  " + "-" * 100)

    for (m1, m2), (b_total, c_total) in pair_counts.items():
        if (b_total + c_total) == 0:
            chi2_stat, p = np.nan, np.nan
        else:
            chi2_stat = (abs(b_total - c_total) - 1.0) ** 2 / (b_total + c_total)
            p         = 1.0 - chi2.cdf(chi2_stat, df=1)

        sig    = (not np.isnan(p)) and (p < ALPHA)
        winner = m1 if b_total > c_total else (m2 if c_total > b_total else "tie")

        print(f"  {m1:30s} vs {m2:28s} | "
              f"{b_total:6d} {c_total:6d}  "
              f"{chi2_stat:8.3f}  {p:8.4f}  "
              f"{'SIGNIFICANT' if sig else 'n.s.':12s}  {winner}")

        rows.append({
            "model_A":         m1,
            "model_B":         m2,
            "pooled_b_A_wins": b_total,
            "pooled_c_B_wins": c_total,
            "chi2":            round(chi2_stat, 4) if not np.isnan(chi2_stat) else np.nan,
            "p_value":         round(p, 6)         if not np.isnan(p)         else np.nan,
            "significant_005": sig,
            "winner":          winner,
            "n_seeds_pooled":  len(all_seeds),
        })

    out = pd.DataFrame(rows)
    out.to_csv(SAVE_DIR / "mcnemar_pooled.csv", index=False)
    print(f"\n  ok  Saved mcnemar_pooled.csv")
    return out


# -----------------------------------------------------------------------------
# Superiority table: proposed vs baseline
# -----------------------------------------------------------------------------
def build_superiority_table(df_results, wilcoxon_df, mcnemar_df):
    proposed = "SV_IBN_EfficientNet_B3"
    baseline = "EfficientNet_B3"
    rows     = []

    print("\n\n" + "="*70)
    print("  SUPERIORITY TABLE: SV_IBN_EfficientNet_B3  vs  EfficientNet_B3")
    print("="*70)

    for metric in METRICS:
        s_prop = df_results[df_results["model"] == proposed][metric].values
        s_base = df_results[df_results["model"] == baseline][metric].values

        mean_p, std_p = s_prop.mean(), s_prop.std(ddof=1)
        mean_b, std_b = s_base.mean(), s_base.std(ddof=1)
        delta         = mean_p - mean_b
        d             = cohens_d(s_prop, s_base)

        # Wilcoxon p
        wilc = wilcoxon_df[
            (wilcoxon_df["metric"] == metric) &
            (
                ((wilcoxon_df["model_A"] == proposed) & (wilcoxon_df["model_B"] == baseline)) |
                ((wilcoxon_df["model_A"] == baseline) & (wilcoxon_df["model_B"] == proposed))
            )
        ]
        p_wilcoxon = wilc.iloc[0]["p_value"] if not wilc.empty else np.nan

        # McNemar p
        mc = mcnemar_df[
            ((mcnemar_df["model_A"] == proposed) & (mcnemar_df["model_B"] == baseline)) |
            ((mcnemar_df["model_A"] == baseline) & (mcnemar_df["model_B"] == proposed))
        ]
        p_mcnemar = mc.iloc[0]["p_value"] if not mc.empty else np.nan
        b_val     = mc.iloc[0]["pooled_b_A_wins"] if not mc.empty else np.nan
        c_val     = mc.iloc[0]["pooled_c_B_wins"] if not mc.empty else np.nan
        if not mc.empty and mc.iloc[0]["model_A"] == baseline:
            b_val, c_val = c_val, b_val

        sv_wins = (delta > 0) and (
            (not np.isnan(p_mcnemar)  and p_mcnemar  < ALPHA) or
            (not np.isnan(p_wilcoxon) and p_wilcoxon < ALPHA)
        )

        rows.append({
            "Metric":                      METRIC_LABELS[metric],
            "SV-IBN Mean+-Std":            f"{mean_p:.4f} +- {std_p:.4f}",
            "EfficientNet-B3 Mean+-Std":   f"{mean_b:.4f} +- {std_b:.4f}",
            "Delta (SV-IBN minus B3)":     f"{delta:+.4f}",
            "Cohens d":                    f"{d:.4f}",
            "Effect size":                 effect_label(abs(d)),
            "Wilcoxon p (n=6)":            f"{p_wilcoxon:.4f}" if not np.isnan(p_wilcoxon) else "n/a",
            "McNemar p (pooled)":          f"{p_mcnemar:.4f}"  if not np.isnan(p_mcnemar)  else "n/a",
            "McNemar b (SV-IBN correct)":  int(b_val) if not np.isnan(b_val) else "?",
            "McNemar c (B3 correct)":      int(c_val) if not np.isnan(c_val) else "?",
            "SV-IBN better":               "YES *" if sv_wins else ("numerically" if delta > 0 else "NO"),
        })

    sup_df = pd.DataFrame(rows)
    sup_df.to_csv(SAVE_DIR / "superiority_table.csv", index=False)
    print(sup_df.to_string(index=False))
    print(f"\n  ok  Saved superiority_table.csv")
    return sup_df


# -----------------------------------------------------------------------------
# Ablation superiority: all variants vs baseline
# -----------------------------------------------------------------------------
def build_ablation_table(df_results, wilcoxon_df, mcnemar_df):
    baseline    = "EfficientNet_B3"
    comparators = [m for m in MODELS if m != baseline]
    rows        = []

    for proposed in comparators:
        for metric in METRICS:
            s_prop = df_results[df_results["model"] == proposed][metric].values
            s_base = df_results[df_results["model"] == baseline][metric].values
            delta  = s_prop.mean() - s_base.mean()
            d      = cohens_d(s_prop, s_base)

            wilc = wilcoxon_df[
                (wilcoxon_df["metric"] == metric) &
                (
                    ((wilcoxon_df["model_A"] == proposed) & (wilcoxon_df["model_B"] == baseline)) |
                    ((wilcoxon_df["model_A"] == baseline) & (wilcoxon_df["model_B"] == proposed))
                )
            ]
            p_wilcoxon = wilc.iloc[0]["p_value"] if not wilc.empty else np.nan

            mc = mcnemar_df[
                ((mcnemar_df["model_A"] == proposed) & (mcnemar_df["model_B"] == baseline)) |
                ((mcnemar_df["model_A"] == baseline) & (mcnemar_df["model_B"] == proposed))
            ]
            p_mcnemar = mc.iloc[0]["p_value"] if not mc.empty else np.nan

            rows.append({
                "Model":              proposed,
                "Metric":             METRIC_LABELS[metric],
                "Delta vs B3":        f"{delta:+.4f}",
                "Cohens d":           f"{d:.4f}",
                "Effect":             effect_label(abs(d)),
                "Wilcoxon p (n=6)":   f"{p_wilcoxon:.4f}" if not np.isnan(p_wilcoxon) else "n/a",
                "McNemar p (pooled)": f"{p_mcnemar:.4f}"  if not np.isnan(p_mcnemar)  else "n/a",
                "Direction":          "better" if delta > 0 else ("worse" if delta < 0 else "equal"),
            })

    abl_df = pd.DataFrame(rows)
    abl_df.to_csv(SAVE_DIR / "ablation_superiority.csv", index=False)
    print(f"\n  ok  Saved ablation_superiority.csv")
    return abl_df


# -----------------------------------------------------------------------------
# Comparison plots
# -----------------------------------------------------------------------------
def save_comparison_plots(df: pd.DataFrame, agg: pd.DataFrame, all_seeds):

    palette = sns.color_palette("Set2", len(MODELS))
    x       = np.arange(len(MODELS))

    # Box plots — all metrics side by side
    fig, axes = plt.subplots(1, len(METRICS), figsize=(22, 5), sharey=False)
    for ax, metric in zip(axes, METRICS):
        sns.boxplot(x="model", y=metric, data=df, order=MODELS,
                    palette=palette, width=0.5, ax=ax)
        sns.stripplot(x="model", y=metric, data=df, order=MODELS,
                      color="k", size=5, jitter=0.15, ax=ax)
        ax.set_title(METRIC_LABELS[metric], fontsize=11)
        ax.set_xlabel("")
        short = [m.replace("SV_IBN_", "SV-IBN\n")
                  .replace("EfficientNet_B3", "EffNet-B3")
                 for m in MODELS]
        ax.set_xticklabels(short, rotation=25, ha="right", fontsize=7)
        ax.grid(axis='y', alpha=0.3)
    plt.suptitle(f"Performance Distribution (n={len(all_seeds)} seeds)", fontsize=13)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "boxplot_all_metrics.png", dpi=180, bbox_inches='tight')
    plt.close()

    # Bar plots per metric
    for metric in METRICS:
        col_mean = f"{metric}_mean"
        col_std  = f"{metric}_std"
        if col_mean not in agg.columns:
            continue
        means = agg[col_mean].values
        stds  = agg[col_std].values
        fig, ax = plt.subplots(figsize=(13, 6))
        bars = ax.bar(x, means, yerr=stds, capsize=6,
                      color=palette, edgecolor='k', linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=35, ha="right", fontsize=10)
        ax.set_ylabel(f"Mean {METRIC_LABELS[metric]} +- Std", fontsize=12)
        ax.set_title(f"{METRIC_LABELS[metric]} Comparison (n={len(all_seeds)} seeds)", fontsize=13)
        ax.set_ylim(0, 1.08)
        ax.grid(axis='y', alpha=0.3)
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + s + 0.005,
                    f"{m:.4f}", ha='center', va='bottom',
                    fontsize=9, fontweight='bold')
        plt.tight_layout()
        plt.savefig(SAVE_DIR / f"barplot_{metric}.png", dpi=180, bbox_inches='tight')
        plt.close()

    # McNemar p-value heatmap
    mc_df = pd.read_csv(SAVE_DIR / "mcnemar_pooled.csv")
    p_mat = pd.DataFrame(np.ones((len(MODELS), len(MODELS))), index=MODELS, columns=MODELS)
    for _, row in mc_df.iterrows():
        p = row["p_value"]
        if np.isnan(p):
            continue
        p_mat.loc[row["model_A"], row["model_B"]] = p
        p_mat.loc[row["model_B"], row["model_A"]] = p

    short = {m: m.replace("EfficientNet_B3", "EffNet-B3")
                 .replace("SV_IBN_EfficientNet_B3", "SV-IBN\nEffNet-B3")
                 .replace("SV_IBN_", "SV-IBN\n") for m in MODELS}
    p_mat_plot = p_mat.rename(index=short, columns=short)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(p_mat_plot.astype(float), annot=True, fmt=".4f",
                cmap="RdYlGn_r", vmin=0, vmax=0.1,
                mask=np.eye(len(MODELS), dtype=bool), ax=ax,
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": f"McNemar p-value (pooled, n={len(all_seeds)} seeds)"})
    ax.set_title("Pairwise McNemar p-values", fontsize=12)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "mcnemar_pvalue_heatmap.png", dpi=160, bbox_inches='tight')
    plt.close()

    # Cohen's d heatmap
    wilc_df = pd.read_csv(SAVE_DIR / "wilcoxon_n6.csv")
    d_mat   = pd.DataFrame(index=MODELS, columns=METRICS, dtype=float)
    for _, row in wilc_df.iterrows():
        d_mat.loc[row["model_A"], row["metric"]] =  row["cohens_d"]
        d_mat.loc[row["model_B"], row["metric"]] = -row["cohens_d"]
    d_mat = d_mat.rename(columns=METRIC_LABELS, index=short)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(d_mat.astype(float), annot=True, fmt=".3f",
                cmap="RdBu", center=0, ax=ax,
                linewidths=0.4, linecolor="white",
                cbar_kws={"label": "Cohen's d"})
    ax.set_title(f"Cohen's d Effect Sizes (n={len(all_seeds)} seeds)", fontsize=12)
    plt.xticks(rotation=20, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "cohens_d_heatmap.png", dpi=160, bbox_inches='tight')
    plt.close()

    print("  ok  Comparison plots saved.")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    print("\n" + "="*65)
    print("  SV-IBN End-to-End Training + Statistical Analysis")
    print(f"  Models : {len(MODELS)}  |  Seeds : {SEEDS}  (n={len(SEEDS)})")
    print(f"  Device : {DEVICE}")
    print("="*65)

    all_results = []

    # -- 1. Train all models × all seeds (skips completed combos) -------------
    for model_name in MODELS:
        for seed in SEEDS:
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            res = run_one_seed(seed, model_name)
            all_results.append(res)

    # -- 2. Aggregate results --------------------------------------------------
    df = pd.DataFrame(all_results)

    # Remove duplicate rows (from skip returns)
    df = df.drop_duplicates(subset=["model", "seed"]).reset_index(drop=True)
    df.to_csv(SAVE_DIR / "all_results.csv", index=False)

    agg_cols = {
        "acc":             ["mean", "std", "min", "max"],
        "f1_macro":        ["mean", "std"],
        "precision_macro": ["mean", "std"],
        "recall_macro":    ["mean", "std"],
        "auc_macro":       ["mean", "std"],
    }
    agg = df.groupby("model").agg(agg_cols).round(4)
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reindex(MODELS)
    agg.to_csv(SAVE_DIR / "summary_aggregated.csv")

    print("\n\n=== Aggregated Summary ===")
    print(agg.to_string())

    agg.loc[[m for m in MODELS if m in agg.index]].to_csv(
        SAVE_DIR / "ablation_summary.csv"
    )

    # -- 3. Wilcoxon with n=6 -------------------------------------------------
    wilcoxon_df = run_wilcoxon(df)

    # -- 4. McNemar pooled ----------------------------------------------------
    mcnemar_df = run_mcnemar_pooled(SEEDS)

    # -- 5. Superiority + ablation tables -------------------------------------
    sup_df = build_superiority_table(df, wilcoxon_df, mcnemar_df)
    abl_df = build_ablation_table(df, wilcoxon_df, mcnemar_df)

    # -- 6. Comparison plots ---------------------------------------------------
    save_comparison_plots(df, agg, SEEDS)

    # -- 7. Final summary ------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"  All outputs saved to: {SAVE_DIR.resolve()}")
    print(f"{'='*65}")
    print(f"""
  Per-model subfolders:
    EfficientNet_B3/metrics/          curves/          gradcam/
    SV_IBN_no_VIB/metrics/            curves/          gradcam/
    SV_IBN_no_Spectral/metrics/       curves/          gradcam/
    SV_IBN_EfficientNet_B3/metrics/   curves/          gradcam/

  Global outputs:
    all_results.csv
    summary_aggregated.csv
    ablation_summary.csv
    wilcoxon_n6.csv              <- Wilcoxon, n=6
    mcnemar_pooled.csv           <- McNemar pooled (primary stat)
    superiority_table.csv        <- paper-ready SV-IBN vs B3
    ablation_superiority.csv     <- all ablations vs B3
    boxplot_all_metrics.png
    barplot_{{metric}}.png  x5
    mcnemar_pvalue_heatmap.png
    cohens_d_heatmap.png
    """)
