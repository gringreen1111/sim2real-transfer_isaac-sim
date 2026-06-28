"""
Weld Bead Segmentation - Synthetic-Only Training (mit_b2)
- Backbone: MiT-B2 (SegFormer encoder) + U-Net decoder
- Data: 합성이미지 5300장 단독 학습 (Isaac Sim Replicator)
- Task: RGB(3ch) -> Mbead 이진 마스크

Usage:
  python train_synth_mitb2.py

주의:
  - 합성 단독 학습 → 실이미지에는 sim2real gap 존재. 본 모델은 fine-tune 사전학습용 base.
  - 5300장이면 과적합 위험 낮으므로 증강/weight_decay 완화, epoch 축소.

출력물 (save_root/exp_name/ 아래):
  - split.json        : train/val/test 분할 정보
  - history.json      : epoch별 metric (매 에폭 갱신)
  - training_log.csv  : epoch별 metric CSV (매 에폭 즉시 append)
"""

import os
import sys
import time
import random
import glob
import json
import csv
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ==================================================================
# ★ CONFIG ★
# ==================================================================
CONFIG = {
    # --- 모델 ---
    "backbone":         "mit_b2",
    "encoder_weights":  "imagenet",
    "decoder":          "unet",
    "in_channels":      3,
    "classes":          1,

    # --- 데이터 ---
    "data_dir":         "/home/kim/issac_sim_synth",   # ← 합성이미지 경로 (수정 필요)
    "rgb_glob":         "rgb_*.png",                           # ← 실제 파일명 패턴
    "mask_subdir":      "binary_masks",
    "mask_glob":        "mask_*.png",
    "save_root":        "./checkpoints",
    "exp_name":         "jun_resumed1_mitb2",

    # --- 학습 하이퍼파라미터 ---
    "epochs":           60,           # 5300장이면 60ep로 충분 수렴
    "batch_size":       2,
    "accum_steps":      4,            # effective batch = 8
    "lr_encoder":       3e-5,
    "lr_decoder":       2e-4,
    "weight_decay":     0.01,         # 데이터 충분 → 과적합 압력 낮음, 0.05→0.01 완화
    "img_size":         1024,
    "warmup_epochs":    3,            # 데이터 많으므로 짧게
    "patience":         12,           # 데이터 충분 → 짧게
    "split_ratio":      (0.85, 0.10, 0.05),  # 5300장: train 4505 / val 530 / test 265
    "freeze_encoder_epochs": 1,       # 데이터 충분 → 1ep만 decoder 적응
    "stronger_aug":     False,        # 5300장 → 강한 증강 불필요(도메인 갭만 줄이는 약한 증강)

    # --- 기타 ---
    "use_amp":          True,
    "seed":             42,
    "num_workers":      8,            # 데이터 많으므로 증가
    "bead_threshold":   0.4,
}
# ==================================================================


# ==================================================================
# 0. Reproducibility
# ==================================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==================================================================
# 1. Dataset
# ==================================================================
class WeldBeadDataset(Dataset):
    def __init__(self, rgb_paths, mask_paths, transform=None):
        assert len(rgb_paths) == len(mask_paths)
        self.rgb_paths = rgb_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb = np.array(Image.open(self.rgb_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            transformed = self.transform(image=rgb, mask=mask)
            rgb = transformed["image"]
            mask = transformed["mask"]

        if isinstance(mask, torch.Tensor):
            mask = mask.unsqueeze(0).float()
        else:
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return rgb, mask


# ==================================================================
# 2. Loss
# ==================================================================
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def dice_loss(self, pred, target, smooth=1.0):
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum(dim=(2, 3))
        union = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()

    def forward(self, pred, target):
        return self.bce(pred, target) + self.dice_loss(pred, target)


# ==================================================================
# 3. Metrics
# ==================================================================
@torch.no_grad()
def compute_iou(pred, target, threshold: float = 0.5):
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_binary * target).sum(dim=(1, 2, 3))
    union = pred_binary.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    iou = torch.where(union > 0, intersection / union.clamp(min=1e-6),
                      torch.ones_like(union))
    return iou.mean().item()


@torch.no_grad()
def compute_dice(pred, target, threshold: float = 0.5):
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_binary * target).sum(dim=(1, 2, 3))
    total = pred_binary.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = torch.where(total > 0, 2.0 * intersection / total.clamp(min=1e-6),
                       torch.ones_like(total))
    return dice.mean().item()


# ==================================================================
# 4. Augmentation
#   - 합성 5300장: 강한 증강 불필요. sim2real 갭 완화 목적의 약한 photometric 위주.
# ==================================================================
def _safe_gauss_noise(p: float = 0.3):
    try:
        return A.GaussNoise(std_range=(0.02, 0.1), p=p)
    except TypeError:
        return A.GaussNoise(var_limit=(10.0, 50.0), p=p)


def get_train_transform(img_size: int = 1024, stronger: bool = False):
    if stronger:
        return A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size,
                          border_mode=0, value=0, mask_value=0),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10,
                               rotate_limit=5, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.4,
                                       contrast_limit=0.4, p=0.7),
            A.ColorJitter(brightness=0.5, contrast=0.5,
                          saturation=0.4, hue=0.15, p=0.6),
            A.RandomGamma(gamma_limit=(50, 150), p=0.4),
            A.CLAHE(clip_limit=4.0, p=0.4),
            A.CoarseDropout(num_holes_range=(1, 4),
                            hole_height_range=(30, 120),
                            hole_width_range=(30, 120), fill=0, p=0.3),
            _safe_gauss_noise(p=0.4),
            A.MotionBlur(blur_limit=5, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        # 5300장 기본: 약~중 강도 증강 (sim2real 대비 photometric 위주)
        return A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size,
                          border_mode=0, value=0, mask_value=0),
            A.ShiftScaleRotate(shift_limit=0.04, scale_limit=0.08,
                               rotate_limit=5, p=0.6),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3,
                                       contrast_limit=0.3, p=0.6),
            A.ColorJitter(brightness=0.4, contrast=0.4,
                          saturation=0.3, hue=0.1, p=0.5),
            A.RandomGamma(gamma_limit=(70, 130), p=0.3),
            A.CLAHE(clip_limit=4.0, p=0.3),
            _safe_gauss_noise(p=0.3),
            A.MotionBlur(blur_limit=3, p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


def get_val_transform(img_size: int = 1024):
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size,
                      border_mode=0, value=0, mask_value=0),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ==================================================================
# 5. Model
# ==================================================================
def build_model(cfg):
    decoder = cfg["decoder"].lower()
    common = dict(
        encoder_name=cfg["backbone"],
        encoder_weights=cfg["encoder_weights"],
        in_channels=cfg["in_channels"],
        classes=cfg["classes"],
    )
    if decoder == "unet":
        return smp.Unet(**common)
    elif decoder == "unetplusplus":
        return smp.UnetPlusPlus(**common)
    elif decoder == "manet":
        return smp.MAnet(**common)
    elif decoder == "fpn":
        return smp.FPN(**common)
    elif decoder == "deeplabv3plus":
        return smp.DeepLabV3Plus(**common)
    else:
        raise ValueError(f"Unknown decoder: {decoder}")


def set_encoder_requires_grad(model, requires_grad: bool):
    for p in model.encoder.parameters():
        p.requires_grad = requires_grad


# ==================================================================
# 6. Optimizer & Scheduler (differential LR)
# ==================================================================
def build_optimizer(model, cfg):
    encoder_params = list(model.encoder.parameters())
    decoder_params = [p for n, p in model.named_parameters()
                      if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": cfg["lr_encoder"]},
            {"params": decoder_params, "lr": cfg["lr_decoder"]},
        ],
        weight_decay=cfg["weight_decay"],
    )
    print(f"  [Optimizer] encoder lr={cfg['lr_encoder']:.2e}, "
          f"decoder lr={cfg['lr_decoder']:.2e}")
    return optimizer


def get_lr_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ==================================================================
# 7. Train / Validate
# ==================================================================
def train_one_epoch(model, loader, criterion, optimizer, scheduler,
                    scaler, device, use_amp, accum_steps):
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    optimizer.zero_grad()

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, masks) / accum_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % accum_steps == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * accum_steps
        total_iou += compute_iou(outputs.detach(), masks)

        if (batch_idx + 1) % 100 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  Batch {batch_idx+1}/{len(loader)} | "
                  f"Loss: {loss.item()*accum_steps:.4f} | lr: {current_lr:.2e}")

    n = len(loader)
    return total_loss / n, total_iou / n


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss = total_iou = total_dice = 0.0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, masks)
        total_loss += loss.item()
        total_iou += compute_iou(outputs, masks)
        total_dice += compute_dice(outputs, masks)
    n = len(loader)
    return total_loss / n, total_iou / n, total_dice / n


# ==================================================================
# 8. Data Collection & Split (stem 기반 매칭)
# ==================================================================
def collect_pairs(cfg):
    rgb_paths = sorted(glob.glob(os.path.join(cfg["data_dir"], cfg["rgb_glob"])))
    mask_dir = os.path.join(cfg["data_dir"], cfg["mask_subdir"])
    mask_paths_all = glob.glob(os.path.join(mask_dir, cfg["mask_glob"]))

    def key(p):
        stem = os.path.splitext(os.path.basename(p))[0]
        digits = "".join(ch for ch in stem if ch.isdigit())
        return digits

    mask_by_key = {key(m): m for m in mask_paths_all}
    rgb_matched, mask_matched = [], []
    for r in rgb_paths:
        k = key(r)
        if k in mask_by_key:
            rgb_matched.append(r)
            mask_matched.append(mask_by_key[k])

    return rgb_matched, mask_matched


def get_or_create_split(rgb_paths, mask_paths, split_file, ratio, seed):
    if os.path.exists(split_file):
        print(f"[SPLIT] 기존 split 로드: {split_file}")
        with open(split_file, "r") as f:
            split = json.load(f)
        return (
            [x[0] for x in split["train"]], [x[1] for x in split["train"]],
            [x[0] for x in split["val"]],   [x[1] for x in split["val"]],
            [x[0] for x in split["test"]],  [x[1] for x in split["test"]],
        )

    n_total = len(rgb_paths)
    r_train, r_val, _ = ratio
    n_train = int(n_total * r_train)
    n_val = int(n_total * r_val)
    indices = np.random.RandomState(seed).permutation(n_total)
    tr, va = indices[:n_train], indices[n_train:n_train + n_val]
    te = indices[n_train + n_val:]

    pick = lambda idxs, src: [src[i] for i in idxs]
    train_rgb, train_mask = pick(tr, rgb_paths), pick(tr, mask_paths)
    val_rgb, val_mask = pick(va, rgb_paths), pick(va, mask_paths)
    test_rgb, test_mask = pick(te, rgb_paths), pick(te, mask_paths)

    split_info = {
        "seed": seed, "n_total": n_total, "ratio": list(ratio),
        "train": [[r, m] for r, m in zip(train_rgb, train_mask)],
        "val":   [[r, m] for r, m in zip(val_rgb, val_mask)],
        "test":  [[r, m] for r, m in zip(test_rgb, test_mask)],
    }
    os.makedirs(os.path.dirname(split_file), exist_ok=True)
    with open(split_file, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"[SPLIT] 새 split 생성: {split_file}")
    return train_rgb, train_mask, val_rgb, val_mask, test_rgb, test_mask


# ==================================================================
# 9. Main
# ==================================================================
def main():
    cfg = CONFIG
    set_seed(cfg["seed"])

    exp_dir = os.path.join(cfg["save_root"], cfg["exp_name"])
    os.makedirs(exp_dir, exist_ok=True)
    split_file = os.path.join(exp_dir, "split.json")
    history_path = os.path.join(exp_dir, "history.json")
    csv_path = os.path.join(exp_dir, "training_log.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}\nSynthetic (mit_b2) — {cfg['exp_name']}\n{'='*60}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    rgb_paths, mask_paths = collect_pairs(cfg)
    if len(rgb_paths) == 0:
        print(f"[ERROR] No matched RGB/mask pairs in {cfg['data_dir']}")
        sys.exit(1)
    print(f"\nMatched image-mask pairs: {len(rgb_paths)}")

    train_rgb, train_mask, val_rgb, val_mask, test_rgb, test_mask = \
        get_or_create_split(rgb_paths, mask_paths, split_file,
                            cfg["split_ratio"], cfg["seed"])
    print(f"Train: {len(train_rgb)}, Val: {len(val_rgb)}, Test: {len(test_rgb)}")

    train_ds = WeldBeadDataset(
        train_rgb, train_mask,
        transform=get_train_transform(cfg["img_size"], stronger=cfg["stronger_aug"]))
    val_ds = WeldBeadDataset(
        val_rgb, val_mask, transform=get_val_transform(cfg["img_size"]))

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=cfg["num_workers"], pin_memory=True)

    model = build_model(cfg).to(device)
    print(f"\nModel: {cfg['decoder']} + {cfg['backbone']} "
          f"(in_ch={cfg['in_channels']}, weights={cfg['encoder_weights']})")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")

    criterion = BCEDiceLoss()
    optimizer = build_optimizer(model, cfg)
    freeze_epochs = cfg.get("freeze_encoder_epochs", 0)
    if freeze_epochs > 0:
        print(f"  [Encoder] freezing for first {freeze_epochs} epochs")
        set_encoder_requires_grad(model, False)

    steps_per_epoch = max(1, len(train_loader) // cfg["accum_steps"])
    scheduler = get_lr_scheduler(optimizer, cfg["warmup_epochs"],
                                 cfg["epochs"], steps_per_epoch)
    scaler = GradScaler(enabled=cfg["use_amp"])

    # --- CSV 로그 초기화 (헤더 작성) ---
    csv_fields = ["epoch", "train_loss", "train_iou",
                  "val_loss", "val_iou", "val_dice", "enc_lr", "dec_lr", "time"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
    print(f"CSV log: {csv_path}")

    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    history = []
    t_start = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            print(f"\n  [Encoder] unfreezing at epoch {epoch}")
            set_encoder_requires_grad(model, True)

        print(f"\n{'-'*60}")
        print(f"[{cfg['exp_name']}] Epoch {epoch}/{cfg['epochs']} "
              f"(enc lr={optimizer.param_groups[0]['lr']:.2e}, "
              f"dec lr={optimizer.param_groups[1]['lr']:.2e})")
        print(f"{'-'*60}")

        t0 = time.time()
        train_loss, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, cfg["use_amp"], cfg["accum_steps"])
        val_loss, val_iou, val_dice = validate(
            model, val_loader, criterion, device, cfg["use_amp"])
        dt = time.time() - t0

        print(f"  Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}")
        print(f"  Time  - {dt:.1f}s")

        enc_lr = optimizer.param_groups[0]["lr"]
        dec_lr = optimizer.param_groups[1]["lr"]

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_iou": train_iou,
            "val_loss": val_loss, "val_iou": val_iou, "val_dice": val_dice,
            "enc_lr": enc_lr, "dec_lr": dec_lr, "time": dt,
        })

        # --- CSV에 이번 에폭 기록 (매 에폭 즉시 저장) ---
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writerow({
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "train_iou": f"{train_iou:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_iou": f"{val_iou:.6f}",
                "val_dice": f"{val_dice:.6f}",
                "enc_lr": f"{enc_lr:.8f}",
                "dec_lr": f"{dec_lr:.8f}",
                "time": f"{dt:.1f}",
            })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss, "val_iou": val_iou, "val_dice": val_dice,
                "config": cfg,
            }, os.path.join(exp_dir, "best.pth"))
            print(f"  *** Best saved! Val Loss: {val_loss:.4f}, "
                  f"IoU: {val_iou:.4f}, Dice: {val_dice:.4f} ***")
        else:
            no_improve += 1
            print(f"  No improvement for {no_improve} epoch(s)")
            if no_improve >= cfg["patience"]:
                print(f"\n[Early Stopping] at epoch {epoch}")
                break

        # --- History 저장 (매 에폭) ---
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    total_time = time.time() - t_start
    print(f"\n{'='*60}\nDONE — {cfg['exp_name']}\n{'='*60}")
    print(f"  Best Epoch: {best_epoch}, Best Val Loss: {best_val_loss:.4f}")
    print(f"  Total Time: {total_time/60:.1f} min")
    print(f"  Split saved:   {split_file}")
    print(f"  History saved: {history_path}")
    print(f"  CSV saved:     {csv_path}")


if __name__ == "__main__":
    main()