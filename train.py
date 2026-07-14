"""
train.py — PyTorch — Nhận diện thời tiết (11 lớp)
==================================================
Chiến lược tối ưu chất lượng tối đa:
  - Backbone    : EfficientNetV2-S (ImageNet pretrained)
  - Augmentation: torchvision transforms (RandAugment + ColorJitter + GaussianBlur)
  - LR Schedule : Cosine Annealing với Linear Warmup
  - Loss        : CrossEntropy + Label Smoothing 0.1
  - Class Weight: Tự động tính bù mất cân bằng
  - 2 giai đoạn : Freeze backbone → Unfreeze fine-tune
  - TTA         : Test-Time Augmentation (5 views)
  - Output      : weather_model.pth + training_charts.png
"""

import os, json, math, time, copy
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import EfficientNet_V2_S_Weights
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────── #
#  CẤU HÌNH
# ─────────────────────────────────────────── #
TRAIN_DIR    = os.path.join("data", "train")
VAL_DIR      = os.path.join("data", "val")
TEST_DIR     = os.path.join("data", "test")
JSON_PATH    = "class_names.json"
MODEL_PATH   = "weather_model.pth"
CSV_LOG      = "training_log.csv"

IMG_SIZE     = 260           # EfficientNetV2-S native size
BATCH_SIZE   = 32
NUM_WORKERS  = 0             # 0 = dùng main thread (an toàn trên Windows)

# Giai đoạn 1: Train head
EPOCHS_HEAD  = 20
LR_HEAD      = 3e-3

# Giai đoạn 2: Fine-tune
EPOCHS_FT    = 40
LR_FT        = 5e-5
UNFREEZE_RATIO = 0.3         # Giữ đóng 30% tầng đầu

WARMUP_EPOCHS = 3
LABEL_SMOOTH  = 0.1
WEIGHT_DECAY  = 1e-4
EARLY_STOP_PATIENCE = 10


# ─────────────────────────────────────────── #
#  PHÁT HIỆN THIẾT BỊ
# ─────────────────────────────────────────── #
def setup_device():
    line = "─" * 52
    if torch.cuda.is_available():
        device    = torch.device("cuda")
        gpu_name  = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        vram_free  = (torch.cuda.get_device_properties(0).total_memory
                      - torch.cuda.memory_allocated(0)) / 1024**3
        torch.backends.cudnn.benchmark = True   # Tăng tốc CUDA

        print(f"\n  ╔{line}╗")
        print(f"  ║  🚀  CHẾ ĐỘ: NVIDIA CUDA GPU (nhanh ~10-20x CPU)       ║")
        print(f"  ╠{line}╣")
        print(f"  ║  🎮  GPU   : {gpu_name:<38}║")
        print(f"  ║  ⚡  CUDA  : {torch.version.cuda:<10}  cuDNN: {torch.backends.cudnn.version():<22}║")
        print(f"  ║  💾  VRAM  : {vram_free:.1f} GB khả dụng / {vram_total:.1f} GB tổng{' '*15}║")
        print(f"  ╚{line}╝\n")
    else:
        device = torch.device("cpu")
        n_cpu  = os.cpu_count()
        torch.set_num_threads(n_cpu)

        print(f"\n  ╔{line}╗")
        print(f"  ║  🐢  CHẾ ĐỘ: CPU (chậm hơn GPU ~10-20 lần)            ║")
        print(f"  ╠{line}╣")
        print(f"  ║  💻  Số luồng  : {n_cpu:<35}║")
        print(f"  ║  💡  Gợi ý     : Cài PyTorch CUDA để tăng tốc GPU      ║")
        print(f"  ║      pip install torch torchvision --index-url          ║")
        print(f"  ║      https://download.pytorch.org/whl/cu121             ║")
        print(f"  ╚{line}╝\n")

    return device


# ─────────────────────────────────────────── #
#  DATASET
# ─────────────────────────────────────────── #
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class WeatherDataset(Dataset):
    def __init__(self, directory, class_names, transform=None):
        self.transform   = transform
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.samples     = []

        for cls in class_names:
            cls_dir = Path(directory) / cls
            if not cls_dir.is_dir():
                continue
            for fpath in cls_dir.iterdir():
                if fpath.suffix.lower() in VALID_EXT:
                    self.samples.append((str(fpath), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(augment=False):
    """Trả về transform phù hợp với EfficientNetV2-S."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if augment:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.1),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def compute_class_weights(dataset, num_classes, device):
    """Tính class weights để bù mất cân bằng."""
    counts = torch.zeros(num_classes)
    for _, label in dataset.samples:
        counts[label] += 1
    max_c = counts.max()
    weights = max_c / counts.clamp(min=1)
    return weights.to(device)


def build_loaders(class_names):
    train_ds = WeatherDataset(TRAIN_DIR, class_names, get_transforms(augment=True))
    val_ds   = WeatherDataset(VAL_DIR,   class_names, get_transforms(augment=False))
    test_ds  = WeatherDataset(TEST_DIR,  class_names, get_transforms(augment=False))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=False)

    print(f"    Train: {len(train_ds):>5} ảnh | {len(train_loader)} batches")
    print(f"    Val  : {len(val_ds):>5} ảnh | {len(val_loader)} batches")
    print(f"    Test : {len(test_ds):>5} ảnh | {len(test_loader)} batches")
    return train_loader, val_loader, test_loader, train_ds


# ─────────────────────────────────────────── #
#  MODEL
# ─────────────────────────────────────────── #
def build_model(num_classes, device, freeze_backbone=True):
    """EfficientNetV2-S với classification head tùy chỉnh."""
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model   = models.efficientnet_v2_s(weights=weights)

    # Thay head phân loại
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes),
    )

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
        print(f"  🔒 Backbone đóng băng | Chỉ train classification head")
    else:
        print(f"  🔓 Toàn bộ model đang được train")

    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  📐 Tổng params: {total:,} | Trainable: {trainable:,}")

    return model.to(device)


def unfreeze_top_layers(model, unfreeze_ratio=0.3):
    """Mở khoá phần cuối của backbone để fine-tune."""
    feature_layers = list(model.features.children())
    n_total   = len(feature_layers)
    n_freeze  = int(n_total * (1 - unfreeze_ratio))

    for i, layer in enumerate(feature_layers):
        for param in layer.parameters():
            param.requires_grad = (i >= n_freeze)

    # Mở khoá toàn bộ head
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  🔓 Đóng băng {n_freeze}/{n_total} blocks đầu | Trainable params: {trainable:,}")


# ─────────────────────────────────────────── #
#  LR SCHEDULE — COSINE WARMUP
# ─────────────────────────────────────────── #
def get_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    """Warmup tuyến tính rồi Cosine Annealing."""
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps  = total_epochs  * steps_per_epoch

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(1e-7, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────── #
#  VÒNG LẶP TRAIN / EVAL
# ─────────────────────────────────────────── #
def run_epoch(model, loader, criterion, optimizer, scheduler, device, train=True):
    model.train(train)
    total_loss = total_correct = total_top3 = total_n = 0

    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            logits = model(imgs)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            # Metrics
            with torch.no_grad():
                top1 = logits.argmax(dim=1).eq(labels).sum().item()
                top3 = torch.topk(logits, k=min(3, logits.size(1)), dim=1).indices
                top3_correct = top3.eq(labels.unsqueeze(1)).any(dim=1).sum().item()

            total_loss    += loss.item() * imgs.size(0)
            total_correct += top1
            total_top3    += top3_correct
            total_n       += imgs.size(0)

    return (
        total_loss    / total_n,
        total_correct / total_n,
        total_top3    / total_n,
    )


# ─────────────────────────────────────────── #
#  TRAINING LOOP (1 GIAI ĐOẠN)
# ─────────────────────────────────────────── #
def train_stage(model, train_loader, val_loader, criterion, device,
                lr, epochs, stage_name, warmup_epochs=WARMUP_EPOCHS):
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=WEIGHT_DECAY,
    )
    scheduler = get_scheduler(optimizer, warmup_epochs, epochs, len(train_loader))

    history = {k: [] for k in
               ["train_loss", "val_loss", "train_acc", "val_acc",
                "train_top3", "val_top3"]}

    best_val_acc   = 0.0
    best_state     = None
    no_improve     = 0
    csv_header_written = os.path.exists(CSV_LOG)

    print(f"\n  {'─'*55}")
    print(f"  {stage_name} | {epochs} epochs | LR={lr}")
    print(f"  {'─'*55}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc, tr_top3 = run_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, train=True)
        va_loss, va_acc, va_top3 = run_epoch(
            model, val_loader, criterion, None, None, device, train=False)

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["train_top3"].append(tr_top3)
        history["val_top3"].append(va_top3)

        # Log CSV
        with open(CSV_LOG, "a") as f:
            if not csv_header_written:
                f.write("stage,epoch,train_loss,val_loss,train_acc,val_acc,train_top3,val_top3,lr\n")
                csv_header_written = True
            f.write(f"{stage_name},{epoch},{tr_loss:.4f},{va_loss:.4f},"
                    f"{tr_acc:.4f},{va_acc:.4f},{tr_top3:.4f},{va_top3:.4f},{lr_now:.2e}\n")

        improved = "⬆" if va_acc > best_val_acc else " "
        print(f"  Ep {epoch:3d}/{epochs} | "
              f"Loss {tr_loss:.4f}/{va_loss:.4f} | "
              f"Acc {tr_acc*100:.1f}/{va_acc*100:.1f}% | "
              f"Top3 {va_top3*100:.1f}% | "
              f"LR {lr_now:.1e} | {elapsed:.0f}s {improved}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state   = copy.deepcopy(model.state_dict())
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"\n  ⏹ Early stopping tại epoch {epoch} (patience={EARLY_STOP_PATIENCE})")
                break

    # Restore best
    if best_state:
        model.load_state_dict(best_state)
    print(f"\n  🏆 Best val accuracy: {best_val_acc*100:.2f}%")
    return history, best_val_acc


# ─────────────────────────────────────────── #
#  TTA (TEST-TIME AUGMENTATION)
# ─────────────────────────────────────────── #
def evaluate_tta(model, test_dir, class_names, device, n_tta=5):
    """TTA với 5 views: original + flip + 3 crop variants."""
    model.eval()
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    base_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    tta_tfs = [
        base_tf,
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                             transforms.RandomHorizontalFlip(p=1.0),
                             transforms.ToTensor(),
                             transforms.Normalize(mean, std)]),
        transforms.Compose([transforms.Resize((int(IMG_SIZE*1.1), int(IMG_SIZE*1.1))),
                             transforms.CenterCrop(IMG_SIZE),
                             transforms.ToTensor(),
                             transforms.Normalize(mean, std)]),
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                             transforms.RandomRotation(10),
                             transforms.ToTensor(),
                             transforms.Normalize(mean, std)]),
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                             transforms.ColorJitter(brightness=0.2),
                             transforms.ToTensor(),
                             transforms.Normalize(mean, std)]),
    ]

    class_to_idx = {c: i for i, c in enumerate(class_names)}
    correct = total = 0

    print(f"\n  🔁 TTA ({n_tta} views/ảnh)...")
    with torch.no_grad():
        for cls in class_names:
            cls_dir = Path(test_dir) / cls
            if not cls_dir.is_dir():
                continue
            for fpath in cls_dir.iterdir():
                if fpath.suffix.lower() not in VALID_EXT:
                    continue
                img = Image.open(fpath).convert("RGB")
                preds = []
                for tf in tta_tfs[:n_tta]:
                    t = tf(img).unsqueeze(0).to(device)
                    out = torch.softmax(model(t), dim=1)
                    preds.append(out)
                avg = torch.stack(preds).mean(dim=0)
                if avg.argmax().item() == class_to_idx[cls]:
                    correct += 1
                total += 1

    tta_acc = correct / total if total > 0 else 0
    print(f"  ✅ TTA Accuracy: {tta_acc*100:.2f}%  ({correct}/{total})")
    return tta_acc


# ─────────────────────────────────────────── #
#  VẼ BIỂU ĐỒ
# ─────────────────────────────────────────── #
def plot_history(h1, h2, save_path="training_charts.png"):
    """Vẽ 4 biểu đồ Accuracy / Loss / Top-3 / Overfitting."""
    def merge(key):
        a = h1.get(key, [])
        b = h2.get(key, [])
        off = len(a)
        return list(range(1, len(a)+1)), a, list(range(off+1, off+len(b)+1)), b

    plt.style.use("dark_background")
    C  = ["#a78bfa", "#60a5fa", "#34d399", "#fbbf24"]
    fig = plt.figure(figsize=(16, 10), facecolor="#0f0f1a")
    fig.suptitle("Training History — Weather CNN (EfficientNetV2-S / PyTorch)",
                 fontsize=14, fontweight="bold", color="white", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#94a3b8", labelsize=9)
        for s in ax.spines.values():
            s.set_edgecolor("#334155")
        ax.grid(color="#334155", ls="--", lw=0.5, alpha=0.7)

    sep = len(h1.get("train_acc", [])) + 0.5

    def draw(ax, key_tr, key_va, title, ylabel):
        e1, tr1, e2, tr2 = merge(key_tr)
        _,  va1, _,  va2 = merge(key_va)
        ax.plot(e1, tr1, color=C[0], lw=1.8, label="Train S1")
        ax.plot(e1, va1, color=C[2], lw=1.8, label="Val S1")
        if e2:
            ax.plot(e2, tr2, color=C[1], lw=1.8, ls="--", label="Train S2")
            ax.plot(e2, va2, color=C[3], lw=1.8, ls="--", label="Val S2")
            ax.axvline(sep, color="#ef4444", lw=1.2, ls=":", label="Fine-tune")
        ax.set_title(title, color="white", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", color="#94a3b8", fontsize=9)
        ax.set_ylabel(ylabel, color="#94a3b8", fontsize=9)
        ax.legend(fontsize=8, facecolor="#0f0f1a", edgecolor="#334155", labelcolor="white")

    draw(axes[0], "train_acc",  "val_acc",  "Accuracy",       "Accuracy")
    axes[0].set_ylim(0, 1.05)
    draw(axes[1], "train_loss", "val_loss", "Loss (Label Smoothing CE)", "Loss")
    draw(axes[2], "train_top3", "val_top3", "Top-3 Accuracy", "Top-3 Acc")
    axes[2].set_ylim(0, 1.05)

    # Overfitting gap
    ax = axes[3]
    all_tr = h1.get("train_acc", []) + h2.get("train_acc", [])
    all_va = h1.get("val_acc",   []) + h2.get("val_acc",   [])
    ep_all = list(range(1, len(all_tr)+1))
    ax.plot(ep_all, all_tr, color=C[0], lw=1.8, label="Train Acc")
    ax.plot(ep_all, all_va, color=C[2], lw=1.8, label="Val Acc")
    ax.fill_between(ep_all, all_va, all_tr, alpha=0.12, color="#ef4444", label="Gap")
    if h1.get("train_acc") and h2.get("train_acc"):
        ax.axvline(sep, color="#ef4444", lw=1.2, ls=":", label="Fine-tune")
    ax.set_ylim(0, 1.05)
    ax.set_title("Overfitting Analysis", color="white", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Accuracy", color="#94a3b8", fontsize=9)
    ax.legend(fontsize=8, facecolor="#0f0f1a", edgecolor="#334155", labelcolor="white")

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  📊 Biểu đồ đã lưu: {save_path}")


# ─────────────────────────────────────────── #
#  MAIN
# ─────────────────────────────────────────── #
def main():
    print("=" * 60)
    print("  HUẤN LUYỆN NHẬN DIỆN THỜI TIẾT — PyTorch")
    print("=" * 60)

    # 0. Phần cứng
    print("\n[0/6] Cấu hình phần cứng...")
    device = setup_device()

    # 1. Nhãn
    print("\n[1/6] Đọc nhãn...")
    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError("❌ Chưa có class_names.json — chạy export_labels.py trước!")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    class_names = raw["class_names"] if isinstance(raw, dict) else raw
    num_classes  = len(class_names)
    print(f"  ✅ {num_classes} lớp: {class_names}")

    # 2. Data loaders
    print("\n[2/6] Xây dựng DataLoaders...")
    train_loader, val_loader, test_loader, train_ds = build_loaders(class_names)

    class_weights = compute_class_weights(train_ds, num_classes, device)
    print(f"\n  ⚖️  Class weights (top 3 thấp nhất): "
          + ", ".join(f"{class_names[i]}={class_weights[i]:.2f}"
                      for i in class_weights.topk(3).indices.tolist()))

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTH,
    )

    # 3. Xây model
    print("\n[3/6] Khởi tạo EfficientNetV2-S...")
    model = build_model(num_classes, device, freeze_backbone=True)

    # 4. Giai đoạn 1 — Train head
    print(f"\n[4/6] Giai đoạn 1 — Train Classification Head ({EPOCHS_HEAD} epochs)...")
    h1, best1 = train_stage(model, train_loader, val_loader, criterion,
                             device, LR_HEAD, EPOCHS_HEAD, "Stage1-Head")
    torch.save({"state_dict": model.state_dict(),
                "class_names": class_names}, "best_stage1.pth")

    # 5. Giai đoạn 2 — Fine-tune
    print(f"\n[5/6] Giai đoạn 2 — Fine-tuning ({EPOCHS_FT} epochs)...")
    unfreeze_top_layers(model, UNFREEZE_RATIO)
    h2, best2 = train_stage(model, train_loader, val_loader, criterion,
                             device, LR_FT, EPOCHS_FT, "Stage2-FineTune",
                             warmup_epochs=WARMUP_EPOCHS)

    # 6. Lưu + TTA + Chart
    print("\n[6/6] Đánh giá TTA, lưu model và biểu đồ...")
    tta_acc = evaluate_tta(model, TEST_DIR, class_names, device, n_tta=5)

    torch.save({
        "state_dict":  model.state_dict(),
        "class_names": class_names,
        "img_size":    IMG_SIZE,
        "num_classes": num_classes,
        "tta_accuracy": round(tta_acc * 100, 2),
    }, MODEL_PATH)
    print(f"  💾 Model đã lưu: {MODEL_PATH}")

    # Cập nhật JSON
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        jdata = json.load(f)
    jdata.update({"model_path": MODEL_PATH, "img_size": IMG_SIZE,
                  "framework": "pytorch", "tta_accuracy": round(tta_acc * 100, 2)})
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(jdata, f, indent=4, ensure_ascii=False)

    # Biểu đồ
    print("\n  📊 Đang vẽ biểu đồ training...")
    plot_history(h1, h2, save_path="training_charts.png")

    print("\n" + "=" * 60)
    print("  🎉 HUẤN LUYỆN HOÀN TẤT!")
    print(f"  📦 Model       : {MODEL_PATH}")
    print(f"  📊 Val Acc S1  : {best1*100:.2f}%")
    print(f"  📊 Val Acc S2  : {best2*100:.2f}%")
    print(f"  🔁 TTA Acc     : {tta_acc*100:.2f}%")
    print(f"  📄 Log         : {CSV_LOG}")
    print(f"  🖼️  Biểu đồ   : training_charts.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
