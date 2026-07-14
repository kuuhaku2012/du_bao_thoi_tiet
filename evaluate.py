"""
evaluate.py — Đánh giá mô hình và Vẽ Ma trận Nhầm lẫn (Confusion Matrix)
======================================================================
Mục đích:
  - Tải mô hình đã train từ `weather_model.pth`
  - Chạy dự đoán trên tập dữ liệu Test (`data/test`) hoặc Validation (`data/val`)
  - Tính toán các chỉ số đánh giá chi tiết (Precision, Recall, F1-score, Accuracy)
  - Vẽ và lưu ma trận nhầm lẫn (Confusion Matrix) chuyên nghiệp dưới dạng hình ảnh
  - Xuất báo cáo đánh giá ra file Markdown và JSON
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

# ─────────────────────────────────────────── #
#  CẤU HÌNH ĐƯỜNG DẪN
# ─────────────────────────────────────────── #
TEST_DIR     = os.path.join("data", "test")
VAL_DIR      = os.path.join("data", "val")
JSON_PATH    = "class_names.json"
MODEL_PATH   = "weather_model.pth"
OUTPUT_IMG   = "confusion_matrix.png"
OUTPUT_REPORT = "evaluation_report.md"

# ─────────────────────────────────────────── #
#  PHÁT HIỆN THIẾT BỊ
# ─────────────────────────────────────────── #
def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Sử dụng GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("🐢 Sử dụng CPU để đánh giá.")
    return device

# ─────────────────────────────────────────── #
#  DATASET & TRANSFORMS
# ─────────────────────────────────────────── #
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class WeatherDataset(Dataset):
    def __init__(self, directory, class_names, img_size, transform=None):
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

def get_eval_transforms(img_size):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

# ─────────────────────────────────────────── #
#  TẢI MÔ HÌNH
# ─────────────────────────────────────────── #
def load_model(model_path, num_classes, device):
    base = models.efficientnet_v2_s(weights=None)
    in_features = base.classifier[1].in_features
    base.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes),
    )
    
    print(f"📂 Đang tải trọng số từ {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    base.load_state_dict(checkpoint["state_dict"])
    base.to(device)
    base.eval()
    return base

# ─────────────────────────────────────────── #
#  ĐÁNH GIÁ MÔ HÌNH
# ─────────────────────────────────────────── #
def run_evaluation(model, dataloader, device):
    all_preds = []
    all_labels = []
    
    print("⏳ Đang chạy suy luận trên tập dữ liệu đánh giá...")
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            
    return np.array(all_labels), np.array(all_preds)

# ─────────────────────────────────────────── #
#  TÍNH MA TRẬN NHẦM LẪN VÀ CHỈ SỐ
# ─────────────────────────────────────────── #
def calculate_confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def compute_metrics(cm, class_names):
    report = {}
    num_classes = len(class_names)
    
    total_samples = cm.sum()
    total_correct = np.diag(cm).sum()
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        support = int(cm[i, :].sum())
        
        report[name] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support
        }
    
    # Tính các trung bình
    macro_precision = np.mean([info["precision"] for info in report.values()])
    macro_recall = np.mean([info["recall"] for info in report.values()])
    macro_f1 = np.mean([info["f1-score"] for info in report.values()])
    
    weighted_precision = sum(info["precision"] * info["support"] for info in report.values()) / total_samples
    weighted_recall = sum(info["recall"] * info["support"] for info in report.values()) / total_samples
    weighted_f1 = sum(info["f1-score"] * info["support"] for info in report.values()) / total_samples
    
    summary = {
        "accuracy": overall_accuracy,
        "macro_avg": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1-score": macro_f1
        },
        "weighted_avg": {
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1-score": weighted_f1
        }
    }
    
    return report, summary

# ─────────────────────────────────────────── #
#  VẼ MA TRẬN NHẦM LẪN SANG TRỌNG
# ─────────────────────────────────────────── #
def plot_confusion_matrix(cm, class_names, accuracy, save_path=OUTPUT_IMG):
    num_classes = len(class_names)
    
    # Thiết lập style và màu sắc
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(13, 11), facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    
    # Vẽ ma trận màu sắc
    # Sử dụng colormap Blues thanh lịch
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    
    # Thêm thanh màu sắc (color bar)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#e6edf3", labelsize=10)
    cbar.outline.set_edgecolor("#30363d")
    
    # Thiết lập các trục tọa độ
    tick_marks = np.arange(num_classes)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", color="#e6edf3", fontsize=11, fontweight="semibold")
    ax.set_yticklabels(class_names, color="#e6edf3", fontsize=11, fontweight="semibold")
    
    # Tiêu đề và nhãn trục
    ax.set_title(f"WEATHER CLASSIFICATION CONFUSION MATRIX\nOverall Accuracy: {accuracy*100:.2f}%", 
                 color="white", fontsize=16, pad=20, fontweight="bold")
    ax.set_ylabel("True Label (Nhãn thực tế)", color="#8b949e", fontsize=13, labelpad=15)
    ax.set_xlabel("Predicted Label (Nhãn dự đoán)", color="#8b949e", fontsize=13, labelpad=15)
    
    # Vẽ viền và ẩn grid lines mặc định
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
        spine.set_linewidth(1.5)
    ax.grid(False)
    
    # Điền giá trị số và tỉ lệ phần trăm vào từng ô
    # Ngưỡng màu để chọn màu chữ (nền tối -> chữ trắng, nền sáng -> chữ đen)
    thresh = cm.max() / 2.0
    
    for i in range(num_classes):
        row_total = cm[i, :].sum()
        for j in range(num_classes):
            val = cm[i, j]
            # Tính phần trăm của class thực tế được dự đoán thành class này
            pct = (val / row_total * 100) if row_total > 0 else 0
            
            # Text hiển thị: Số lượng bên trên, % bên dưới
            text_str = f"{val}\n({pct:.1f}%)" if val > 0 else "0"
            
            # Chọn màu chữ tương phản
            text_color = "white" if val > thresh else "#8b949e"
            if val == 0:
                text_color = "#30363d" # Màu mờ cho ô có giá trị 0
                
            ax.text(j, i, text_str,
                    ha="center", va="center",
                    color=text_color,
                    fontsize=10 if val > 0 else 9,
                    fontweight="bold" if val > 0 else "normal")
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"📊 Đã vẽ và lưu ma trận nhầm lẫn tại: {save_path}")

# ─────────────────────────────────────────── #
#  XUẤT BÁO CÁO ĐÁNH GIÁ CHI TIẾT
# ─────────────────────────────────────────── #
def save_reports(report, summary, class_names, cm, eval_dir):
    # 1. Xuất file Markdown
    md_content = []
    md_content.append("# BÁO CÁO ĐÁNH GIÁ MÔ HÌNH NHẬN DIỆN THỜI TIẾT")
    md_content.append(f"- **Thư mục dữ liệu đánh giá**: `{eval_dir}`")
    md_content.append(f"- **Tổng số lớp**: {len(class_names)}")
    md_content.append(f"- **Độ chính xác tổng thể (Overall Accuracy)**: **{summary['accuracy']*100:.2f}%**")
    md_content.append("")
    
    md_content.append("## 1. Chi tiết các chỉ số theo từng lớp (Class-wise Metrics)")
    md_content.append("| Lớp thời tiết | Precision | Recall | F1-Score | Số lượng mẫu (Support) |")
    md_content.append("| :--- | :---: | :---: | :---: | :---: |")
    
    for cls in class_names:
        info = report[cls]
        md_content.append(f"| **{cls}** | {info['precision']*100:.2f}% | {info['recall']*100:.2f}% | {info['f1-score']*100:.2f}% | {info['support']} |")
        
    md_content.append(f"| **Trung bình Macro (Macro Avg)** | {summary['macro_avg']['precision']*100:.2f}% | {summary['macro_avg']['recall']*100:.2f}% | {summary['macro_avg']['f1-score']*100:.2f}% | {cm.sum()} |")
    md_content.append(f"| **Trung bình Trọng số (Weighted Avg)** | {summary['weighted_avg']['precision']*100:.2f}% | {summary['weighted_avg']['recall']*100:.2f}% | {summary['weighted_avg']['f1-score']*100:.2f}% | {cm.sum()} |")
    md_content.append("")
    
    md_content.append("## 2. Phân tích chi tiết ma trận nhầm lẫn (Confusion Matrix Analysis)")
    md_content.append("Dưới đây là một số lớp dễ bị nhầm lẫn với nhau nhất:")
    
    # Phân tích các cặp hay bị nhầm lẫn nhất (không tính đường chéo chính)
    confusions = []
    num_classes = len(class_names)
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                confusions.append((class_names[i], class_names[j], cm[i, j]))
                
    confusions.sort(key=lambda x: x[2], reverse=True)
    
    top_n = min(5, len(confusions))
    for k in range(top_n):
        true_cls, pred_cls, count = confusions[k]
        pct = (count / cm[class_names.index(true_cls), :].sum() * 100)
        md_content.append(f"- Thực tế là **{true_cls}** bị đoán nhầm thành **{pred_cls}**: **{count} lần** ({pct:.1f}% của lớp thực tế)")
        
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))
    print(f"📄 Đã lưu báo cáo dạng Markdown tại: {OUTPUT_REPORT}")
    
    # 2. Xuất console report dạng đẹp
    print("\n" + "=" * 70)
    print(f"{'BÁO CÁO ĐÁNH GIÁ MÔ HÌNH':^70}")
    print("=" * 70)
    print(f"Độ chính xác tổng thể (Accuracy): {summary['accuracy']*100:.2f}%")
    print("-" * 70)
    print(f"{'Lớp':<15} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10} | {'Mẫu':>6}")
    print("-" * 70)
    for cls in class_names:
        info = report[cls]
        print(f"{cls:<15} | {info['precision']*100:9.2f}% | {info['recall']*100:9.2f}% | {info['f1-score']*100:9.2f}% | {info['support']:6d}")
    print("-" * 70)
    print(f"{'Macro Avg':<15} | {summary['macro_avg']['precision']*100:9.2f}% | {summary['macro_avg']['recall']*100:9.2f}% | {summary['macro_avg']['f1-score']*100:9.2f}% | {cm.sum():6d}")
    print(f"{'Weighted Avg':<15} | {summary['weighted_avg']['precision']*100:9.2f}% | {summary['weighted_avg']['recall']*100:9.2f}% | {summary['weighted_avg']['f1-score']*100:9.2f}% | {cm.sum():6d}")
    print("=" * 70)

# ─────────────────────────────────────────── #
#  HÀM MAIN KHỞI CHẠY
# ─────────────────────────────────────────── #
def main():
    print("=" * 60)
    print("  ĐÁNH GIÁ MÔ HÌNH VÀ TẠO CONFUSION MATRIX")
    print("=" * 60)
    
    # 1. Phát hiện thiết bị
    device = setup_device()
    
    # 2. Đọc nhãn và cấu hình từ JSON
    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(f"❌ Không tìm thấy file {JSON_PATH}. Vui lòng chạy export_labels.py trước!")
        
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    class_names = config["class_names"]
    img_size = config.get("img_size", 260)
    num_classes = len(class_names)
    
    # 3. Chọn thư mục đánh giá (Thử TEST_DIR trước, nếu không có thử VAL_DIR)
    eval_dir = TEST_DIR
    if not os.path.exists(eval_dir) or not any(os.path.isdir(Path(eval_dir)/c) for c in class_names):
        print(f"⚠️ Không tìm thấy tập test tại {TEST_DIR}, chuyển sang dùng tập val tại {VAL_DIR}...")
        eval_dir = VAL_DIR
        
    if not os.path.exists(eval_dir) or not any(os.path.isdir(Path(eval_dir)/c) for c in class_names):
        raise FileNotFoundError(f"❌ Không tìm thấy thư mục dữ liệu đánh giá hợp lệ tại {TEST_DIR} hoặc {VAL_DIR}!")
        
    print(f"📁 Dữ liệu đánh giá: {eval_dir}")
    
    # 4. Xây dựng DataLoader
    transform = get_eval_transforms(img_size)
    dataset = WeatherDataset(eval_dir, class_names, img_size, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"📊 Tìm thấy {len(dataset)} ảnh trong {len(class_names)} lớp thời tiết.")
    
    # 5. Tải mô hình
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Không tìm thấy file mô hình: {MODEL_PATH}")
    model = load_model(MODEL_PATH, num_classes, device)
    
    # 6. Chạy suy luận
    y_true, y_pred = run_evaluation(model, dataloader, device)
    
    # 7. Tính toán ma trận nhầm lẫn và các chỉ số
    cm = calculate_confusion_matrix(y_true, y_pred, num_classes)
    report, summary = compute_metrics(cm, class_names)
    
    # 8. Vẽ ma trận nhầm lẫn
    plot_confusion_matrix(cm, class_names, summary["accuracy"], OUTPUT_IMG)
    
    # 9. Lưu báo cáo và in kết quả
    save_reports(report, summary, class_names, cm, eval_dir)
    print(f"\n🎉 Hoàn thành đánh giá mô hình!")
    print(f"🖼️ Ma trận nhầm lẫn: {OUTPUT_IMG}")
    print(f"📄 Báo cáo chi tiết: {OUTPUT_REPORT}")
    print("=" * 60)

if __name__ == "__main__":
    main()
