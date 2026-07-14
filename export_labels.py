"""
export_labels.py
================
Mục đích: Phân nhãn và chia dữ liệu tự động
  - Đọc thư mục 'dataset/' gốc chứa các class (dew, fog, frost, ...)
  - Chia tự động 80% Train / 20% Test vào 'data/train' và 'data/test'
  - Xuất file 'class_names.json' chứa thông tin nhãn
"""

import os
import json
import shutil
import random

# ==================== CẤU HÌNH ====================
SOURCE_DIR  = "dataset"          # Thư mục dataset gốc
OUTPUT_DIR  = "data"             # Thư mục đầu ra
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42                 # Seed để tái tạo được kết quả


TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR   = os.path.join(OUTPUT_DIR, "val")
TEST_DIR  = os.path.join(OUTPUT_DIR, "test")
JSON_PATH = "class_names.json"

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# ===================================================


def get_class_folders(source_dir):
    """Trả về danh sách các thư mục lớp hợp lệ."""
    if not os.path.isdir(source_dir):
        print(f"❌ Không tìm thấy thư mục nguồn: '{source_dir}'")
        return []
    classes = sorted([
        d for d in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, d))
    ])
    return classes


def split_and_copy(source_dir, train_dir, val_dir, test_dir,
                   classes, train_ratio, val_ratio, seed):

    random.seed(seed)
    stats = {}

    for cls in classes:
        cls_src = os.path.join(source_dir, cls)

        images = [
            f for f in os.listdir(cls_src)
            if os.path.splitext(f)[1].lower() in VALID_EXT
        ]

        random.shuffle(images)

        total = len(images)

        n_train = int(total * train_ratio)
        n_val   = int(total * val_ratio)

        train_imgs = images[:n_train]
        val_imgs   = images[n_train:n_train+n_val]
        test_imgs  = images[n_train+n_val:]

        splits = [
            (train_dir, train_imgs),
            (val_dir, val_imgs),
            (test_dir, test_imgs)
        ]

        for base_dir, imgs in splits:
            dest_dir = os.path.join(base_dir, cls)
            os.makedirs(dest_dir, exist_ok=True)

            for img in imgs:
                shutil.copy2(
                    os.path.join(cls_src, img),
                    os.path.join(dest_dir, img)
                )

        stats[cls] = {
            "total": total,
            "train": len(train_imgs),
            "val": len(val_imgs),
            "test": len(test_imgs)
        }

        print(
            f"  ✅ {cls:<15}"
            f" | Tổng: {total:>4}"
            f" | Train: {len(train_imgs):>4}"
            f" | Val: {len(val_imgs):>4}"
            f" | Test: {len(test_imgs):>4}"
        )

    return stats


def export_json(classes, stats, json_path):
    """Xuất file JSON chứa thông tin nhãn."""
    class_indices = {cls: i for i, cls in enumerate(classes)}
    data = {
        "dataset_name": SOURCE_DIR,
        "total_classes": len(classes),
        "class_names": classes,
        "class_indices": class_indices,
        "split_stats": stats
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"\n📄 Đã lưu file JSON: {json_path}")


def main():
    print("=" * 55)
    print("  PHÂN NHÃN & CHIA DỮ LIỆU THỜI TIẾT TỰ ĐỘNG")
    print("=" * 55)

    classes = get_class_folders(SOURCE_DIR)
    if not classes:
        return

    print(f"\n🔍 Tìm thấy {len(classes)} lớp: {classes}")

    # Hỏi người dùng nếu thư mục data/ đã tồn tại
    if os.path.isdir(OUTPUT_DIR):
        ans = input(f"\n⚠️  Thư mục '{OUTPUT_DIR}/' đã tồn tại. Xóa và chia lại? (y/n): ").strip().lower()
        if ans != "y":
            print("↩️  Hủy thao tác. Giữ nguyên dữ liệu cũ.")
            return
        shutil.rmtree(OUTPUT_DIR)
        print(f"🗑️  Đã xóa thư mục '{OUTPUT_DIR}/' cũ.\n")

    print(f"\n📂 Đang chia dữ liệu ({int(TRAIN_RATIO*100)}% Train / {int(VAL_RATIO*100)}% Val / {int(TEST_RATIO*100)}% Test)...")
    stats = split_and_copy(SOURCE_DIR,TRAIN_DIR,VAL_DIR,TEST_DIR,classes,TRAIN_RATIO,VAL_RATIO,RANDOM_SEED)
    export_json(classes, stats, JSON_PATH)

    total_imgs = sum(v["total"] for v in stats.values())
    total_train = sum(v["train"] for v in stats.values())
    total_val   = sum(v["val"] for v in stats.values())
    total_test  = sum(v["test"] for v in stats.values())

    print("\n" + "=" * 55)
    print("  ✔️ HOÀN TẤT!")
    print(f"  📊 Tổng ảnh : {total_imgs}")
    print(f"  🚂 Train    : {total_train} → data/train/")
    print(f"  🔍 Val      : {total_val} → data/val/")
    print(f"  🧪 Test     : {total_test} → data/test/")
    print("=" * 55)


if __name__ == "__main__":
    main()
