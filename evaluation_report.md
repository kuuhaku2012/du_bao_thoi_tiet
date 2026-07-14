# BÁO CÁO ĐÁNH GIÁ MÔ HÌNH NHẬN DIỆN THỜI TIẾT
- **Thư mục dữ liệu đánh giá**: `data\test`
- **Tổng số lớp**: 11
- **Độ chính xác tổng thể (Overall Accuracy)**: **93.85%**

## 1. Chi tiết các chỉ số theo từng lớp (Class-wise Metrics)
| Lớp thời tiết | Precision | Recall | F1-Score | Số lượng mẫu (Support) |
| :--- | :---: | :---: | :---: | :---: |
| **dew** | 96.33% | 99.06% | 97.67% | 106 |
| **fogsmog** | 96.88% | 96.12% | 96.50% | 129 |
| **frost** | 82.28% | 90.28% | 86.09% | 72 |
| **glaze** | 91.30% | 86.60% | 88.89% | 97 |
| **hail** | 98.89% | 98.89% | 98.89% | 90 |
| **lightning** | 100.00% | 100.00% | 100.00% | 58 |
| **rain** | 100.00% | 90.00% | 94.74% | 80 |
| **rainbow** | 100.00% | 100.00% | 100.00% | 36 |
| **rime** | 89.71% | 90.23% | 89.97% | 174 |
| **sandstorm** | 93.58% | 97.14% | 95.33% | 105 |
| **snow** | 91.40% | 90.43% | 90.91% | 94 |
| **Trung bình Macro (Macro Avg)** | 94.58% | 94.43% | 94.45% | 1041 |
| **Trung bình Trọng số (Weighted Avg)** | 93.96% | 93.85% | 93.86% | 1041 |

## 2. Phân tích chi tiết ma trận nhầm lẫn (Confusion Matrix Analysis)
Dưới đây là một số lớp dễ bị nhầm lẫn với nhau nhất:
- Thực tế là **rime** bị đoán nhầm thành **frost**: **9 lần** (5.2% của lớp thực tế)
- Thực tế là **glaze** bị đoán nhầm thành **rime**: **7 lần** (7.2% của lớp thực tế)
- Thực tế là **snow** bị đoán nhầm thành **rime**: **6 lần** (6.4% của lớp thực tế)
- Thực tế là **fogsmog** bị đoán nhầm thành **sandstorm**: **4 lần** (3.1% của lớp thực tế)
- Thực tế là **frost** bị đoán nhầm thành **rime**: **4 lần** (5.6% của lớp thực tế)