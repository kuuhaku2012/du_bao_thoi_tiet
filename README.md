# 🌦️ WeatherVision AI — Nhận Diện Và Phân Tích Thời Tiết Đa Nguồn

Hệ thống trí tuệ nhân tạo nhận diện **11 hiện tượng thời tiết từ ảnh**, giải thích vùng ảnh ảnh hưởng đến quyết định bằng **Grad-CAM**, đồng thời tra cứu thời tiết theo vị trí và dự báo nguy cơ mưa trong **3, 6 và 12 giờ tiếp theo**.

Dự án sử dụng **EfficientNetV2-S, PyTorch, OpenCV, Streamlit và Open-Meteo API**.

## 👤 Người thực hiện

| Họ và tên | MSSV |
|---|---|
| **Ngô Long Thiên** | **2001230920** |

## 🚀 Tính năng chính

### 🧠 Nhận diện hiện tượng thời tiết

- Nhận diện 11 lớp: dew, fog/smog, frost, glaze, hail, lightning, rain, rainbow, rime, sandstorm và snow
- Hiển thị **Top 3 dự đoán** cùng xác suất
- Cho phép điều chỉnh **ngưỡng tin cậy**
- Cảnh báo khi ảnh nằm ngoài miền dữ liệu hoặc kết quả có độ tin cậy thấp
- Tự động sử dụng GPU CUDA khi có sẵn

### 🔍 Giải thích mô hình

- Sinh bản đồ nhiệt **Grad-CAM**
- Hiển thị vùng ảnh ảnh hưởng mạnh nhất đến quyết định
- Hỗ trợ kiểm tra mô hình có tập trung đúng vào hiện tượng thời tiết hay không

### 🌍 Dữ liệu thời tiết theo vị trí

- Tìm thành phố bằng tên tiếng Việt hoặc tiếng Anh
- Hiển thị nhiệt độ, cảm giác thực, độ ẩm, gió và lượng mưa hiện tại
- Dự báo xác suất mưa sau **3 giờ, 6 giờ và 12 giờ**
- Biểu đồ xác suất mưa trong 12 giờ tiếp theo
- Bảng dự báo thời tiết 7 ngày
- Đối chiếu tham khảo giữa kết quả AI từ ảnh và thời tiết trực tuyến

### 📊 Huấn luyện và đánh giá

- Transfer Learning với EfficientNetV2-S
- Huấn luyện hai giai đoạn: đóng băng backbone và fine-tune toàn mạng
- Data augmentation, AdamW, label smoothing và cosine scheduler
- Đánh giá Accuracy, Precision, Recall, F1-score và Confusion Matrix
- Export model sang ONNX
- GitHub Actions kiểm tra cú pháp và unit test

## 📈 Kết quả mô hình

| Chỉ số | Kết quả |
|---|---:|
| Accuracy | **93,85%** |
| Macro Precision | **94,58%** |
| Macro Recall | **94,43%** |
| Macro F1-score | **94,45%** |
| Weighted F1-score | **93,86%** |
| Số ảnh test | **1.041** |
| Số lớp | **11** |

<p align="center">
  <img src="docs/training_charts.png" width="900" alt="Biểu đồ huấn luyện WeatherVision AI">
</p>

<p align="center">
  <img src="docs/confusion_matrix.png" width="760" alt="Ma trận nhầm lẫn WeatherVision AI">
</p>

## 🛠️ Công nghệ sử dụng

| Thành phần | Công nghệ |
|---|---|
| Ngôn ngữ | Python |
| Deep Learning | PyTorch, Torchvision |
| Mô hình | EfficientNetV2-S, Transfer Learning |
| Giải thích mô hình | Grad-CAM |
| Xử lý ảnh | OpenCV, Pillow |
| Giao diện | Streamlit |
| Weather API | Open-Meteo |
| Đánh giá | Scikit-learn, Matplotlib |
| Tối ưu triển khai | ONNX |
| Kiểm thử | Pytest, GitHub Actions |

## ⚙️ Cài đặt

### Yêu cầu

- Python 3.10 hoặc mới hơn
- Git
- GPU NVIDIA hỗ trợ CUDA là tùy chọn

### 1. Clone repository

```bash
git clone https://github.com/Thienshinn1608/WeatherVision-AI.git
cd WeatherVision-AI
```

### 2. Tạo môi trường ảo

Windows CMD:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

### 3. Cài thư viện

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Chuyển model từ repo cũ

Đặt repo mới và repo cũ cạnh nhau:

```text
projects/
├── du_bao_thoi_tiet/
└── WeatherVision-AI/
```

Trong `WeatherVision-AI`, chạy:

```bash
python scripts/migrate_from_old_repo.py --source ../du_bao_thoi_tiet
```

Model sẽ được chép tới:

```text
models/weather_model.pth
```

### 5. Chạy ứng dụng

```bash
streamlit run app.py
```

Mở địa chỉ:

```text
http://localhost:8501
```

## 🏋️ Huấn luyện lại mô hình

Dataset dùng cấu trúc `ImageFolder`:

```text
data/
├── train/
│   ├── dew/
│   ├── fogsmog/
│   └── ...
├── val/
└── test/
```

Huấn luyện:

```bash
python train.py --data data --batch-size 16
```

Đánh giá:

```bash
python evaluate.py --data data/test
```

Export ONNX:

```bash
python export_onnx.py
```

## 📁 Cấu trúc thư mục

```text
WeatherVision-AI/
├── app.py                              # Dashboard Streamlit
├── train.py                            # Huấn luyện hai giai đoạn
├── evaluate.py                         # Đánh giá mô hình
├── export_onnx.py                      # Export ONNX
├── class_names.json                    # Danh sách 11 lớp
├── requirements.txt
├── src/weathervision/
│   ├── config.py                       # Metadata và cấu hình
│   ├── model.py                        # Kiến trúc EfficientNetV2-S
│   ├── predictor.py                    # Pipeline suy luận
│   ├── gradcam.py                      # Giải thích Grad-CAM
│   └── weather_api.py                  # Open-Meteo API
├── scripts/
│   └── migrate_from_old_repo.py
├── models/
│   ├── weather_model.pth               # Chép từ repo cũ
│   └── README.md
├── docs/
│   ├── training_charts.png
│   ├── confusion_matrix.png
│   └── evaluation_report.md
├── tests/
│   └── test_weather_codes.py
└── .github/workflows/quality.yml
```

## 👨‍💻 Nội dung thực hiện

- Fine-tune EfficientNetV2-S để phân loại 11 hiện tượng thời tiết
- Xây dựng pipeline tiền xử lý và suy luận ảnh
- Triển khai Top-3 prediction và kiểm tra ngưỡng tin cậy
- Tích hợp Grad-CAM để giải thích quyết định của mô hình
- Xây dựng dashboard Streamlit
- Tích hợp Open-Meteo để lấy thời tiết theo vị trí
- Xây dựng dự báo nguy cơ mưa 3/6/12 giờ
- Đánh giá bằng Accuracy, Precision, Recall, F1-score và Confusion Matrix
- Chuẩn hóa cấu trúc dự án, kiểm thử và export ONNX

## ⚠️ Giới hạn

- Mô hình chỉ nhận diện các lớp đã xuất hiện trong tập huấn luyện
- Kết quả có thể giảm khi ảnh mờ, thiếu sáng hoặc chứa nhiều hiện tượng cùng lúc
- Grad-CAM chỉ giải thích vùng chú ý, không phải bounding box đối tượng chính xác
- Dự báo theo vị trí phụ thuộc dữ liệu từ Open-Meteo và kết nối Internet
- Kết quả chỉ phục vụ học tập, nghiên cứu và tham khảo

## 📄 License

Dự án được phát hành theo giấy phép MIT.

**Người thực hiện: Ngô Long Thiên — MSSV 2001230920**
