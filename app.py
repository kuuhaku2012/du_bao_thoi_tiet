import os
import sys
import json
import threading
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageTk, ImageDraw, ImageFilter
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox

# ─────────────────────────────────────────────
#  Hằng số & metadata
# ─────────────────────────────────────────────
MODEL_PATH = "weather_model.pth"
JSON_PATH  = "class_names.json"
DEFAULT_IMG_SIZE = 260

WEATHER_META = {
    "dew":       {"emoji": "💧", "vi": "Sương móc",  "color": "#38bdf8"},
    "fogsmog":   {"emoji": "🌫️", "vi": "Sương mù",   "color": "#94a3b8"},
    "frost":     {"emoji": "🌨️", "vi": "Giá lạnh",   "color": "#bae6fd"},
    "glaze":     {"emoji": "🧊", "vi": "Băng tráng", "color": "#67e8f9"},
    "hail":      {"emoji": "⛈️", "vi": "Mưa đá",     "color": "#7dd3fc"},
    "lightning": {"emoji": "⚡",  "vi": "Sét",         "color": "#fbbf24"},
    "rain":      {"emoji": "🌧️", "vi": "Mưa",        "color": "#60a5fa"},
    "rainbow":   {"emoji": "🌈", "vi": "Cầu vồng",   "color": "#a78bfa"},
    "rime":      {"emoji": "❄️", "vi": "Sương giá",  "color": "#e0f2fe"},
    "sandstorm": {"emoji": "🌪️", "vi": "Bão cát",    "color": "#fcd34d"},
    "snow":      {"emoji": "🌨️", "vi": "Tuyết",      "color": "#f0f9ff"},
}

# Màu sắc giao diện
COLORS = {
    "bg":           "#0d1117",
    "surface":      "#161b22",
    "surface2":     "#21262d",
    "border":       "#30363d",
    "accent":       "#7c3aed",
    "accent2":      "#3b82f6",
    "accent3":      "#10b981",
    "text":         "#e6edf3",
    "text_muted":   "#8b949e",
    "text_dim":     "#6e7681",
    "success":      "#238636",
    "warning":      "#d29922",
    "error":        "#da3633",
    "gradient_1":   "#7c3aed",
    "gradient_2":   "#3b82f6",
}


# ─────────────────────────────────────────────
#  Tải tài nguyên (chạy ngoài main thread)
# ─────────────────────────────────────────────
def get_resource_path(relative_path: str) -> str:
    """Trả về đường dẫn đúng khi chạy EXE (PyInstaller)."""
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)


def load_resources():
    model_path = get_resource_path(MODEL_PATH)
    json_path  = get_resource_path(JSON_PATH)

    if not os.path.exists(model_path):
        return None, None, DEFAULT_IMG_SIZE, "Không tìm thấy file mô hình: weather_model.pth"

    img_size    = DEFAULT_IMG_SIZE
    class_names = sorted(WEATHER_META.keys())

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            class_names = raw.get("class_names", class_names)
            img_size    = raw.get("img_size", DEFAULT_IMG_SIZE)
        else:
            class_names = raw

    num_classes = len(class_names)
    base = models.efficientnet_v2_s(weights=None)
    in_features = base.classifier[1].in_features
    base.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes),
    )

    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    base.load_state_dict(ckpt["state_dict"])
    base.eval()

    return base, class_names, img_size, None


def preprocess(img: Image.Image, img_size: int) -> torch.Tensor:
    tf_img = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    return tf_img(img.convert("RGB")).unsqueeze(0)


# ─────────────────────────────────────────────
#  Widget: Animated progress bar (Canvas)
# ─────────────────────────────────────────────
class AnimatedBar(tk.Canvas):
    def __init__(self, master, width=300, height=10, color="#7c3aed", bg="#21262d", **kwargs):
        super().__init__(master, width=width, height=height,
                         bg=bg, highlightthickness=0, **kwargs)
        self._color  = color
        self._width  = width
        self._height = height
        self._target = 0.0
        self._current = 0.0
        self._bar_id = self.create_rectangle(0, 0, 0, height,
                                              fill=color, outline="")
        self._anim_running = False

    def set_value(self, pct: float):
        """pct: 0.0 → 1.0"""
        self._target = max(0.0, min(1.0, pct))
        if not self._anim_running:
            self._anim_running = True
            self._animate()

    def _animate(self):
        delta = self._target - self._current
        if abs(delta) < 0.003:
            self._current = self._target
            self._anim_running = False
        else:
            self._current += delta * 0.18
            self.after(16, self._animate)
        fill_w = int(self._current * self._width)
        self.coords(self._bar_id, 0, 0, fill_w, self._height)


# ─────────────────────────────────────────────
#  Main Application
# ─────────────────────────────────────────────
class WeatherApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Thiết lập chủ đề CustomTkinter
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("🌦️  Nhận Diện Thời Tiết AI")
        self.geometry("1080x720")
        self.minsize(900, 620)
        self.configure(fg_color=COLORS["bg"])

        # Trạng thái
        self.model        = None
        self.class_names  = None
        self.img_size     = DEFAULT_IMG_SIZE
        self.current_pil  = None   # ảnh gốc PIL
        self.displayed_pil = None  # ảnh đang hiển thị PIL (có thể có bounding box)
        self.bar_widgets  = {}     # label → AnimatedBar
        self.pct_labels   = {}     # label → StringVar

        self._build_ui()
        self._load_model_async()

    # ──────────────────────────────────────────
    #  Xây dựng UI
    # ──────────────────────────────────────────
    def _build_ui(self):
        # ── Header ──────────────────────────────
        header = ctk.CTkFrame(self, fg_color=COLORS["surface"], corner_radius=0, height=72)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        ctk.CTkLabel(
            header,
            text="🌦️  Nhận Diện Thời Tiết AI",
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
            text_color=COLORS["text"],
        ).pack(side="left", padx=24, pady=0)

        self.status_badge = ctk.CTkLabel(
            header,
            text="⏳ Đang tải mô hình...",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["warning"],
            fg_color=COLORS["surface2"],
            corner_radius=8,
            padx=12, pady=4,
        )
        self.status_badge.pack(side="right", padx=24)

        # ── Nội dung chính ───────────────────────
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=20, pady=16)
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        # Panel trái: upload ảnh
        self._build_left_panel(content)

        # Panel phải: kết quả
        self._build_right_panel(content)

        # ── Footer ──────────────────────────────
        footer = ctk.CTkFrame(self, fg_color=COLORS["surface"], corner_radius=0, height=36)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)
        ctk.CTkLabel(
            footer,
            text="EfficientNetV2S  •  PyTorch  •  11 lớp thời tiết",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_dim"],
        ).pack(side="left", padx=20)
        ctk.CTkLabel(
            footer,
            text="Deep Learning Project 2026",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_dim"],
        ).pack(side="right", padx=20)

    def _build_left_panel(self, parent):
        frame = ctk.CTkFrame(
            parent,
            fg_color=COLORS["surface"],
            corner_radius=16,
            border_width=1,
            border_color=COLORS["border"],
        )
        frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=0)
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)

        # Tiêu đề panel
        ctk.CTkLabel(
            frame,
            text="📂  Tải Ảnh Lên",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLORS["text"],
        ).grid(row=0, column=0, sticky="w", padx=20, pady=(16, 8))

        # Vùng hiển thị ảnh (canvas)
        self.canvas = tk.Canvas(
            frame,
            bg=COLORS["surface2"],
            highlightthickness=1,
            highlightbackground=COLORS["border"],
        )
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 12))
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Placeholder text trên canvas
        self._draw_placeholder()

        # Nút Upload
        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.grid(row=2, column=0, padx=16, pady=(0, 16), sticky="ew")
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        ctk.CTkButton(
            btn_frame,
            text="📁  Chọn Ảnh",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=COLORS["accent"],
            hover_color="#6d28d9",
            corner_radius=10,
            height=40,
            command=self._open_file,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 6))

        ctk.CTkButton(
            btn_frame,
            text="🗑️  Xóa",
            font=ctk.CTkFont(size=13),
            fg_color=COLORS["surface2"],
            hover_color=COLORS["error"],
            border_width=1,
            border_color=COLORS["border"],
            corner_radius=10,
            height=40,
            command=self._clear_image,
        ).grid(row=0, column=1, sticky="ew", padx=(6, 0))

        # Tên file
        self.filename_label = ctk.CTkLabel(
            frame,
            text="Chưa chọn ảnh",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_muted"],
        )
        self.filename_label.grid(row=3, column=0, padx=16, pady=(0, 12))

    def _build_right_panel(self, parent):
        frame = ctk.CTkFrame(
            parent,
            fg_color=COLORS["surface"],
            corner_radius=16,
            border_width=1,
            border_color=COLORS["border"],
        )
        frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=0)
        frame.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            frame,
            text="🔍  Kết Quả Phân Tích",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLORS["text"],
        ).grid(row=0, column=0, sticky="w", padx=20, pady=(16, 0))

        # Kết quả chính (emoji + nhãn + độ tin cậy)
        result_card = ctk.CTkFrame(
            frame,
            fg_color=COLORS["surface2"],
            corner_radius=14,
            border_width=1,
            border_color=COLORS["border"],
        )
        result_card.grid(row=1, column=0, sticky="ew", padx=16, pady=(12, 0))
        result_card.columnconfigure(0, weight=1)

        self.emoji_label = ctk.CTkLabel(
            result_card,
            text="—",
            font=ctk.CTkFont(size=52),
        )
        self.emoji_label.grid(row=0, column=0, pady=(18, 0))

        self.result_label = ctk.CTkLabel(
            result_card,
            text="Chưa có kết quả",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=COLORS["text"],
        )
        self.result_label.grid(row=1, column=0, pady=(4, 0))

        self.conf_label = ctk.CTkLabel(
            result_card,
            text="Tải ảnh để bắt đầu nhận diện",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["text_muted"],
        )
        self.conf_label.grid(row=2, column=0, pady=(4, 18))

        # Phân phối xác suất Top-5
        ctk.CTkLabel(
            frame,
            text="📊  Phân Phối Xác Suất (Top 5)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS["text_muted"],
        ).grid(row=2, column=0, sticky="w", padx=20, pady=(16, 4))

        self.bars_frame = ctk.CTkFrame(frame, fg_color="transparent")
        self.bars_frame.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 0))
        self.bars_frame.columnconfigure(1, weight=1)

        # Khởi tạo 5 hàng thanh bar (placeholder)
        self.bar_rows = []
        for i in range(5):
            name_var = tk.StringVar(value="—")
            pct_var  = tk.StringVar(value="")

            lbl = ctk.CTkLabel(
                self.bars_frame,
                textvariable=name_var,
                font=ctk.CTkFont(size=12),
                text_color=COLORS["text_muted"],
                width=110,
                anchor="e",
            )
            lbl.grid(row=i, column=0, sticky="e", padx=(0, 8), pady=4)

            bar = AnimatedBar(
                self.bars_frame,
                width=200, height=8,
                color=COLORS["accent2"],
                bg=COLORS["surface"],
            )
            bar.grid(row=i, column=1, sticky="ew", pady=4)

            pct_lbl = ctk.CTkLabel(
                self.bars_frame,
                textvariable=pct_var,
                font=ctk.CTkFont(size=11),
                text_color=COLORS["text_dim"],
                width=50,
                anchor="w",
            )
            pct_lbl.grid(row=i, column=2, sticky="w", padx=(8, 0), pady=4)

            self.bar_rows.append((name_var, bar, pct_var))

        # Nút phân tích
        self.analyze_btn = ctk.CTkButton(
            frame,
            text="🔎  Phân Tích Ngay",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=COLORS["accent2"],
            hover_color="#2563eb",
            corner_radius=10,
            height=44,
            state="disabled",
            command=self._run_inference,
        )
        self.analyze_btn.grid(row=4, column=0, sticky="ew", padx=16, pady=(16, 0))

        self.warn_label = ctk.CTkLabel(
            frame,
            text="",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["warning"],
        )
        self.warn_label.grid(row=5, column=0, pady=(6, 16))

    # ──────────────────────────────────────────
    #  Placeholder trên canvas
    # ──────────────────────────────────────────
    def _draw_placeholder(self):
        self.canvas.delete("all")
        w = self.canvas.winfo_width()  or 400
        h = self.canvas.winfo_height() or 380
        cx, cy = w // 2, h // 2

        # Icon upload
        self.canvas.create_text(
            cx, cy - 30,
            text="📷",
            font=("Segoe UI Emoji", 40),
            fill=COLORS["text_dim"],
        )
        self.canvas.create_text(
            cx, cy + 30,
            text="Nhấn 'Chọn Ảnh' hoặc kéo thả ảnh vào đây",
            font=("Segoe UI", 12),
            fill=COLORS["text_muted"],
        )
        self.canvas.create_text(
            cx, cy + 55,
            text="Hỗ trợ: JPG, PNG, WEBP, BMP",
            font=("Segoe UI", 10),
            fill=COLORS["text_dim"],
        )

    def _on_canvas_resize(self, event):
        if self.displayed_pil is None:
            self._draw_placeholder()
        else:
            self._display_image(self.displayed_pil)

    # ──────────────────────────────────────────
    #  Mở file ảnh
    # ──────────────────────────────────────────
    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh thời tiết",
            filetypes=[
                ("Ảnh", "*.jpg *.jpeg *.png *.webp *.bmp *.tiff"),
                ("Tất cả", "*.*"),
            ]
        )
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB")
            self.current_pil = img
            self.displayed_pil = img
            self._display_image(img)
            fname = os.path.basename(path)
            if len(fname) > 40:
                fname = fname[:37] + "..."
            self.filename_label.configure(text=f"📄 {fname}")
            if self.model is not None:
                self.analyze_btn.configure(state="normal")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể mở ảnh:\n{e}")

    def _clear_image(self):
        self.current_pil = None
        self.displayed_pil = None
        self._draw_placeholder()
        self.filename_label.configure(text="Chưa chọn ảnh")
        self.analyze_btn.configure(state="disabled")
        self._reset_results()

    def _display_image(self, pil_img: Image.Image):
        self.canvas.delete("all")
        w = max(self.canvas.winfo_width(),  100)
        h = max(self.canvas.winfo_height(), 100)

        # Fit ảnh vào canvas
        img_w, img_h = pil_img.size
        ratio = min(w / img_w, h / img_h)
        new_w, new_h = int(img_w * ratio), int(img_h * ratio)
        resized = pil_img.resize((new_w, new_h), Image.LANCZOS)

        self._tk_img = ImageTk.PhotoImage(resized)
        x, y = (w - new_w) // 2, (h - new_h) // 2
        self.canvas.create_image(x, y, anchor="nw", image=self._tk_img)

    # ──────────────────────────────────────────
    #  Tải mô hình (background thread)
    # ──────────────────────────────────────────
    def _load_model_async(self):
        def _worker():
            model, class_names, img_size, err = load_resources()
            self.after(0, lambda: self._on_model_loaded(model, class_names, img_size, err))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_model_loaded(self, model, class_names, img_size, err):
        if err:
            self.status_badge.configure(text=f"❌ {err}", text_color=COLORS["error"])
            messagebox.showerror("Lỗi tải mô hình", err)
            return

        self.model       = model
        self.class_names = class_names
        self.img_size    = img_size

        self.status_badge.configure(
            text=f"✅ Mô hình sẵn sàng  ({len(class_names)} lớp)",
            text_color=COLORS["accent3"],
        )
        if self.current_pil is not None:
            self.analyze_btn.configure(state="normal")

    # ──────────────────────────────────────────
    #  Chạy suy luận
    # ──────────────────────────────────────────
    def _run_inference(self):
        if self.current_pil is None or self.model is None:
            return

        self.analyze_btn.configure(state="disabled", text="⏳ Đang phân tích...")
        self.warn_label.configure(text="")

        def _worker():
            tensor = preprocess(self.current_pil, self.img_size)
            marked_img = self.current_pil.copy()
            
            try:
                # Grad-CAM requires gradient calculation, which is disabled in torch.no_grad().
                # We run it with enable_grad.
                with torch.enable_grad():
                    # Set inputs to track gradients
                    input_tensor = tensor.clone().detach()
                    input_tensor.requires_grad = True
                    
                    self.model.eval()
                    target_layer = self.model.features[-1]
                    
                    activations = []
                    gradients = []
                    
                    def forward_hook(module, inp, out):
                        activations.append(out)
                        
                    def backward_hook(module, grad_in, grad_out):
                        gradients.append(grad_out[0])
                        
                    h_f = target_layer.register_forward_hook(forward_hook)
                    h_b = target_layer.register_full_backward_hook(backward_hook)
                    
                    # Forward pass
                    logits = self.model(input_tensor)
                    preds = torch.softmax(logits, dim=1)[0].detach().numpy()
                    
                    # Compute gradients for predicted class
                    class_idx = int(np.argmax(preds))
                    score = logits[0, class_idx]
                    
                    self.model.zero_grad()
                    score.backward()
                    
                    # Remove hooks immediately
                    h_f.remove()
                    h_b.remove()
                    
                    if len(activations) > 0 and len(gradients) > 0:
                        act = activations[0].detach()
                        grad = gradients[0].detach()
                        
                        # Grad-CAM math
                        weights = torch.mean(grad, dim=(2, 3), keepdim=True)
                        cam = torch.sum(weights * act, dim=1, keepdim=True)
                        cam = torch.clamp(cam, min=0)  # ReLU
                        
                        # Normalization
                        cam_min = cam.min()
                        cam_max = cam.max()
                        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
                        cam_np = cam[0, 0].cpu().numpy()
                        
                        # Bounding box extraction
                        threshold = 0.5
                        mask = cam_np > threshold
                        y_indices, x_indices = np.where(mask)
                        
                        if len(x_indices) > 0 and len(y_indices) > 0:
                            h_grid, w_grid = cam_np.shape
                            img_w, img_h = marked_img.size
                            
                            x_min_grid = np.min(x_indices)
                            x_max_grid = np.max(x_indices)
                            y_min_grid = np.min(y_indices)
                            y_max_grid = np.max(y_indices)
                            
                            x_min = int((x_min_grid / w_grid) * img_w)
                            x_max = int(((x_max_grid + 1) / w_grid) * img_w)
                            y_min = int((y_min_grid / h_grid) * img_h)
                            y_max = int(((y_max_grid + 1) / h_grid) * img_h)
                            
                            # Bound checks
                            x_min = max(0, min(x_min, img_w - 1))
                            x_max = max(0, min(x_max, img_w))
                            y_min = max(0, min(y_min, img_h - 1))
                            y_max = max(0, min(y_max, img_h))
                            
                            # Draw bounding box
                            draw = ImageDraw.Draw(marked_img)
                            top_cls = self.class_names[class_idx]
                            meta = WEATHER_META.get(top_cls, {"emoji": "🌡️", "vi": top_cls, "color": COLORS["accent"]})
                            box_color = meta.get("color", COLORS["accent"])
                            
                            # Draw outline
                            draw.rectangle([x_min, y_min, x_max, y_max], outline=box_color, width=4)
                            
                            # Draw text label banner
                            from PIL import ImageFont
                            try:
                                font = ImageFont.truetype("arial.ttf", 12)
                            except:
                                font = ImageFont.load_default()
                                
                            label_text = f" Vung quyet dinh ({meta['vi']})"
                            banner_height = 18
                            banner_y_min = max(0, y_min - banner_height)
                            banner_y_max = y_min
                            
                            # Draw label backdrop
                            draw.rectangle([x_min, banner_y_min, x_max, banner_y_max], fill=box_color)
                            # Draw text
                            draw.text((x_min + 4, banner_y_min + 2), label_text, fill="#ffffff", font=font)
                
                # Show results with bounding box image
                self.after(0, lambda: self._show_results(preds, marked_img))
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.after(0, lambda: messagebox.showerror("Lỗi phân tích", f"Lỗi trong quá trình tính toán Grad-CAM:\n{e}"))
                self.after(0, lambda: self.analyze_btn.configure(state="normal", text="🔎  Phân Tích Ngay"))

        threading.Thread(target=_worker, daemon=True).start()

    def _show_results(self, preds: np.ndarray, marked_img: Image.Image = None):
        top_idx  = int(np.argmax(preds))
        top_cls  = self.class_names[top_idx]
        top_conf = float(preds[top_idx])
        meta     = WEATHER_META.get(top_cls, {"emoji": "🌡️", "vi": top_cls, "color": COLORS["accent"]})

        # Update display image with bounding box
        if marked_img is not None:
            self.displayed_pil = marked_img
            self._display_image(marked_img)

        # Kết quả chính
        self.emoji_label.configure(text=meta["emoji"])
        self.result_label.configure(text=meta["vi"], text_color=meta["color"])
        self.conf_label.configure(
            text=f"Độ tin cậy: {top_conf*100:.1f}%",
            text_color=COLORS["accent3"] if top_conf >= 0.7
                        else (COLORS["warning"] if top_conf >= 0.5 else COLORS["error"]),
        )

        # Top-5 bars
        top5_idx = np.argsort(preds)[::-1][:5]
        bar_colors = [meta["color"], COLORS["accent2"], COLORS["accent"],
                      COLORS["accent3"], "#f59e0b"]

        for i, idx in enumerate(top5_idx):
            cls   = self.class_names[idx]
            prob  = float(preds[idx])
            m     = WEATHER_META.get(cls, {"emoji": "🌡️", "vi": cls})
            display = f"{m['emoji']} {m['vi']}"

            name_var, bar, pct_var = self.bar_rows[i]
            name_var.set(display)
            pct_var.set(f"{prob*100:.1f}%")
            bar.configure(bg=COLORS["surface2"])
            bar._color = bar_colors[i % len(bar_colors)]
            bar._bar_id and bar.itemconfig(bar._bar_id, fill=bar._color)
            bar.set_value(prob)

        # Cảnh báo
        if top_conf < 0.5:
            self.warn_label.configure(
                text="⚠️ Độ tin cậy thấp — ảnh có thể không thuộc dataset đã học.",
                text_color=COLORS["warning"],
            )
        else:
            self.warn_label.configure(text="")

        self.analyze_btn.configure(state="normal", text="🔎  Phân Tích Ngay")

    def _reset_results(self):
        self.emoji_label.configure(text="—")
        self.result_label.configure(text="Chưa có kết quả", text_color=COLORS["text"])
        self.conf_label.configure(
            text="Tải ảnh để bắt đầu nhận diện",
            text_color=COLORS["text_muted"],
        )
        self.warn_label.configure(text="")
        for name_var, bar, pct_var in self.bar_rows:
            name_var.set("—")
            pct_var.set("")
            bar.set_value(0.0)
        
        # Reset display image back to original if present
        if self.current_pil is not None:
            self.displayed_pil = self.current_pil
            self._display_image(self.current_pil)


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = WeatherApp()
    app.mainloop()
