# AI Viet Nam — Face Re-Identification System

Hệ thống nhận diện khuôn mặt thời gian thực kết hợp **Desktop GUI**, **REST API**, **PostgreSQL** và **React Dashboard**.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green)
![React](https://img.shields.io/badge/React-18-61DAFB)
![Docker](https://img.shields.io/badge/Docker-required-2496ED)

---

## Kiến trúc hệ thống

```
┌──────────────┐   WebSocket    ┌──────────────────┐
│  GUI (PyQt6) │ ─────────────► │  API (FastAPI)    │
│  Camera Feed │ ◄───────────── │  SCRFD + ArcFace  │
└──────────────┘   bbox/names   │  FAISS Search     │
                                └────────┬─────────┘
                                         │ asyncpg
                                ┌────────▼─────────┐
                                │   PostgreSQL DB   │
                                │  (Docker)         │
                                └────────┬─────────┘
                                         │ REST API
                                ┌────────▼─────────┐
                                │  React Dashboard  │
                                │  localhost:5173    │
                                └──────────────────┘
```

---

## Yêu cầu hệ thống

| Thành phần | Phiên bản |
|---|---|
| Python | 3.10+ |
| Node.js | 18+ |
| Docker Desktop | mới nhất |
| GPU (tuỳ chọn) | CUDA 11.8+ |

---

## Cài đặt

### 1. Clone repository

```bash
git clone git@github.com:NguyenDinhTiem/face-reidentification.git
cd face-reidentification
```

### 2. Tải model weights

Tải về và đặt vào thư mục `weights/`:

| Model | Link | Dung lượng |
|---|---|---|
| SCRFD 10G (detection) | [det_10g.onnx](https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_10g.onnx) | 16.1 MB |
| SCRFD 500M (nhẹ) | [det_500m.onnx](https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_500m.onnx) | 2.4 MB |
| ArcFace MobileFace | [w600k_mbf.onnx](https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/w600k_mbf.onnx) | 13 MB |
| ArcFace ResNet-50 | [w600k_r50.onnx](https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/w600k_r50.onnx) | 166 MB |

```bash
# Linux/Mac — tải tự động
sh download.sh
```

### 3. Thêm ảnh khuôn mặt cần nhận diện

Đặt ảnh khuôn mặt vào thư mục `assets/faces/`. **Tên file = tên người**.

```
assets/faces/
├── NguyenVanA.jpg
├── TranThiB.jpg
└── ...
```

> Mỗi người chỉ cần 1 ảnh, chụp rõ mặt, ánh sáng tốt.

### 4. Khởi động PostgreSQL (Docker)

```bash
docker compose up -d
```

Kiểm tra DB đã sẵn sàng:
```bash
docker compose ps
```

### 5. Cài Python dependencies

```bash
pip install -r requirements.txt
```

> **GPU:** Đổi `onnxruntime-gpu` thay `onnxruntime` trong `requirements.txt` nếu có CUDA.
>
> **Windows + Conda:** Nên cài bằng `python -m pip install -r requirements.txt` sau khi `conda activate <env>`, tránh dùng `pip --user`.
>
> **Quan trọng:** Nếu Python đang ưu tiên package trong `C:\Users\<user>\AppData\Roaming\Python\...` thì `onnxruntime-gpu` trong conda env có thể bị che bởi bản `onnxruntime` CPU, khiến model chỉ chạy bằng CPU dù máy có CUDA.
>
> Có thể chặn hiện tượng này bằng cách:
> ```bash
> conda env config vars set -n <env> PYTHONNOUSERSITE=1
> conda activate <env>
> python -m pip install -r requirements.txt
> python -c "import onnxruntime as ort; print(ort.__file__); print(ort.get_available_providers())"
> ```
>
> Nếu cài mới trên Windows, nên dùng Python 3.11 cho môi trường dự án để giảm rủi ro lệch package.

### 6. Chạy API backend

```bash
python api.py
```

API khởi động tại `http://localhost:8000`. Swagger docs: `http://localhost:8000/docs`

### 7. Chạy GUI desktop

```bash
python gui.py
```

- Nhấn **▶ Start** để bật camera và bắt đầu nhận diện
- Nhấn **⚙ Settings** để đổi model / nguồn camera / ngưỡng

### 8. Chạy React Dashboard (tuỳ chọn)

```bash
cd web
npm install
npm run dev
```

Mở trình duyệt: `http://localhost:5173`

---

## Cấu trúc dự án

```
face-reidentification/
├── api.py                 # FastAPI backend (REST + WebSocket)
├── gui.py                 # Desktop GUI (PyQt6)
├── db.py                  # PostgreSQL helpers (asyncpg)
├── main.py                # CLI entry point (không dùng GUI)
│
├── models/                # SCRFD, ArcFace model wrappers
├── database/              # FAISS database implementation
├── utils/                 # Logging, helpers
│
├── weights/               # Model weights (.onnx) — không upload lên git
├── assets/
│   ├── faces/             # Ảnh khuôn mặt cần nhận diện
│   └── captures/          # Ảnh crop tự động lưu — không upload lên git
│
├── web/                   # React dashboard (Vite)
│   ├── src/
│   │   ├── pages/         # Dashboard, Attendance, Unknowns, Settings
│   │   ├── api.js         # API service layer
│   │   └── App.jsx
│   └── package.json
│
├── init.sql               # PostgreSQL schema
├── docker-compose.yml     # PostgreSQL container
└── requirements.txt
```

---

## Thông tin kết nối PostgreSQL

| Tham số | Giá trị mặc định |
|---|---|
| Host | `localhost` |
| Port | `5432` |
| Database | `faceid_db` |
| Username | `faceid_user` |
| Password | `faceid_pass` |

Đổi thông tin kết nối qua biến môi trường:
```bash
DATABASE_URL=postgresql://user:pass@host:5432/dbname python api.py
```

---

## API Endpoints

| Method | Endpoint | Mô tả |
|---|---|---|
| `GET` | `/api/settings` | Lấy cấu hình hiện tại |
| `POST` | `/api/settings` | Cập nhật cấu hình |
| `POST` | `/api/infer/start` | Bật inference |
| `POST` | `/api/infer/stop` | Tắt inference |
| `GET` | `/api/attendance` | Lịch sử điểm danh |
| `GET` | `/api/unknowns` | Log người lạ |
| `GET` | `/api/stats` | Thống kê tổng quan |
| `POST` | `/api/attendance/log` | Ghi log điểm danh (multipart) |
| `POST` | `/api/unknown/log` | Ghi log người lạ (multipart) |
| `WS` | `/ws/infer` | WebSocket nhận diện realtime |

---

## Troubleshooting

**Camera không mở được:**
- Kiểm tra nguồn camera trong Settings (0 = webcam mặc định, hoặc dùng URL RTSP)

**Lỗi ONNX Runtime / CUDA:**
- Dùng `onnxruntime` (CPU) nếu không có GPU hoặc CUDA chưa cài đúng
- Xem log lỗi tại `app.log`
- Nếu log không báo lỗi nhưng model vẫn chỉ chạy CPU, kiểm tra runtime đang import từ đâu:
```bash
python -c "import onnxruntime as ort; print(ort.__file__); print(ort.get_available_providers())"
```
- Nếu kết quả nằm trong `AppData\Roaming\Python\...` thay vì conda env, môi trường đang bị lẫn `user-site package`
- Cách xử lý ổn định nhất là tạo env mới, bật `PYTHONNOUSERSITE=1`, rồi cài lại dependency bằng `python -m pip`

**Không kết nối được PostgreSQL:**
- Chắc chắn Docker đang chạy: `docker compose up -d`
- Kiểm tra: `docker compose ps`

**Web không hiển thị ảnh:**
- Đảm bảo Vite dev server đang chạy (`npm run dev` trong thư mục `web/`)
- Ảnh được serve qua proxy `/captures` → FastAPI

---

## References

- [SCRFD: Efficient Face Detection](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)
- [ArcFace: Deep Face Recognition](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)
- [YOLOFace training guide](https://drive.google.com/drive/folders/1Df3xxfUsWDbMfqwTgOE7q2CeXakW4V8D?usp=sharing)
- [FAISS: Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)
