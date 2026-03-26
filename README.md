# AI Xử Lý Báo Cáo Rác - Capstone 1

## 1. Cách chạy dự án bằng câu lệnh

### Yêu cầu
- Python 3.10.x
- Windows PowerShell (hoặc terminal tương đương)

### Chạy nhanh trên máy hiện tại
```powershell
cd D:\project_Capstone_1_full\model_process\model_process

# Nếu đã có venv
..\.venv\Scripts\Activate.ps1

# Chạy API
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Chạy từ đầu trên máy mới
```powershell
# 1) Vào thư mục dự án
cd D:\project_Capstone_1_full\model_process\model_process

# 2) Tạo virtual environment
python -m venv ..\.venv

# 3) Kích hoạt môi trường
..\.venv\Scripts\Activate.ps1

# 4) Cài thư viện
pip install -r requirements.txt

# 5) Chạy API
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

API sau khi chạy sẽ có địa chỉ:
- `http://127.0.0.1:8000`
- `http://127.0.0.1:8000/docs`

## 2. Cài thư viện để chạy trên máy khác

### Cách khuyến nghị
Sử dụng file `requirements.txt` đã được thêm trong dự án:

```powershell
pip install -r requirements.txt
```

### Danh sách thư viện cần có
Nếu cần xem nhanh các thư viện chính trước khi cài, dự án đang dùng:

- fastapi==0.135.2
- uvicorn[standard]==0.42.0
- python-multipart==0.0.22
- pydantic==2.12.5
- ultralytics==8.4.27
- opencv-python==4.13.0.92
- numpy==2.2.6
- scipy==1.15.3
- matplotlib==3.10.8
- PyYAML==6.0.3
- torch==2.11.0
- torchvision==0.26.0

Bạn chỉ cần chạy đúng 1 lệnh là cài toàn bộ:

```powershell
pip install -r requirements.txt
```

### Nếu cần đúng 100% môi trường máy hiện tại
Bạn có thể xuất full package đang dùng rồi cài lại ở máy khác:

```powershell
pip freeze > requirements-lock.txt
pip install -r requirements-lock.txt
```

## 3. Lưu ý quan trọng khi mang sang máy khác

Trong file `app/config/config.py`, đường dẫn model hiện đang để dạng tuyệt đối (ví dụ ổ `D:`). Ở máy mới bạn cần chỉnh lại cho đúng đường dẫn thực tế:
- `MODEL_DETECT_PATH`
- `MODEL_CLASSIFY_PATH`

Nếu không chỉnh, server vẫn chạy nhưng các API AI sẽ lỗi khi load model.
