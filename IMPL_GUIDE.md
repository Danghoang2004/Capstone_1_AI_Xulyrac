## HƯỚNG DẪN SỬ DỤNG CHỨC NĂNG PHÂN LOẠI RÁC THẢI

### 📋 Tóm tắt những gì đã được thêm:

#### 1. **Tệp cấu hình mới: `app/config.py`**
   - Định nghĩa mapping từ tên class gốc sang tiếng Việt
   - Quản lý đường dẫn model
   - Cấu hình threshold

```python
WASTE_CLASS_MAPPING_VIETNAMESE = {
    "bi_rac": "Túi rác",
    "chai_nhua": "Chai nhựa",
    "dong_rac": "Thùng rác",
    "giay_hopgiay": "Giấy/Hộp giấy",
    "rac_detvai": "Rác theo vải",
    "rac_kimloai": "Rác kim loại",
    "rac_thai_nhua": "Rác thải nhựa",
    "rac_thuytinh": "Rác thủy tinh",
    "tuinilong_rac": "Túi nilon rác",
}
```

#### 2. **Endpoint mới: `/ai/classify-waste` (POST)**

**Cách sử dụng:**
```bash
curl -X POST http://127.0.0.1:8000/ai/classify-waste \
  -F "file=@/path/to/image.jpg"
```

**Response:**
```json
{
  "success": true,
  "data": {
    "is_trash": true,
    "overall_confidence": 0.85,
    "total_objects_detected": 2,
    "waste_types": {
      "Chai nhựa": 60.5,
      "Rác thải nhựa": 39.5
    },
    "detections": [
      {
        "class_name_vietnamese": "Chai nhựa",
        "class_name_raw": "chai_nhua",
        "confidence": 0.92,
        "bbox": [100, 120, 250, 350]
      },
      {
        "class_name_vietnamese": "Rác thải nhựa",
        "class_name_raw": "rac_thai_nhua",
        "confidence": 0.78,
        "bbox": [300, 200, 450, 400]
      }
    ],
    "output_image": "outputs/analyzed/uuid_classified.jpg"
  }
}
```

#### 3. **Các hàm mới trong `yolo_service.py`**

**`run_waste_classification(image_path: str)`**
- Chạy model `trainphanloai` để phân loại rác
- Trả về kết quả với Vietnamese labels
- Tính type_percentage cho từng loại rác

**`draw_classification_boxes(image_path, detections, output_path)`**
- Vẽ bounding boxes lên ảnh với Vietnamese labels
- Hiển thị confidence score

#### 4. **Cập nhật các module khác**

**`app/utils.py`**
- Thêm hàm `translate_class_name()` để dịch tên class

**`app/schemas.py`**
- Thêm `ClassificationDetection` model
- Thêm `ClassificationResponse` model

**`app/main.py`**
- Thêm endpoint `/ai/classify-waste`

---

### 🚀 Cách chạy:

1. **Chuẩn bị môi trường:**
```bash
# Kích hoạt virtual environment
d:\AI_XuLyBaoCaoRac\model_process\.venv\Scripts\Activate.ps1

# Cài đặt dependencies (nếu chưa có)
pip install fastapi uvicorn ultralytics opencv-python pydantic
```

2. **Chạy server:**
```bash
python -m app.main
# Hoặc
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

3. **Test API:**
- Dùng Postman hoặc curl để gửi ảnh
- Hoặc sử dụng file `test_main.http` trong VS Code

---

### 📸 Flow hoạt động:

```
[Người dùng gửi ảnh]
        ↓
[POST /ai/classify-waste]
        ↓
[Model trainphanloai nhận diện]
        ↓
[Dịch tên vào tiếng Việt]
        ↓
[Vẽ boxes lên ảnh]
        ↓
[Trả về ảnh + danh sách rác phân loại]
```

---

### 🎯 Đặc điểm chính:

✅ **Vietnamese Labels**: Tất cả nhãn được hiển thị tiếng Việt  
✅ **Confidence Score**: Hiển thị độ chính xác của mỗi phát hiện  
✅ **Waste Summary**: Tóm tắt % các loại rác thải  
✅ **Visual Output**: Trả về ảnh đã vẽ boxes  
✅ **Detailed Data**: Trả về toàn bộ thông tin chi tiết về mỗi object

---

### 🔧 Nếu muốn tùy chỉnh:

1. **Thay đổi Vietnamese labels**: Chỉnh sửa dictionary trong `app/config.py`
2. **Thay đổi confidence threshold**: Cập nhật `CONF_THRESHOLD` trong `app/config.py`
3. **Thay đổi model**: Cập nhật đường dẫn trong `MODEL_CLASSIFY_PATH`
4. **Thêm loại rác mới**: Cập nhật mapping trong `WASTE_CLASS_MAPPING_VIETNAMESE`

---

### ⚠️ Lưu ý:

- Model `trainphanloai` phải được train trước khi sử dụng
- File weights phải tồn tại tại: `models/trainphanloai/weights/best.pt`
- Đường dẫn model phải đúng trong `config.py`, hoặc thay đổi thành đường dẫn tuyệt đối
