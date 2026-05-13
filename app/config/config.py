
# ============================================
# PHẦN 1: XÁC MINH RÁC THẢI (train5)
# ============================================
WASTE_VERIFY_MAPPING_VIETNAMESE = {
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

# ============================================
# PHẦN 2: PHÂN LOẠI RÁC THẢI (dùng train5)
# ============================================
WASTE_CLASSIFICATION_MAPPING_VIETNAMESE = {
    "Bia_carton": "Vỏ hộp bia/Bìa carton",
    "ChaiNhua": "Chai nhựa",
    "Chai_thuytinh": "Chai thủy tinh",
    "HopXop": "Hộp xốp",
    "LoThuyTinh": "Lọ thủy tinh",
    "LonKimLoai": "Lon kim loại",
    "LyGiay": "Ly giấy",
    "TuiNiLong": "Túi nilon",
    "ao": "Áo",
    "quan": "Quần",
}

import os
from pathlib import Path

# ============================================
# Model paths - đọc từ environment hoặc dùng mặc định
# ============================================
# Lấy thư mục hiện tại của project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Cách 1: Đọc từ env vars (dùng khi deploy trên cloud)
# Cách 2: Dùng đường dẫn tương đối từ repo
MODEL_DETECT_PATH = os.getenv(
    "MODEL_DETECT_PATH",
    str(BASE_DIR / "models" / "train5" / "weights" / "best.pt")
)
MODEL_CLASSIFY_PATH = os.getenv(
    "MODEL_CLASSIFY_PATH",
    str(BASE_DIR / "models" / "train5" / "weights" / "best.pt")
)

# Model mặc định nhẹ từ ultralytics (dùng nếu không có custom model)
# Cách dùng: set env MODEL_USE_DEFAULT=true để dùng yolov8n thay vì custom
USE_DEFAULT_MODEL = os.getenv("MODEL_USE_DEFAULT", "false").lower() == "true"
if USE_DEFAULT_MODEL:
    MODEL_DETECT_PATH = "yolov8n.pt"
    MODEL_CLASSIFY_PATH = "yolov8n.pt"

MODEL_COCO_PATH = "yolov8n.pt"

# YAML data files - dùng đường dẫn tương đối
YAML_DETECT_PATH = str(BASE_DIR / "mydata.yaml")
YAML_CLASSIFY_PATH = str(BASE_DIR / "mydata.yaml")

# Confidence threshold
CONF_THRESHOLD = 0.6
