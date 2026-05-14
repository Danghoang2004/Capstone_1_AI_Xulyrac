
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
# Model paths
# ============================================
MODEL_DETECT_PATH = os.getenv("MODEL_DETECT_PATH", r"models/train5/weights/best.pt")
MODEL_CLASSIFY_PATH = os.getenv("MODEL_CLASSIFY_PATH", r"models/train5/weights/best.pt")
MODEL_COCO_PATH = os.getenv("MODEL_COCO_PATH", "yolov8n.pt")

# YAML data files
YAML_DETECT_PATH = os.getenv("YAML_DETECT_PATH", r"mydata.yaml")  # Cho xác minh rác
YAML_CLASSIFY_PATH = os.getenv("YAML_CLASSIFY_PATH", r"mydata.yaml")  # Cho phân loại rác (dùng train5)

# Confidence threshold
CONF_THRESHOLD = 0.6
