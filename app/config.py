"""
Configuration and mapping for waste classification
"""

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
# PHẦN 2: PHÂN LOẠI RÁC THẢI (trainphanloai)
# ============================================
WASTE_CLASSIFICATION_MAPPING_VIETNAMESE = {
    "Bia_carton": "Vỏ hộp bia/Bìa carton",
    "ChaiNhua": "Chai nhựa",
    "Chai_thuytinh": "Chai thủy tinh",
    "HopXop": "Hộp xốp",
    "LoThuyTinh": "Lõ thủy tinh",
    "LonKimLoai": "Lon kim loại",
    "LyGiay": "Ly giấy",
    "TuiNiLong": "Túi nilon",
    "ao": "Áo",
    "quan": "Quần",
}

# Model paths
MODEL_DETECT_PATH = r"D:\AI_XuLyBaoCaoRac\model_process\models\train5\weights\best.pt"
MODEL_CLASSIFY_PATH = r"D:\AI_XuLyBaoCaoRac\model_process\models\trainphanloai\weights\best.pt"

# YAML data files
YAML_DETECT_PATH = r"mydata.yaml"  # Cho xác minh rác
YAML_CLASSIFY_PATH = r"mydata_phanloai.yaml"  # Cho phân loại rác

# Confidence threshold
CONF_THRESHOLD = 0.2
