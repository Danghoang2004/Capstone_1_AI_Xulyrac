from collections import defaultdict
from app.config import (
    CONF_THRESHOLD, 
    WASTE_VERIFY_MAPPING_VIETNAMESE, 
    WASTE_CLASSIFICATION_MAPPING_VIETNAMESE
)


def filter_valid_detections(detections):
    return [d for d in detections if d["confidence"] >= CONF_THRESHOLD]


def translate_class_name(class_name: str, use_classification: bool = False) -> str:
    """
    Dịch tên class thô sang tiếng Việt
    
    Args:
        class_name: Tên class gốc
        use_classification: 
            - False: dùng mapping cho xác minh (train5)
            - True: dùng mapping cho phân loại (trainphanloai)
    """
    if use_classification:
        return WASTE_CLASSIFICATION_MAPPING_VIETNAMESE.get(class_name, class_name)
    else:
        return WASTE_VERIFY_MAPPING_VIETNAMESE.get(class_name, class_name)


def calculate_type_percentage(valid_detections):
    type_conf_sum = defaultdict(float)

    for d in valid_detections:
        type_conf_sum[d["class_name"]] += d["confidence"]

    total_conf = sum(type_conf_sum.values())
    if total_conf == 0:
        return {}

    return {
        k: round(v / total_conf * 100, 2)
        for k, v in type_conf_sum.items()
    }


def calculate_overall_confidence(valid_detections):
    if not valid_detections:
        return 0.0

    return round(
        sum(d["confidence"] for d in valid_detections)
        / len(valid_detections),
        2
    )
