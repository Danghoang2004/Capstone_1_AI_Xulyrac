import os
import cv2
from typing import Dict, List, Tuple
from ultralytics import YOLO
from app.service.yolo_service.utils import (
    filter_valid_detections,
    calculate_type_percentage,
    calculate_overall_confidence,
    translate_class_name
)
from app.service.yolo_service.coco_mapping import translate_coco_class_name
from app.config.config import MODEL_DETECT_PATH, MODEL_CLASSIFY_PATH, MODEL_COCO_PATH

# Load detection model (train5) for waste verification
detection_model = YOLO(MODEL_DETECT_PATH)

# Load classification model (train5) for waste classification
classification_model = YOLO(MODEL_CLASSIFY_PATH)

# Load COCO pretrained model for context validation
coco_context_model = YOLO(MODEL_COCO_PATH)

GENERAL_OBJECT_CONTEXT_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
}

WASTE_TYPE_RISK_SCORE = {
    "PAPER": 0.30,
    "ORGANIC": 0.50,
    "PLASTIC": 0.60,
    "METAL": 0.65,
    "GLASS": 0.75,
    "MIXED_WASTE": 0.80,
    "UNKNOWN": 0.40,
}

POLLUTION_LEVEL_TEMPLATES = {
    "LOW": "Hình ảnh cho thấy có một lượng nhỏ rác. Mức độ ô nhiễm thấp và có thể xử lý bằng việc thu gom đơn giản.",
    "MEDIUM": "Hình ảnh cho thấy có rác xuất hiện rõ trong khu vực. Mức độ ô nhiễm trung bình và nên được theo dõi hoặc dọn dẹp sớm.",
    "HIGH": "Hình ảnh cho thấy có nhiều vật thể rác nằm rải rác trong khu vực. Rác chiếm diện tích đáng kể và có thể cần được ưu tiên dọn dẹp.",
    "CRITICAL": "Hình ảnh cho thấy lượng rác lớn hoặc khu vực ô nhiễm nặng. Vị trí này nên được ưu tiên xử lý khẩn cấp.",
    "UNCONFIRMED": "AI phát hiện vật thể có thể liên quan đến rác hoặc tái chế, nhưng ảnh chưa thể hiện rõ đây là rác bị bỏ đi. Người dùng nên chụp lại ảnh có bối cảnh rõ hơn hoặc xác nhận trước khi gửi báo cáo.",
}

RECOMMENDATION_TEMPLATES = {
    "PLASTIC": "Rác nhựa nên được phân loại riêng và đưa đến điểm thu gom tái chế nếu có thể.",
    "PAPER": "Rác giấy nên được giữ khô và phân loại riêng để tái chế.",
    "METAL": "Rác kim loại nên được thu gom riêng và chuyển đến cơ sở tái chế phù hợp.",
    "GLASS": "Rác thủy tinh cần được xử lý cẩn thận và phân loại riêng để tránh gây nguy hiểm.",
    "ORGANIC": "Rác hữu cơ có thể được phân loại để ủ phân hoặc xử lý đúng cách.",
    "MIXED_WASTE": "Rác hỗn hợp nên được phân loại thành nhóm có thể tái chế và không thể tái chế trước khi xử lý.",
    "UNKNOWN": "Ảnh chưa đủ rõ để đưa ra gợi ý xử lý chính xác. Vui lòng chụp lại ảnh rõ hơn, đủ sáng và có bối cảnh xung quanh.",
}


def _translate_coco_detections_to_vietnamese(coco_detections: List[Dict]) -> List[str]:
    """Dịch danh sách COCO detected items sang tiếng Việt"""
    return [translate_coco_class_name(d["class_name"]) for d in coco_detections]


def _is_waste_related_class(class_name: str) -> bool:
    normalized = class_name.lower()
    keywords = (
        "rac",
        "rác",
        "nhua",
        "giay",
        "paper",
        "metal",
        "kimloai",
        "thuytinh",
        "glass",
        "tuinilong",
        "bi_rac",
        "dong_rac",
        "bin_rac",
    )
    return any(keyword in normalized for keyword in keywords)


def _is_low_context_false_positive(object_count: int, waste_area_ratio: float, image_quality_score: float) -> Tuple[bool, str]:
    """
    Dataset hiện tại chỉ có lớp rác, nên model không thể gọi tên vật dụng không phải rác.
    Hàm này chỉ đánh dấu ảnh là thiếu ngữ cảnh để xác nhận rác, thay vì khẳng định đó là vật gì.
    """
    if object_count == 1 and waste_area_ratio < 0.08 and image_quality_score < 0.6:
        return True, "Ảnh chỉ có một vật thể nhỏ, ngữ cảnh chưa đủ để xác nhận là rác."
    if object_count == 1 and waste_area_ratio < 0.05:
        return True, "Vật thể quá nhỏ trong ảnh, có thể là nhầm lẫn với đồ vật thông thường."
    return False, ""


def _run_coco_context_detection(image_path: str) -> List[Dict]:
    results = coco_context_model(image_path)[0]
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = results.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        detections.append({
            "class_id": cls_id,
            "class_name": cls_name,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2],
        })

    return filter_valid_detections(detections)


def _detect_non_waste_context(coco_detections: List[Dict]) -> Tuple[bool, str]:
    """
    Dùng COCO pretrained model để bắt các đồ vật phổ biến không phải rác.
    Nếu phát hiện các vật dụng rõ ràng trong bối cảnh sử dụng, coi ảnh chưa đủ điều kiện xác nhận rác.
    """
    if not coco_detections:
        return False, ""

    matched_items = [
        d["class_name"]
        for d in coco_detections
        if d["class_name"] in GENERAL_OBJECT_CONTEXT_CLASSES and d["confidence"] >= 0.45
    ]
    if matched_items:
        return True, f"COCO pretrained model phát hiện vật dụng trong bối cảnh sử dụng: {', '.join(matched_items)}"

    return False, ""


def _resolve_final_waste_decision(
    waste_score: float,
    has_general_object_context: bool,
    object_count: int,
    waste_area_ratio: float,
) -> Tuple[str, str, bool]:
    """
    Hợp nhất Model A và Model B thành quyết định cuối cùng.

    Returns:
        Tuple[ai_decision, report_status, need_manual_review]
    """
    if waste_score >= 0.80:
        if has_general_object_context and (object_count <= 1 or waste_area_ratio < 0.20):
            return "UNCERTAIN", "NEED_REVIEW", True
        return "WASTE_DETECTED", "AI_VERIFIED", False

    if waste_score >= 0.50:
        return "UNCERTAIN", "NEED_REVIEW", True

    if has_general_object_context:
        return "NOT_WASTE", "REQUEST_REUPLOAD", False

    return "NOT_WASTE", "REQUEST_REUPLOAD", False


def _calculate_image_quality_score(image_path: str) -> Tuple[float, str | None]:
    image = cv2.imread(image_path)
    if image is None:
        return 0.0, "Không thể đọc ảnh để phân tích."

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = gray.mean()

    normalized_blur = min(blur_score / 250.0, 1.0)
    normalized_brightness = min(max(brightness / 140.0, 0.0), 1.0)
    quality_score = round((normalized_blur * 0.6) + (normalized_brightness * 0.4), 2)

    if quality_score < 0.35:
        return quality_score, "Ảnh quá mờ, tối hoặc chưa đủ rõ để phân tích."
    return quality_score, None


def _calculate_bbox_area_ratio(bbox: List[int], image_width: int, image_height: int) -> float:
    x1, y1, x2, y2 = bbox
    width = max(x2 - x1, 0)
    height = max(y2 - y1, 0)
    image_area = max(image_width * image_height, 1)
    return (width * height) / image_area


def _calculate_object_count_score(object_count: int) -> float:
    if object_count <= 0:
        return 0.0
    if object_count == 1:
        return 0.25
    if object_count <= 3:
        return 0.60
    if object_count <= 6:
        return 0.80
    return 1.00


def _calculate_waste_area_score(waste_area_ratio: float) -> float:
    if waste_area_ratio < 0.05:
        return 0.10
    if waste_area_ratio <= 0.15:
        return 0.35
    if waste_area_ratio <= 0.30:
        return 0.60
    if waste_area_ratio <= 0.50:
        return 0.80
    return 1.00


def _calculate_waste_diversity_score(waste_types: List[str]) -> float:
    diversity = len(set(waste_types))
    if diversity <= 1:
        return 0.30
    if diversity == 2:
        return 0.60
    return 1.00


def _calculate_scene_context_score(object_count: int, waste_area_ratio: float, image_quality_score: float) -> float:
    if object_count == 1 and waste_area_ratio < 0.08:
        return 0.20 if image_quality_score >= 0.5 else 0.10
    if object_count <= 2 and waste_area_ratio < 0.18:
        return 0.35
    if object_count <= 4 and waste_area_ratio < 0.35:
        return 0.60
    if object_count <= 6 or waste_area_ratio < 0.50:
        return 0.80
    return 0.95


def _calculate_discarded_sign_score(object_count: int, waste_area_ratio: float, average_confidence: float) -> float:
    if object_count == 0:
        return 0.0
    if object_count == 1 and waste_area_ratio < 0.08 and average_confidence >= 0.85:
        return 0.40
    if waste_area_ratio < 0.18:
        return 0.55
    if waste_area_ratio < 0.35:
        return 0.80
    return 1.00


def _map_waste_type(class_names: List[str]) -> str:
    mapped_types = []
    for class_name in class_names:
        normalized = class_name.lower()
        if any(keyword in normalized for keyword in ("nhua", "tuinilong", "plastic")):
            mapped_types.append("PLASTIC")
        elif any(keyword in normalized for keyword in ("giay", "paper", "carton", "hopgiay")):
            mapped_types.append("PAPER")
        elif any(keyword in normalized for keyword in ("kimloai", "lon", "metal")):
            mapped_types.append("METAL")
        elif any(keyword in normalized for keyword in ("thuytinh", "glass")):
            mapped_types.append("GLASS")
        elif any(keyword in normalized for keyword in ("huuco", "organic")):
            mapped_types.append("ORGANIC")
        elif _is_waste_related_class(class_name):
            mapped_types.append("UNKNOWN")

    mapped_types = [waste_type for waste_type in mapped_types if waste_type != "UNKNOWN"]

    if not mapped_types:
        return "UNKNOWN"
    if len(set(mapped_types)) > 1:
        return "MIXED_WASTE"
    return mapped_types[0]


def _map_severity_score_to_level(severity_score: float) -> str:
    if severity_score <= 30:
        return "LOW"
    if severity_score <= 60:
        return "MEDIUM"
    if severity_score <= 80:
        return "HIGH"
    return "CRITICAL"


def _calculate_severity_score(
    waste_area_score: float,
    object_count_score: float,
    waste_type: str,
    final_waste_score: float,
) -> float:
    risk_score = WASTE_TYPE_RISK_SCORE.get(waste_type, WASTE_TYPE_RISK_SCORE["UNKNOWN"])
    return round(
        (waste_area_score * 40)
        + (object_count_score * 25)
        + (risk_score * 20)
        + (final_waste_score * 15),
        2,
    )


def _analyze_detections(image_path: str, detections: List[Dict]) -> Dict:
    image = cv2.imread(image_path)
    if image is None:
        return {
            "is_waste": False,
            "ai_decision": "AI_ERROR",
            "report_status": "REQUEST_REUPLOAD",
            "error_message": "Không thể đọc ảnh để phân tích.",
            "need_manual_review": False,
        }

    image_height, image_width = image.shape[:2]
    image_quality_score, quality_error = _calculate_image_quality_score(image_path)
    coco_detections = _run_coco_context_detection(image_path)
    has_general_object_context, general_object_reason = _detect_non_waste_context(coco_detections)

    valid_waste_detections = [d for d in detections if _is_waste_related_class(d["class_name"])]
    object_count = len(valid_waste_detections)
    detected_items = [d["class_name"] for d in detections]  # Tất cả detected items
    waste_type = _map_waste_type([d["class_name"] for d in valid_waste_detections])

    # === BƯỚC 1: Kiểm tra chất lượng ảnh ===
    if image_quality_score < 0.35:
        return {
            "is_waste": False,
            "ai_decision": "NOT_WASTE",
            "report_status": "REQUEST_REUPLOAD",
            "detected_items": detected_items,
            "waste_type": waste_type,
            "object_confidence_score": 0.0,
            "waste_context_score": 0.0,
            "final_waste_score": 0.0,
            "waste_area_ratio": 0.0,
            "object_count": object_count,
            "object_count_score": 0.0,
            "waste_area_score": 0.0,
            "waste_diversity_score": 0.0,
            "scene_context_score": 0.0,
            "discarded_sign_score": 0.0,
            "severity_score": None,
            "pollution_level": "UNCONFIRMED",
            "severity_description": POLLUTION_LEVEL_TEMPLATES["UNCONFIRMED"],
            "recommendation": RECOMMENDATION_TEMPLATES["UNKNOWN"],
            "need_manual_review": False,
            "error_message": quality_error,
            "image_quality_score": image_quality_score,
            "false_positive_reason": None,
            "model_a_has_general_object": has_general_object_context,
            "model_a_detected_items": _translate_coco_detections_to_vietnamese(coco_detections),
            "model_a_reason": general_object_reason,
            "model_b_detected_items": detected_items,
            "final_waste_decision": "REQUEST_REUPLOAD",
        }

    # === BƯỚC 3: Không có phát hiện rác ===
    if not valid_waste_detections:
        final_ai_decision, final_report_status, need_manual_review = _resolve_final_waste_decision(
            0.0,
            has_general_object_context,
            0,
            0.0,
        )
        return {
            "is_waste": False,
            "ai_decision": final_ai_decision,
            "report_status": final_report_status,
            "detected_items": detected_items,
            "waste_type": "UNKNOWN",
            "object_confidence_score": 0.0,
            "waste_context_score": 0.0,
            "final_waste_score": 0.0,
            "waste_area_ratio": 0.0,
            "object_count": 0,
            "object_count_score": 0.0,
            "waste_area_score": 0.0,
            "waste_diversity_score": 0.0,
            "scene_context_score": 0.0,
            "discarded_sign_score": 0.0,
            "severity_score": None,
            "pollution_level": "UNCONFIRMED",
            "severity_description": general_object_reason or POLLUTION_LEVEL_TEMPLATES["UNCONFIRMED"],
            "recommendation": RECOMMENDATION_TEMPLATES["UNKNOWN"],
            "need_manual_review": need_manual_review,
            "error_message": None,
            "image_quality_score": image_quality_score,
            "false_positive_reason": general_object_reason or None,
            "model_a_has_general_object": has_general_object_context,
            "model_a_detected_items": _translate_coco_detections_to_vietnamese(coco_detections),
            "model_a_reason": general_object_reason,
            "model_b_detected_items": detected_items,
            "final_waste_decision": final_report_status,
        }

    # === BƯỚC 4: Tính toán điểm số rác ====
    object_confidence_score = round(
        sum(d["confidence"] for d in valid_waste_detections) / max(object_count, 1),
        2,
    )
    waste_area_ratio = round(
        min(
            sum(_calculate_bbox_area_ratio(d["bbox"], image_width, image_height) for d in valid_waste_detections),
            1.0,
        ),
        2,
    )

    low_context_fp, fp_reason = _is_low_context_false_positive(object_count, waste_area_ratio, image_quality_score)
    if low_context_fp:
        return {
            "is_waste": False,
            "ai_decision": "NOT_WASTE",
            "report_status": "REQUEST_REUPLOAD",
            "detected_items": detected_items,
            "waste_type": waste_type,
            "object_confidence_score": object_confidence_score,
            "waste_context_score": 0.0,
            "final_waste_score": 0.0,
            "waste_area_ratio": waste_area_ratio,
            "object_count": object_count,
            "object_count_score": _calculate_object_count_score(object_count),
            "waste_area_score": 0.0,
            "waste_diversity_score": 0.0,
            "scene_context_score": 0.0,
            "discarded_sign_score": 0.0,
            "severity_score": None,
            "pollution_level": "UNCONFIRMED",
            "severity_description": fp_reason,
            "recommendation": RECOMMENDATION_TEMPLATES["UNKNOWN"],
            "need_manual_review": False,
            "error_message": None,
            "image_quality_score": image_quality_score,
            "false_positive_reason": fp_reason,
        }

    object_count_score = _calculate_object_count_score(object_count)
    waste_area_score = _calculate_waste_area_score(waste_area_ratio)
    waste_diversity_score = _calculate_waste_diversity_score([d["class_name"] for d in valid_waste_detections])
    scene_context_score = _calculate_scene_context_score(object_count, waste_area_ratio, image_quality_score)
    discarded_sign_score = _calculate_discarded_sign_score(object_count, waste_area_ratio, object_confidence_score)

    waste_context_score = round(
        (object_count_score * 0.25)
        + (waste_area_score * 0.30)
        + (waste_diversity_score * 0.15)
        + (scene_context_score * 0.20)
        + (discarded_sign_score * 0.10),
        2,
    )

    final_waste_score = round((object_confidence_score * 0.5) + (waste_context_score * 0.5), 2)

    # === BƯỚC 5: Quyết định cuối cùng ===
    ai_decision, report_status, need_manual_review = _resolve_final_waste_decision(
        final_waste_score,
        has_general_object_context,
        object_count,
        waste_area_ratio,
    )

    if ai_decision == "WASTE_DETECTED":
        severity_score = _calculate_severity_score(waste_area_score, object_count_score, waste_type, final_waste_score)
        pollution_level = _map_severity_score_to_level(severity_score)
        severity_description = POLLUTION_LEVEL_TEMPLATES[pollution_level]
        recommendation = RECOMMENDATION_TEMPLATES.get(waste_type, RECOMMENDATION_TEMPLATES["UNKNOWN"])
    elif ai_decision == "UNCERTAIN":
        severity_score = None
        pollution_level = "UNCONFIRMED"
        severity_description = general_object_reason or POLLUTION_LEVEL_TEMPLATES["UNCONFIRMED"]
        recommendation = RECOMMENDATION_TEMPLATES.get(waste_type, RECOMMENDATION_TEMPLATES["UNKNOWN"])
    else:
        severity_score = None
        pollution_level = "UNCONFIRMED"
        severity_description = general_object_reason or "Ảnh chưa thể hiện rõ rác thải hoặc không phù hợp với chức năng báo cáo rác."
        recommendation = RECOMMENDATION_TEMPLATES["UNKNOWN"]

    return {
        "is_waste": ai_decision == "WASTE_DETECTED",
        "detected_items": detected_items,
        "waste_type": waste_type,
        "object_confidence_score": object_confidence_score,
        "waste_context_score": waste_context_score,
        "final_waste_score": final_waste_score,
        "waste_area_ratio": waste_area_ratio,
        "object_count": object_count,
        "severity_score": severity_score,
        "pollution_level": pollution_level,
        "severity_description": severity_description,
        "recommendation": recommendation,
        "ai_decision": ai_decision,
        "report_status": report_status,
        "need_manual_review": need_manual_review,
        "error_message": None,
        "image_quality_score": image_quality_score,
        "object_count_score": object_count_score,
        "waste_area_score": waste_area_score,
        "waste_diversity_score": waste_diversity_score,
        "scene_context_score": scene_context_score,
        "discarded_sign_score": discarded_sign_score,
        "false_positive_reason": None,
        "model_a_has_general_object": has_general_object_context,
        "model_a_detected_items": _translate_coco_detections_to_vietnamese(coco_detections),
        "model_a_reason": general_object_reason,
        "model_b_detected_items": detected_items,
        "final_waste_decision": report_status,
    }


def run_detection(image_path: str):
    """Chạy detection model (train5) để xác minh rác thải"""
    results = detection_model(image_path)[0]

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = results.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        detections.append({
            "class_id": cls_id,
            "class_name": cls_name,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2]
        })

    valid_detections = filter_valid_detections(detections)

    overall_conf = calculate_overall_confidence(valid_detections)
    type_percentage = calculate_type_percentage(valid_detections)
    analysis = _analyze_detections(image_path, valid_detections)

    return {
        "is_waste": analysis["is_waste"],
        "overall_confidence": overall_conf,
        "model_a_has_general_object": analysis.get("model_a_has_general_object", False),
        "model_a_detected_items": analysis.get("model_a_detected_items", []),
        "model_a_reason": analysis.get("model_a_reason"),
        "model_b_detected_items": analysis.get("model_b_detected_items", []),
        "final_waste_decision": analysis.get("final_waste_decision"),
        "object_confidence_score": analysis["object_confidence_score"],
        "waste_context_score": analysis["waste_context_score"],
        "final_waste_score": analysis["final_waste_score"],
        "waste_area_ratio": analysis["waste_area_ratio"],
        "object_count": analysis["object_count"],
        "severity_score": analysis["severity_score"],
        "pollution_level": analysis["pollution_level"],
        "severity_description": analysis["severity_description"],
        "recommendation": analysis["recommendation"],
        "ai_decision": analysis["ai_decision"],
        "report_status": analysis["report_status"],
        "need_manual_review": analysis["need_manual_review"],
        "error_message": analysis["error_message"],
        "detected_items": analysis["detected_items"],
        "image_quality_score": analysis["image_quality_score"],
        "object_count_score": analysis["object_count_score"],
        "waste_area_score": analysis["waste_area_score"],
        "waste_diversity_score": analysis["waste_diversity_score"],
        "scene_context_score": analysis["scene_context_score"],
        "discarded_sign_score": analysis["discarded_sign_score"],
        "total_objects_detected": len(valid_detections),
        "type_percentage": type_percentage,
        "detections": valid_detections
    }


def draw_boxes(image_path: str, detections: list, output_path: str, use_vietnamese: bool = False):
    """Vẽ bounding boxes lên hình ảnh. Nếu use_vietnamese=True sẽ dùng label tiếng Việt"""
    image = cv2.imread(image_path)

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        class_name = d["class_name"]
        
        # Dịch sang tiếng Việt nếu cần
        if use_vietnamese:
            class_name = translate_class_name(class_name)
        
        label = f'{class_name} {d["confidence"]:.2f}'

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imwrite(output_path, image)


def run_waste_classification(image_path: str):
    """
    Chạy classification model (train5) để phân loại rác thải
    Trả về thông tin phân loại với Vietnamese labels
    """
    results = classification_model(image_path)[0]

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = results.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Dịch tên class sang tiếng Việt theo mapping của train5
        vietnamese_name = translate_class_name(cls_name)

        detections.append({
            "class_id": cls_id,
            "class_name": cls_name,
            "class_name_vietnamese": vietnamese_name,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2]
        })

    valid_detections = filter_valid_detections(detections)

    is_trash = len(valid_detections) > 0
    overall_conf = calculate_overall_confidence(valid_detections)
    type_percentage = calculate_type_percentage(valid_detections)

    # Tính toán type_percentage với Vietnamese names
    type_percentage_vietnamese = {}
    for k, v in type_percentage.items():
        vietnamese_name = translate_class_name(k)
        type_percentage_vietnamese[vietnamese_name] = v

    return {
        "is_trash": is_trash,
        "overall_confidence": overall_conf,
        "total_objects_detected": len(valid_detections),
        "type_percentage_raw": type_percentage,  # Tên gốc
        "type_percentage": type_percentage_vietnamese,  # Tên tiếng Việt
        "detections": valid_detections
    }


def draw_classification_boxes(image_path: str, detections: list, output_path: str):
    """
    Vẽ bounding boxes lên hình ảnh với Vietnamese labels cho classification
    """
    image = cv2.imread(image_path)

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        # Dùng Vietnamese name
        vietnamese_name = d.get("class_name_vietnamese", d["class_name"])
        label = f'{vietnamese_name} {d["confidence"]:.2f}'

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imwrite(output_path, image)
 