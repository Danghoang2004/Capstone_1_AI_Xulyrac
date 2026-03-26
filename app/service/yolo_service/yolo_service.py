import os
import cv2
from ultralytics import YOLO
from app.service.yolo_service.utils import (
    filter_valid_detections,
    calculate_type_percentage,
    calculate_overall_confidence,
    translate_class_name
)
from app.config.config import MODEL_DETECT_PATH, MODEL_CLASSIFY_PATH

# Load detection model (train5) for waste verification
detection_model = YOLO(MODEL_DETECT_PATH)

# Load classification model (trainphanloai) for waste classification
classification_model = YOLO(MODEL_CLASSIFY_PATH)


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

    is_waste = len(valid_detections) > 0
    overall_conf = calculate_overall_confidence(valid_detections)
    type_percentage = calculate_type_percentage(valid_detections)

    return {
        "is_waste": is_waste,
        "overall_confidence": overall_conf,
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
    Chạy classification model (trainphanloai) để phân loại rác thải
    Trả về thông tin phân loại với Vietnamese labels
    """
    results = classification_model(image_path)[0]

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = results.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Dịch tên class sang tiếng Việt (sử dụng mapping cho phân loại)
        vietnamese_name = translate_class_name(cls_name, use_classification=True)

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
        vietnamese_name = translate_class_name(k, use_classification=True)
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
