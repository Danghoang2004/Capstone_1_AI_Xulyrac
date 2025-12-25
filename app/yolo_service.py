import os
import cv2
from ultralytics import YOLO
from app.utils import (
    filter_valid_detections,
    calculate_type_percentage,
    calculate_overall_confidence
)

MODEL_PATH = r"D:\AI_XuLyBaoCaoRac\model_process\models\train5\weights\best.pt"
model = YOLO(MODEL_PATH)


def run_detection(image_path: str):
    results = model(image_path)[0]

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


def draw_boxes(image_path: str, detections: list, output_path: str):
    image = cv2.imread(image_path)

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        label = f'{d["class_name"]} {d["confidence"]:.2f}'

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
