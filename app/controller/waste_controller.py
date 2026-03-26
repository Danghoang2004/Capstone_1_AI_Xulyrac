import os
import uuid
import sys

import uvicorn

# Support running this file directly: `python model_process/app/main.py`
if __package__ is None or __package__ == "":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from fastapi import FastAPI, UploadFile, File
from app.service.hotspot.clustering import cluster_hotspots
from app.service.hotspot.prediction import predict_hotspots
from app.request.hotspot_schemas import (
    ClusterHotspotRequest,
    ClusterHotspotResponse,
    PredictHotspotRequest,
    PredictHotspotResponse,
)
from app.service.yolo_service.yolo_service import (
    run_detection, 
    draw_boxes,
    run_waste_classification,
    draw_classification_boxes
)

app = FastAPI()

RAW_DIR = "outputs/raw"
ANALYZED_DIR = "outputs/analyzed"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(ANALYZED_DIR, exist_ok=True)


@app.post("/ai/cluster-hotspots", response_model=ClusterHotspotResponse)
async def cluster_waste_hotspots(payload: ClusterHotspotRequest):
    return cluster_hotspots(payload)


@app.post("/ai/predict-hotspots", response_model=PredictHotspotResponse)
async def predict_waste_hotspots(payload: PredictHotspotRequest):
    return predict_hotspots(payload)


@app.post("/ai/detect-waste")
async def detect_waste(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    raw_path = f"{RAW_DIR}/{file_id}.jpg"
    analyzed_path = f"{ANALYZED_DIR}/{file_id}.jpg"

    with open(raw_path, "wb") as f:
        f.write(await file.read())

    ai_result = run_detection(raw_path)

    output_image = None
    if ai_result["is_waste"]:
        draw_boxes(raw_path, ai_result["detections"], analyzed_path)
        output_image = analyzed_path

    return {
        "success": True,
        "data": {
            **ai_result,
            "output_image": output_image
        }
    }


@app.post("/ai/classify-waste")
async def classify_waste(file: UploadFile = File(...)):
    """
    Endpoint để phân loại rác thải
    - Nhận ảnh từ người dùng
    - Chạy model trainphanloai để nhận diện và phân loại
    - Trả về ảnh được vẽ boxes với Vietnamese labels
    - Trả về danh sách các loại rác thải được phân loại
    """
    file_id = str(uuid.uuid4())
    raw_path = f"{RAW_DIR}/{file_id}.jpg"
    analyzed_path = f"{ANALYZED_DIR}/{file_id}_classified.jpg"

    with open(raw_path, "wb") as f:
        f.write(await file.read())

    # Chạy classification model
    classification_result = run_waste_classification(raw_path)

    output_image = None
    if classification_result["is_trash"]:
        # Vẽ boxes với Vietnamese labels
        draw_classification_boxes(
            raw_path, 
            classification_result["detections"], 
            analyzed_path
        )
        output_image = analyzed_path

    return {
        "success": True,
        "data": {
            "is_trash": classification_result["is_trash"],
            "overall_confidence": classification_result["overall_confidence"],
            "total_objects_detected": classification_result["total_objects_detected"],
            "waste_types": classification_result["type_percentage"],  # Tiếng Việt
            "detections": [
                {
                    "class_name_vietnamese": d["class_name_vietnamese"],
                    "class_name_raw": d["class_name"],
                    "confidence": d["confidence"],
                    "bbox": d["bbox"]
                }
                for d in classification_result["detections"]
            ],
            "output_image": output_image
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )