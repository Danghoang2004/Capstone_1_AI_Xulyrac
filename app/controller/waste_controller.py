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

app = FastAPI(title="EcoTrack AI Service", version="1.0.0")

RAW_DIR = "outputs/raw"
ANALYZED_DIR = "outputs/analyzed"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(ANALYZED_DIR, exist_ok=True)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "EcoTrack AI Service",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/",
            "detect_waste": "/ai/detect-waste",
            "classify_waste": "/ai/classify-waste",
            "cluster_hotspots": "/ai/cluster-hotspots",
            "predict_hotspots": "/ai/predict-hotspots"
        }
    }


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
    if ai_result.get("detections") and ai_result.get("ai_decision") != "NOT_WASTE":
        draw_boxes(raw_path, ai_result["detections"], analyzed_path)
        output_image = analyzed_path

    return {
        "success": True,
        "data": {
            **ai_result,
            "output_image": output_image,
            "detected_items": ai_result.get("detected_items", []),
            "object_confidence_score": ai_result.get("object_confidence_score", 0.0),
            "waste_context_score": ai_result.get("waste_context_score", 0.0),
            "final_waste_score": ai_result.get("final_waste_score", 0.0),
            "waste_area_ratio": ai_result.get("waste_area_ratio", 0.0),
            "object_count": ai_result.get("object_count", 0),
            "severity_score": ai_result.get("severity_score"),
            "pollution_level": ai_result.get("pollution_level", "UNCONFIRMED"),
            "severity_description": ai_result.get("severity_description"),
            "recommendation": ai_result.get("recommendation"),
            "ai_decision": ai_result.get("ai_decision", "NOT_WASTE"),
            "report_status": ai_result.get("report_status", "REQUEST_REUPLOAD"),
            "need_manual_review": ai_result.get("need_manual_review", False),
            "error_message": ai_result.get("error_message"),
            "image_quality_score": ai_result.get("image_quality_score", 0.0),
            "object_count_score": ai_result.get("object_count_score", 0.0),
            "waste_area_score": ai_result.get("waste_area_score", 0.0),
            "waste_diversity_score": ai_result.get("waste_diversity_score", 0.0),
            "scene_context_score": ai_result.get("scene_context_score", 0.0),
            "discarded_sign_score": ai_result.get("discarded_sign_score", 0.0)
        }
    }


@app.post("/ai/classify-waste")
async def classify_waste(file: UploadFile = File(...)):
    """
    Endpoint để phân loại rác thải
    - Nhận ảnh từ người dùng
    - Chạy model train5 để nhận diện và phân loại
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