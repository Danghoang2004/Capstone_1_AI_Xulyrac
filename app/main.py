import os
import uuid

import uvicorn
from fastapi import FastAPI, UploadFile, File
from app.yolo_service import run_detection, draw_boxes

app = FastAPI()

RAW_DIR = "outputs/raw"
ANALYZED_DIR = "outputs/analyzed"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(ANALYZED_DIR, exist_ok=True)


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
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )