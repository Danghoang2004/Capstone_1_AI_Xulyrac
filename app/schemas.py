from typing import List, Dict
from pydantic import BaseModel


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: List[int]


class ClassificationDetection(BaseModel):
    class_id: int
    class_name: str
    class_name_vietnamese: str
    confidence: float
    bbox: List[int]


class AiResponse(BaseModel):
    is_waste: bool
    overall_confidence: float
    total_objects_detected: int
    type_percentage: Dict[str, float]
    detections: List[Detection]
    output_image: str | None


class ClassificationResponse(BaseModel):
    is_trash: bool
    overall_confidence: float
    total_objects_detected: int
    type_percentage_raw: Dict[str, float]  # Tên gốc
    type_percentage: Dict[str, float]  # Tên tiếng Việt
    detections: List[ClassificationDetection]
    output_image: str | None
