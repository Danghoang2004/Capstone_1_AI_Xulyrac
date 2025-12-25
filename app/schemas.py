from typing import List, Dict
from pydantic import BaseModel


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: List[int]


class AiResponse(BaseModel):
    is_waste: bool
    overall_confidence: float
    total_objects_detected: int
    type_percentage: Dict[str, float]
    detections: List[Detection]
    output_image: str | None
