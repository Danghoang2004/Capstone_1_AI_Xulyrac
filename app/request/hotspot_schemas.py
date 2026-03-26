from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ReportPoint(BaseModel):
    id: int | None = None
    lat: float
    lng: float
    timestamp: datetime | None = None
    user_id: int | None = None
    category: str | None = None
    ai_verified: bool | None = None


class ClusterHotspotRequest(BaseModel):
    reports: list[ReportPoint]
    eps_km: float = Field(default=0.5, gt=0)
    min_samples: int = Field(default=2, ge=1)


class HotspotZone(BaseModel):
    cluster_id: int
    center_lat: float
    center_lng: float
    radius_km: float
    report_count: int
    report_ids: list[int]
    risk_score: float | None = None
    predicted_count_7d: int | None = None


class ClusterHotspotResponse(BaseModel):
    success: bool = True
    hotspots: list[HotspotZone] = Field(default_factory=list)
    total_clusters: int = 0
    noise_points: int = 0


class PredictHotspotRequest(BaseModel):
    reports: list[ReportPoint]
    horizon_days: int = Field(default=7, ge=1, le=14)
    grid_size_m: int = Field(default=200, ge=50, le=1000)
    top_percent: float = Field(default=0.1, gt=0.0, le=1.0)
    min_predicted_count: int = Field(default=3, ge=1)
    dbscan_eps_km: float = Field(default=0.35, gt=0)
    dbscan_min_samples: int = Field(default=2, ge=1)


class GridRisk(BaseModel):
    grid_id: str
    center_lat: float
    center_lng: float
    predicted_count_7d: int
    risk_score: float


class HeatmapPoint(BaseModel):
    lat: float
    lng: float
    intensity: float
    predicted_count_7d: int


class PredictHotspotResponse(BaseModel):
    success: bool = True
    predicted_hotspots_7_days: list[HotspotZone] = Field(default_factory=list)
    risk_score_by_grid: list[GridRisk] = Field(default_factory=list)
    heatmap_points: list[HeatmapPoint] = Field(default_factory=list)
    top_risk_areas: list[GridRisk] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
