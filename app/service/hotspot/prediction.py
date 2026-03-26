import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta

from app.service.hotspot.clustering import cluster_hotspots
from app.request.hotspot_schemas import (
    ClusterHotspotRequest,
    GridRisk,
    HeatmapPoint,
    PredictHotspotRequest,
    PredictHotspotResponse,
    ReportPoint,
)

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


EARTH_RADIUS_M = 6378137.0


@dataclass
class _GridCellState:
    grid_id: str
    center_lat: float
    center_lng: float
    counts_by_day: dict[date, int]


def _to_web_mercator(lat: float, lng: float) -> tuple[float, float]:
    lat = max(min(lat, 89.9), -89.9)
    x = EARTH_RADIUS_M * math.radians(lng)
    y = EARTH_RADIUS_M * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


def _from_web_mercator(x: float, y: float) -> tuple[float, float]:
    lng = math.degrees(x / EARTH_RADIUS_M)
    lat = math.degrees(2 * math.atan(math.exp(y / EARTH_RADIUS_M)) - math.pi / 2)
    return lat, lng


def _grid_id(lat: float, lng: float, grid_size_m: int) -> tuple[str, float, float]:
    x, y = _to_web_mercator(lat, lng)
    gx = int(math.floor(x / grid_size_m))
    gy = int(math.floor(y / grid_size_m))
    center_x = (gx + 0.5) * grid_size_m
    center_y = (gy + 0.5) * grid_size_m
    center_lat, center_lng = _from_web_mercator(center_x, center_y)
    return f"{gx}:{gy}", center_lat, center_lng


def _build_grid_states(reports: list[ReportPoint], grid_size_m: int) -> dict[str, _GridCellState]:
    states: dict[str, _GridCellState] = {}

    for report in reports:
        if report.timestamp is None:
            continue

        gid, center_lat, center_lng = _grid_id(report.lat, report.lng, grid_size_m)
        if gid not in states:
            states[gid] = _GridCellState(
                grid_id=gid,
                center_lat=center_lat,
                center_lng=center_lng,
                counts_by_day=defaultdict(int),
            )

        day_key = report.timestamp.date()
        states[gid].counts_by_day[day_key] += 1

    return states


def _sum_days(counts_by_day: dict[date, int], end_day: date, days: int) -> int:
    return sum(counts_by_day.get(end_day - timedelta(days=d), 0) for d in range(days))


def _predict_by_heuristic(counts_by_day: dict[date, int], today: date, is_weekend: bool) -> float:
    lag1 = counts_by_day.get(today - timedelta(days=1), 0)
    roll3 = _sum_days(counts_by_day, today - timedelta(days=1), 3)
    roll7 = _sum_days(counts_by_day, today - timedelta(days=1), 7)

    base = 0.45 * lag1 + 0.35 * (roll3 / 3.0) + 0.20 * (roll7 / 7.0)
    if is_weekend:
        base *= 1.12

    return max(0.0, base * 7.0)


def _build_training_samples(states: dict[str, _GridCellState], horizon_days: int) -> tuple[list[list[float]], list[float]]:
    rows_x: list[list[float]] = []
    rows_y: list[float] = []

    for state in states.values():
        if not state.counts_by_day:
            continue

        start_day = min(state.counts_by_day.keys())
        end_day = max(state.counts_by_day.keys())

        current = start_day + timedelta(days=7)
        while current <= end_day - timedelta(days=horizon_days):
            lag1 = state.counts_by_day.get(current - timedelta(days=1), 0)
            roll3 = _sum_days(state.counts_by_day, current - timedelta(days=1), 3)
            roll7 = _sum_days(state.counts_by_day, current - timedelta(days=1), 7)
            dow = current.weekday()
            is_weekend = 1 if dow >= 5 else 0

            target = sum(
                state.counts_by_day.get(current + timedelta(days=offset), 0)
                for offset in range(horizon_days)
            )

            rows_x.append([lag1, roll3, roll7, dow, is_weekend])
            rows_y.append(float(target))
            current += timedelta(days=1)

    return rows_x, rows_y


def _predict_with_xgboost(
    states: dict[str, _GridCellState],
    horizon_days: int,
    today: date,
) -> dict[str, float]:
    x_train, y_train = _build_training_samples(states, horizon_days)

    if len(x_train) < 40 or XGBRegressor is None:
        return {}

    model = XGBRegressor(
        n_estimators=180,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(x_train, y_train)

    predictions: dict[str, float] = {}
    for gid, state in states.items():
        lag1 = state.counts_by_day.get(today - timedelta(days=1), 0)
        roll3 = _sum_days(state.counts_by_day, today - timedelta(days=1), 3)
        roll7 = _sum_days(state.counts_by_day, today - timedelta(days=1), 7)
        dow = today.weekday()
        is_weekend = 1 if dow >= 5 else 0

        pred = float(model.predict([[lag1, roll3, roll7, dow, is_weekend]])[0])
        predictions[gid] = max(0.0, pred)

    return predictions


def _filter_reports(reports: list[ReportPoint]) -> list[ReportPoint]:
    cleaned: list[ReportPoint] = []

    # Deduplicate by user+location+day to reduce repeated spam reports in short windows.
    dedup_keys: set[tuple[int | None, int, int, date | None]] = set()

    for report in reports:
        if report.lat < -90 or report.lat > 90 or report.lng < -180 or report.lng > 180:
            continue
        if report.ai_verified is False:
            continue

        day_key = report.timestamp.date() if report.timestamp else None
        key = (report.user_id, round(report.lat, 4), round(report.lng, 4), day_key)
        if key in dedup_keys:
            continue

        dedup_keys.add(key)
        cleaned.append(report)

    return cleaned


def predict_hotspots(request: PredictHotspotRequest) -> PredictHotspotResponse:
    reports = _filter_reports(request.reports)
    if not reports:
        return PredictHotspotResponse(
            success=True,
            predicted_hotspots_7_days=[],
            risk_score_by_grid=[],
            heatmap_points=[],
            top_risk_areas=[],
            metadata={"reason": "no_valid_reports"},
        )

    states = _build_grid_states(reports, request.grid_size_m)
    if not states:
        return PredictHotspotResponse(
            success=True,
            predicted_hotspots_7_days=[],
            risk_score_by_grid=[],
            heatmap_points=[],
            top_risk_areas=[],
            metadata={"reason": "no_reports_with_timestamp"},
        )

    latest_day = max(
        (day for state in states.values() for day in state.counts_by_day.keys()),
        default=date.today(),
    )
    predict_day = latest_day + timedelta(days=1)

    xgb_predictions = _predict_with_xgboost(states, request.horizon_days, predict_day)

    grid_predictions: dict[str, float] = {}
    for gid, state in states.items():
        if gid in xgb_predictions:
            grid_predictions[gid] = xgb_predictions[gid]
            continue

        heuristic = _predict_by_heuristic(
            state.counts_by_day,
            predict_day,
            is_weekend=predict_day.weekday() >= 5,
        )
        grid_predictions[gid] = heuristic

    max_pred = max(grid_predictions.values()) if grid_predictions else 1.0

    risks: list[GridRisk] = []
    for gid, predicted in grid_predictions.items():
        state = states[gid]
        risk_score = min(1.0, predicted / max(max_pred, 1.0))
        risks.append(
            GridRisk(
                grid_id=gid,
                center_lat=state.center_lat,
                center_lng=state.center_lng,
                predicted_count_7d=max(0, int(round(predicted))),
                risk_score=risk_score,
            )
        )

    risks.sort(key=lambda r: (r.risk_score, r.predicted_count_7d), reverse=True)

    top_n = max(1, int(math.ceil(len(risks) * request.top_percent)))
    top_candidates = [r for r in risks[:top_n] if r.predicted_count_7d >= request.min_predicted_count]

    candidate_reports = [
        ReportPoint(
            id=index,
            lat=r.center_lat,
            lng=r.center_lng,
        )
        for index, r in enumerate(top_candidates)
    ]

    cluster_result = cluster_hotspots(
        ClusterHotspotRequest(
            reports=candidate_reports,
            eps_km=request.dbscan_eps_km,
            min_samples=request.dbscan_min_samples,
        )
    )

    candidate_lookup = {idx: r for idx, r in enumerate(top_candidates)}
    hotspot_zones = []
    for zone in cluster_result.hotspots:
        selected_cells = [candidate_lookup[rid] for rid in zone.report_ids if rid in candidate_lookup]
        if selected_cells:
            total_pred = sum(cell.predicted_count_7d for cell in selected_cells)
            avg_risk = sum(cell.risk_score for cell in selected_cells) / len(selected_cells)
            zone.predicted_count_7d = total_pred
            zone.risk_score = avg_risk
        hotspot_zones.append(zone)

    hotspot_zones.sort(
        key=lambda z: (
            z.predicted_count_7d if z.predicted_count_7d is not None else 0,
            z.risk_score if z.risk_score is not None else 0,
        ),
        reverse=True,
    )

    heatmap_points = [
        HeatmapPoint(
            lat=r.center_lat,
            lng=r.center_lng,
            intensity=r.risk_score,
            predicted_count_7d=r.predicted_count_7d,
        )
        for r in risks
    ]

    return PredictHotspotResponse(
        success=True,
        predicted_hotspots_7_days=hotspot_zones,
        risk_score_by_grid=risks,
        heatmap_points=heatmap_points,
        top_risk_areas=risks[:10],
        metadata={
            "model": "xgboost" if xgb_predictions else "heuristic-fallback",
            "horizon_days": request.horizon_days,
            "grid_size_m": request.grid_size_m,
            "total_cells": len(risks),
            "candidate_cells": len(top_candidates),
            "clustered_hotspots": len(hotspot_zones),
            "noise_cells": cluster_result.noise_points,
        },
    )
