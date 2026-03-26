import math
from dataclasses import dataclass

from app.request.hotspot_schemas import ClusterHotspotRequest, ClusterHotspotResponse, HotspotZone


EARTH_RADIUS_KM = 6371.0088


@dataclass
class _ClusterPoint:
    idx: int
    report_id: int
    lat: float
    lng: float


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lng = math.radians(lng2 - lng1)

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * (math.sin(delta_lng / 2) ** 2)
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_KM * c


def _region_query(points: list[_ClusterPoint], point_index: int, eps_km: float) -> list[int]:
    origin = points[point_index]
    neighbors: list[int] = []
    for candidate in points:
        if haversine_km(origin.lat, origin.lng, candidate.lat, candidate.lng) <= eps_km:
            neighbors.append(candidate.idx)
    return neighbors


def _expand_cluster(
    points: list[_ClusterPoint],
    labels: list[int],
    visited: list[bool],
    point_index: int,
    neighbors: list[int],
    cluster_id: int,
    eps_km: float,
    min_samples: int,
) -> None:
    labels[point_index] = cluster_id
    i = 0
    while i < len(neighbors):
        neighbor_index = neighbors[i]
        if not visited[neighbor_index]:
            visited[neighbor_index] = True
            neighbor_neighbors = _region_query(points, neighbor_index, eps_km)
            if len(neighbor_neighbors) >= min_samples:
                for nn in neighbor_neighbors:
                    if nn not in neighbors:
                        neighbors.append(nn)

        if labels[neighbor_index] == -1:
            labels[neighbor_index] = cluster_id

        i += 1


def run_dbscan(points: list[_ClusterPoint], eps_km: float, min_samples: int) -> list[int]:
    labels = [-1] * len(points)
    visited = [False] * len(points)
    current_cluster = 0

    for idx in range(len(points)):
        if visited[idx]:
            continue

        visited[idx] = True
        neighbors = _region_query(points, idx, eps_km)

        if len(neighbors) < min_samples:
            labels[idx] = -1
            continue

        current_cluster += 1
        _expand_cluster(
            points,
            labels,
            visited,
            idx,
            neighbors,
            current_cluster,
            eps_km,
            min_samples,
        )

    return labels


def cluster_hotspots(request: ClusterHotspotRequest) -> ClusterHotspotResponse:
    if not request.reports:
        return ClusterHotspotResponse(success=True, hotspots=[], total_clusters=0, noise_points=0)

    points: list[_ClusterPoint] = []
    for idx, report in enumerate(request.reports):
        points.append(
            _ClusterPoint(
                idx=idx,
                report_id=report.id if report.id is not None else idx,
                lat=report.lat,
                lng=report.lng,
            )
        )

    labels = run_dbscan(points, request.eps_km, request.min_samples)

    clusters: dict[int, list[_ClusterPoint]] = {}
    noise_count = 0

    for idx, label in enumerate(labels):
        if label == -1:
            noise_count += 1
            continue
        clusters.setdefault(label, []).append(points[idx])

    hotspots: list[HotspotZone] = []
    for cluster_id, cluster_points in clusters.items():
        center_lat = sum(p.lat for p in cluster_points) / len(cluster_points)
        center_lng = sum(p.lng for p in cluster_points) / len(cluster_points)

        radius_km = 0.0
        for p in cluster_points:
            radius_km = max(radius_km, haversine_km(center_lat, center_lng, p.lat, p.lng))

        hotspots.append(
            HotspotZone(
                cluster_id=cluster_id,
                center_lat=center_lat,
                center_lng=center_lng,
                radius_km=radius_km,
                report_count=len(cluster_points),
                report_ids=[p.report_id for p in cluster_points],
            )
        )

    hotspots.sort(key=lambda h: h.report_count, reverse=True)

    return ClusterHotspotResponse(
        success=True,
        hotspots=hotspots,
        total_clusters=len(hotspots),
        noise_points=noise_count,
    )
