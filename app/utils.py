from collections import defaultdict


CONF_THRESHOLD = 0.6


def filter_valid_detections(detections):
    return [d for d in detections if d["confidence"] >= CONF_THRESHOLD]


def calculate_type_percentage(valid_detections):
    type_conf_sum = defaultdict(float)

    for d in valid_detections:
        type_conf_sum[d["class_name"]] += d["confidence"]

    total_conf = sum(type_conf_sum.values())
    if total_conf == 0:
        return {}

    return {
        k: round(v / total_conf * 100, 2)
        for k, v in type_conf_sum.items()
    }


def calculate_overall_confidence(valid_detections):
    if not valid_detections:
        return 0.0

    return round(
        sum(d["confidence"] for d in valid_detections)
        / len(valid_detections),
        2
    )
