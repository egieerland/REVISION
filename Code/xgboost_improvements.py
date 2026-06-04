"""Compatibility helpers for the XGBoost CAN bus simulator.

This module provides the small feature engineering and safety helpers that the
XGBoost simulator expects at import time.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Dict, Iterable, List


def enhance_features_dict(base: Dict[str, float]) -> Dict[str, float]:
    """Return a feature dictionary with a few derived safety features."""

    features = dict(base)
    features.setdefault("severity_score", 0.0)
    features["severity_score"] = float(
        3 * features.get("lidar_full_brake", 0)
        + 2 * features.get("lidar_partial_brake", 0)
        + features.get("lidar_slowdown", 0)
        + 2 * features.get("is_close_behind", 0)
        + features.get("is_medium_behind", 0)
        + features.get("is_object_left", 0)
        + features.get("is_object_right", 0)
    )
    features["blocking_objects"] = int(
        bool(features.get("is_object_left", 0)) + bool(features.get("is_object_right", 0))
    )
    return features


def compute_class_weights(labels: Iterable[int]) -> Dict[int, float]:
    """Compute inverse-frequency class weights."""

    counts = Counter(labels)
    if not counts:
        return {}

    max_count = max(counts.values())
    return {label: max_count / count for label, count in counts.items() if count > 0}


def proba_to_decision(proba: List[float]) -> int:
    """Convert a probability vector into the index of the best class."""

    if not proba:
        return 0
    return max(range(len(proba)), key=lambda idx: proba[idx])


class DecisionSmoother:
    """Simple rolling majority smoother for predicted decisions."""

    def __init__(self, window: int = 5):
        self.window = max(1, int(window))
        self.history = deque(maxlen=self.window)

    def step(self, decision: str, proba: List[float] | None = None) -> str:
        self.history.append(decision)
        return Counter(self.history).most_common(1)[0][0]


class SafetyGuard:
    """Rule-based override that keeps the model conservative."""

    def apply(self, sensors_dict: Dict[str, object], features_dict: Dict[str, float], decision: str) -> str:
        lidar = sensors_dict.get("lidar", "NO_Action")
        fl_cam = sensors_dict.get("fl_cam", "Free")
        fr_cam = sensors_dict.get("fr_cam", "Free")
        back_dist = sensors_dict.get("back_dist_cm", None)

        if lidar == "Full_Brake":
            return "Full_Brake"
        if lidar == "Partial_Brake" and decision != "Full_Brake":
            return "Partial_Brake"
        if lidar == "Slowdown" and decision == "NO_Action":
            return "Slowdown"

        if fl_cam == "Not_Free" or fr_cam == "Not_Free":
            if decision == "NO_Action":
                return "Partial_Brake"

        if isinstance(back_dist, (int, float)):
            if back_dist < 50:
                return "Full_Brake"
            if back_dist <= 400 and decision == "NO_Action":
                return "Slowdown"

        if features_dict.get("severity_score", 0) >= 4 and decision == "NO_Action":
            return "Slowdown"

        return decision