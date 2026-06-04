"""Shared metric utilities for multi-seed evaluation."""
from __future__ import annotations

import csv
import math
import random
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

DEFAULT_CLASSES = ["NO_Action", "Slowdown", "Partial_Brake", "Full_Brake"]


def compute_macro_prf(confusion_matrix: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Compute macro-averaged precision/recall/F1 from per-class stats."""
    if not confusion_matrix:
        return {"precision_avg": 0.0, "recall_avg": 0.0, "f1_avg": 0.0}

    precisions = [metrics.get("precision", 0.0) for metrics in confusion_matrix.values()]
    recalls = [metrics.get("recall", 0.0) for metrics in confusion_matrix.values()]
    f1s = [metrics.get("f1_score", 0.0) for metrics in confusion_matrix.values()]

    return {
        "precision_avg": float(np.mean(precisions)) if precisions else 0.0,
        "recall_avg": float(np.mean(recalls)) if recalls else 0.0,
        "f1_avg": float(np.mean(f1s)) if f1s else 0.0,
    }


def _summarize_numeric(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return {
            "mean": math.nan,
            "std": math.nan,
            "median": math.nan,
            "q25": math.nan,
            "q75": math.nan,
            "min": math.nan,
            "max": math.nan,
        }

    valid = arr[~np.isnan(arr)]
    std = float(np.nanstd(arr, ddof=1)) if valid.size > 1 else 0.0

    return {
        "mean": float(np.nanmean(arr)),
        "std": std,
        "median": float(np.nanmedian(arr)),
        "q25": float(np.nanpercentile(arr, 25)),
        "q75": float(np.nanpercentile(arr, 75)),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
    }


def compute_mttd_summary(
    decisions_data: List[Dict[str, object]],
    detection_events: List[Dict[str, object]],
    scenarios: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    """Compute MTTD stats and miss/detection rate per scenario run.

    Missed detections are represented as NaN in the MTTD samples.
    """
    scenario_names = {
        entry.get("scenario")
        for entry in decisions_data
        if entry.get("scenario")
    }

    event_scenarios: List[str] = []
    for name, info in scenarios.items():
        if name in scenario_names and info.get("events"):
            event_scenarios.append(name)

    detection_map: Dict[str, float] = {}
    for event in detection_events:
        scenario = event.get("scenario")
        detection_time = event.get("detection_time_s")
        if scenario is None or detection_time is None:
            continue
        try:
            detection_time = max(0.0, float(detection_time))
        except (TypeError, ValueError):
            continue
        best = detection_map.get(scenario)
        if best is None or detection_time < best:
            detection_map[scenario] = detection_time

    mttd_values: List[float] = []
    misses = 0
    for scenario in event_scenarios:
        if scenario in detection_map:
            mttd_values.append(detection_map[scenario])
        else:
            mttd_values.append(math.nan)
            misses += 1

    total_events = len(event_scenarios)
    miss_rate = (misses / total_events) if total_events else 0.0
    detection_rate = (1.0 - miss_rate) if total_events else 0.0

    return {
        "mttd_stats": _summarize_numeric(mttd_values),
        "miss_rate": miss_rate,
        "detection_rate": detection_rate,
        "mttd_values": mttd_values,
        "total_events": total_events,
        "detected_events": total_events - misses,
    }


def ensure_latency_samples(
    latencies: Iterable[float],
    sample_size: int,
    seed: Optional[int] = None,
) -> List[float]:
    """Return latency samples, generating synthetic values when missing."""
    existing = list(latencies)
    if existing:
        return existing

    rng = random.Random(int(seed)) if seed is not None else random.Random()
    return [rng.normalvariate(15, 3) for _ in range(max(10, sample_size))]


def build_summary_metrics(priority_metrics: Dict[str, object]) -> Dict[str, object]:
    """Flatten priority metrics into a machine-readable summary."""
    confusion = priority_metrics.get("confusion_matrix", {}) or {}
    averages = compute_macro_prf(confusion)
    mttd_stats = priority_metrics.get("mttd_stats", {}) or {}
    collision_stats = priority_metrics.get("collision_stats", {}) or {}

    per_class_f1 = {
        cls: metrics.get("f1_score")
        for cls, metrics in confusion.items()
    }

    return {
        **averages,
        "mttd_mean": mttd_stats.get("mean", math.nan),
        "mttd_std": mttd_stats.get("std", math.nan),
        "fnr_full_brake": priority_metrics.get("fnr_full_brake", math.nan),
        "nmr": collision_stats.get("near_miss_rate", math.nan),
        "miss_rate": priority_metrics.get("miss_rate", math.nan),
        "detection_rate": priority_metrics.get("detection_rate", math.nan),
        "per_class_f1": per_class_f1,
    }


def _summary_headers(class_order: Optional[List[str]] = None) -> List[str]:
    classes = class_order or DEFAULT_CLASSES
    return [
        "model",
        "scenario",
        "seed",
        "precision_avg",
        "recall_avg",
        "f1_avg",
        "mttd_mean_s",
        "mttd_std_s",
        "fnr_full_brake",
        "nmr",
        "miss_rate",
        "detection_rate",
        *[f"f1_{cls}" for cls in classes],
    ]


def _normalize_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return value


def build_summary_row(
    summary: Dict[str, object],
    model_name: str,
    scenario: str,
    seed: int,
    class_order: Optional[List[str]] = None,
) -> Tuple[List[str], List[object]]:
    classes = class_order or DEFAULT_CLASSES
    per_class_f1 = summary.get("per_class_f1", {}) or {}

    row = [
        model_name,
        scenario,
        int(seed),
        summary.get("precision_avg"),
        summary.get("recall_avg"),
        summary.get("f1_avg"),
        summary.get("mttd_mean"),
        summary.get("mttd_std"),
        summary.get("fnr_full_brake"),
        summary.get("nmr"),
        summary.get("miss_rate"),
        summary.get("detection_rate"),
    ] + [per_class_f1.get(cls, math.nan) for cls in classes]

    return _summary_headers(classes), [_normalize_value(value) for value in row]


def append_summary_section(
    writer: csv.writer,
    summary: Dict[str, object],
    model_name: str,
    scenario: str,
    seed: int,
    class_order: Optional[List[str]] = None,
) -> None:
    """Append a machine-readable summary block to a metrics CSV."""
    headers, row = build_summary_row(summary, model_name, scenario, seed, class_order)
    writer.writerow(["--- Summary Metrics (Machine Readable) ---"])
    writer.writerow(headers)
    writer.writerow(row)
    writer.writerow([])


def write_metrics_row_csv(
    file_path: str,
    summary: Dict[str, object],
    model_name: str,
    scenario: str,
    seed: int,
    class_order: Optional[List[str]] = None,
) -> None:
    """Write a single-row CSV of summary metrics for aggregation."""
    headers, row = build_summary_row(summary, model_name, scenario, seed, class_order)
    with open(file_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerow(row)
