"""Shared helpers for robustness stress testing in CAN simulators."""
from __future__ import annotations

import csv
import math
import os
import random
import time
from datetime import datetime
from typing import Dict, Iterable, List

from metrics_utils import build_summary_metrics

_DECISIONS = ["NO_Action", "Slowdown", "Partial_Brake", "Full_Brake"]


def apply_stress_to_scenario_values(
    values: Dict[str, object],
    noise_level: float,
    rng: random.Random,
) -> Dict[str, object]:
    """Apply configurable sensor perturbations to scenario-provided sensor values."""
    if noise_level <= 0.0:
        return dict(values)

    out = dict(values)

    lidar = str(out.get("lidar_decision", "NO_Action"))
    if lidar in _DECISIONS and rng.random() < noise_level:
        idx = _DECISIONS.index(lidar)
        neighbor_candidates: List[str] = []
        if idx > 0:
            neighbor_candidates.append(_DECISIONS[idx - 1])
        if idx < (len(_DECISIONS) - 1):
            neighbor_candidates.append(_DECISIONS[idx + 1])
        fallback_candidates = [d for d in _DECISIONS if d != lidar and d not in neighbor_candidates]
        weighted = neighbor_candidates * 3 + fallback_candidates
        if weighted:
            out["lidar_decision"] = rng.choice(weighted)

    for key in ("front_left_camera", "front_right_camera"):
        cam = str(out.get(key, "Free"))
        if cam in ("Free", "Not_Free") and rng.random() < noise_level:
            out[key] = "Not_Free" if cam == "Free" else "Free"

    if "back_distance" in out:
        try:
            distance = float(out["back_distance"])
            std_cm = max(1.0, 60.0 * noise_level)
            noisy = distance + rng.gauss(0.0, std_cm)
            out["back_distance"] = max(2.0, min(500.0, noisy))
        except Exception:
            pass

    return out


def sample_jitter_seconds(jitter_ms: float, rng: random.Random) -> float:
    if jitter_ms <= 0.0:
        return 0.0
    return rng.uniform(-abs(jitter_ms), abs(jitter_ms)) / 1000.0


def should_drop_message(drop_prob: float, rng: random.Random) -> bool:
    return drop_prob > 0.0 and rng.random() < drop_prob


def _run_scenarios_for_mode(simulator, mode_name: str, timeout_buffer_s: float = 5.0) -> Dict[str, Dict[str, float]]:
    scenarios = [name for name in simulator.scenarios.keys() if name != "Normal Driving"]
    result: Dict[str, Dict[str, float]] = {}

    for scenario in scenarios:
        if hasattr(simulator, "reset_metrics_collection"):
            simulator.reset_metrics_collection()

        simulator.scenario_var.set(scenario)
        simulator.start_scenario()

        duration = float(simulator.scenarios.get(scenario, {}).get("duration", 0) or 0)
        timeout = max(3.0, duration + timeout_buffer_s)
        started = time.time()

        while getattr(simulator, "scenario_active", False) and (time.time() - started) < timeout:
            try:
                simulator.root.update()
            except Exception:
                break
            device_thread = getattr(simulator, "device_thread", None)
            if device_thread is None or not device_thread.is_alive():
                try:
                    simulator.update_decision_logic()
                except Exception:
                    pass
            time.sleep(0.05)

        if getattr(simulator, "scenario_active", False):
            try:
                simulator.stop_scenario()
                simulator.root.update()
            except Exception:
                pass

        priority_metrics = simulator.calculate_priority_metrics()
        summary = build_summary_metrics(priority_metrics)

        result[scenario] = {
            "precision": float(summary.get("precision_avg", math.nan)),
            "recall": float(summary.get("recall_avg", math.nan)),
            "f1": float(summary.get("f1_avg", math.nan)),
            "mttd": float(summary.get("mttd_mean", math.nan)),
            "fnr": float(summary.get("fnr_full_brake", math.nan)),
            "nmr": float(summary.get("nmr", math.nan)),
        }

    return result


def _safe_mean(values: Iterable[float]) -> float:
    clean = [float(v) for v in values if not math.isnan(float(v))]
    if not clean:
        return math.nan
    return sum(clean) / len(clean)


def run_robustness_comparison(simulator, output_dir: str) -> str:
    """Run baseline vs stressed scenarios and write per-scenario + per-model degradation CSVs."""
    os.makedirs(output_dir, exist_ok=True)

    original_enabled = bool(getattr(simulator, "stress_enabled", False))

    simulator.stress_enabled = False
    baseline = _run_scenarios_for_mode(simulator, "baseline")

    simulator.stress_enabled = True
    stressed = _run_scenarios_for_mode(simulator, "stressed")

    simulator.stress_enabled = original_enabled

    model_name = str(getattr(simulator, "model_name", simulator.__class__.__name__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    compare_path = os.path.join(output_dir, f"robustness_{model_name.lower()}_{timestamp}.csv")
    with open(compare_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "scenario",
                "baseline_precision",
                "stressed_precision",
                "degradation_precision",
                "baseline_recall",
                "stressed_recall",
                "degradation_recall",
                "baseline_f1",
                "stressed_f1",
                "degradation_f1",
                "baseline_mttd",
                "stressed_mttd",
                "degradation_mttd",
                "baseline_fnr",
                "stressed_fnr",
                "degradation_fnr",
                "baseline_nmr",
                "stressed_nmr",
                "degradation_nmr",
                "noise_level",
                "drop_prob",
                "jitter_ms",
                "stress_seed",
            ]
        )

        for scenario in sorted(baseline.keys()):
            base = baseline.get(scenario, {})
            stress = stressed.get(scenario, {})
            writer.writerow(
                [
                    model_name,
                    scenario,
                    base.get("precision", math.nan),
                    stress.get("precision", math.nan),
                    stress.get("precision", math.nan) - base.get("precision", math.nan),
                    base.get("recall", math.nan),
                    stress.get("recall", math.nan),
                    stress.get("recall", math.nan) - base.get("recall", math.nan),
                    base.get("f1", math.nan),
                    stress.get("f1", math.nan),
                    stress.get("f1", math.nan) - base.get("f1", math.nan),
                    base.get("mttd", math.nan),
                    stress.get("mttd", math.nan),
                    stress.get("mttd", math.nan) - base.get("mttd", math.nan),
                    base.get("fnr", math.nan),
                    stress.get("fnr", math.nan),
                    stress.get("fnr", math.nan) - base.get("fnr", math.nan),
                    base.get("nmr", math.nan),
                    stress.get("nmr", math.nan),
                    stress.get("nmr", math.nan) - base.get("nmr", math.nan),
                    getattr(simulator, "stress_noise_level", 0.0),
                    getattr(simulator, "stress_drop_prob", 0.0),
                    getattr(simulator, "stress_jitter_ms", 0.0),
                    getattr(simulator, "stress_seed", 0),
                ]
            )

    summary_path = os.path.join(output_dir, "robustness_degradation_summary.csv")
    has_summary = os.path.exists(summary_path)

    f1_deltas = [stressed[s]["f1"] - baseline[s]["f1"] for s in baseline]
    precision_deltas = [stressed[s]["precision"] - baseline[s]["precision"] for s in baseline]
    recall_deltas = [stressed[s]["recall"] - baseline[s]["recall"] for s in baseline]
    mttd_deltas = [stressed[s]["mttd"] - baseline[s]["mttd"] for s in baseline]
    fnr_deltas = [stressed[s]["fnr"] - baseline[s]["fnr"] for s in baseline]
    nmr_deltas = [stressed[s]["nmr"] - baseline[s]["nmr"] for s in baseline]

    with open(summary_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not has_summary:
            writer.writerow(
                [
                    "timestamp",
                    "model",
                    "noise_level",
                    "drop_prob",
                    "jitter_ms",
                    "stress_seed",
                    "precision_degradation_mean",
                    "recall_degradation_mean",
                    "f1_degradation_mean",
                    "mttd_degradation_mean",
                    "fnr_degradation_mean",
                    "nmr_degradation_mean",
                ]
            )

        writer.writerow(
            [
                timestamp,
                model_name,
                getattr(simulator, "stress_noise_level", 0.0),
                getattr(simulator, "stress_drop_prob", 0.0),
                getattr(simulator, "stress_jitter_ms", 0.0),
                getattr(simulator, "stress_seed", 0),
                _safe_mean(precision_deltas),
                _safe_mean(recall_deltas),
                _safe_mean(f1_deltas),
                _safe_mean(mttd_deltas),
                _safe_mean(fnr_deltas),
                _safe_mean(nmr_deltas),
            ]
        )

    return compare_path
