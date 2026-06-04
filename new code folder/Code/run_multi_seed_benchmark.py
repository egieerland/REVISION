"""Run multi-seed benchmarks for CAN bus simulators (headless)."""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import time
import tkinter as tk
from typing import Dict, Iterable, List

from metrics_utils import build_summary_metrics, write_metrics_row_csv
from seed_utils import seed_everything

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_SPECS: Dict[str, Dict[str, str]] = {
    "xgboost": {"file": "can_bus_simulation_XGBoost copy.py", "label": "XGBoost"},
    "smallmlp": {"file": "can_bus_simulation_SmallMLP.py", "label": "SmallMLP"},
    "sequence": {"file": "can_bus_simulation_Sequence.py", "label": "Sequence"},
    "hybrid": {"file": "can_bus_simulation_hybrid.py", "label": "Hybrid"},
}


def safe_filename(name: str) -> str:
    """Return a filesystem-safe name."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return cleaned or "scenario"


def load_module(module_path: str, module_name: str):
    """Load a Python module from a path."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_models(value: str) -> List[str]:
    if not value:
        return list(MODEL_SPECS.keys())
    models = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = [model for model in models if model not in MODEL_SPECS]
    if unknown:
        raise ValueError(f"Unknown models: {', '.join(unknown)}")
    return models


def parse_seeds(seed_list: str, seed_start: int, num_seeds: int) -> List[int]:
    if seed_list:
        return [int(part.strip()) for part in seed_list.split(",") if part.strip()]
    return [seed_start + offset for offset in range(num_seeds)]


def wait_for_scenario(simulator, root: tk.Tk, duration: float, buffer_s: float, step_s: float) -> None:
    timeout = max(1.0, duration + buffer_s)
    start = time.time()
    while simulator.scenario_active and (time.time() - start) < timeout:
        try:
            root.update()
        except tk.TclError:
            break
        time.sleep(step_s)

    if simulator.scenario_active:
        simulator.stop_scenario()
        try:
            root.update()
        except tk.TclError:
            pass

    time.sleep(step_s)


def run_model(
    model_key: str,
    seeds: Iterable[int],
    output_dir: str,
    buffer_s: float,
    step_s: float,
) -> None:
    spec = MODEL_SPECS[model_key]
    module_path = os.path.join(BASE_DIR, spec["file"])
    module_name = f"sim_{safe_filename(model_key)}"
    module = load_module(module_path, module_name)
    simulator_class = getattr(module, "CANBusSimulator")

    model_out_dir = os.path.join(output_dir, model_key)
    os.makedirs(model_out_dir, exist_ok=True)


    for seed in seeds:
        seed_everything(seed)
        root = tk.Tk()
        root.withdraw()
        simulator = simulator_class(root, seed=seed, headless_mode=True)
        time.sleep(step_s)

        scenarios = [name for name in simulator.scenarios.keys() if name != "Normal Driving"]
        for scenario in scenarios:
            simulator.scenario_var.set(scenario)
            simulator.start_scenario()

            duration = float(simulator.scenarios[scenario].get("duration", 0) or 0)
            wait_for_scenario(simulator, root, duration, buffer_s, step_s)

            priority_metrics = simulator.calculate_priority_metrics()
            summary = build_summary_metrics(priority_metrics)
            scenario_safe = safe_filename(scenario)
            filename = f"metrics_{model_key}_{scenario_safe}_seed{seed}.csv"
            out_path = os.path.join(model_out_dir, filename)
            write_metrics_row_csv(out_path, summary, simulator.model_name, scenario, seed)

        try:
            simulator.on_closing()
        except Exception:
            simulator.running = False
            try:
                root.destroy()
            except tk.TclError:
                pass


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multi-seed CAN bus benchmarks")
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(MODEL_SPECS.keys()),
        help="Comma-separated list of models to run",
    )
    parser.add_argument("--seeds", type=str, default="", help="Comma-separated seed list")
    parser.add_argument("--seed-start", type=int, default=0, help="First seed (used with --num-seeds)")
    parser.add_argument("--num-seeds", type=int, default=10, help="Number of seeds to run")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(BASE_DIR, "results", "multi_seed"),
        help="Directory for per-seed metrics",
    )
    parser.add_argument("--timeout-buffer", type=float, default=5.0, help="Extra seconds after scenario duration")
    parser.add_argument("--step-s", type=float, default=0.05, help="Sleep step for polling loop")
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    models = parse_models(args.models)
    seeds = parse_seeds(args.seeds, args.seed_start, args.num_seeds)

    os.makedirs(args.output_dir, exist_ok=True)

    for model_key in models:
        run_model(model_key, seeds, args.output_dir, args.timeout_buffer, args.step_s)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
