#!/usr/bin/env python3
"""
Run robustness benchmark for all 7 fusion models under three stress conditions.

This wrapper runs each model script via subprocess with the same fixed seed
and stress parameters, collects per-run CSV outputs and aggregates a summary 
CSV containing the requested metrics.

Usage:
  python run_robustness_benchmark_all_models.py --seed 42 --output-dir results/robustness

"""
import argparse
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

MODELS = {
    "RB": "can_bus_simulation_Rule Based copy 5.py",
    "BF": "can_bus_simulation_Bayesian fusion.py",
    "DS": "can_bus_simulation_Dempster-Shafer fusion.py",
    "XGB": "can_bus_simulation_XGBoost copy.py",
    "SMLP": "can_bus_simulation_SmallMLP.py",
    "TS": "can_bus_simulation_Sequence.py",
    "HB": "can_bus_simulation_hybrid.py",
}

CONDITIONS = {
    "baseline": {"noise_level": 0.0, "drop_prob": 0.0, "jitter_ms": 0},
    "moderate": {"noise_level": 0.10, "drop_prob": 0.05, "jitter_ms": 25},
    "severe": {"noise_level": 0.20, "drop_prob": 0.10, "jitter_ms": 40},
}

METRICS = ["Precision", "Recall", "F1", "Full_Brake_FNR", "MTTD", "NMR"]


def run_model_headless(model_code: str, script_path: Path, seed: int, condition_name: str, 
                      params: Dict, out_dir: Path, timeout_seconds: int) -> bool:
    """Run model using subprocess with proper event loop handling."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running {model_code} [{condition_name}] (seed={seed})")
    
    # Build command to run model with all required parameters
    cmd = [
        sys.executable, str(script_path),
        "--seed", str(seed),
        "--headless",
        "--stress-noise-level", str(params["noise_level"]),
        "--stress-drop-prob", str(params["drop_prob"]),
        "--stress-jitter-ms", str(params["jitter_ms"]),
        "--stress-seed", str(seed),
        "--output-dir", str(out_dir),
    ]
    
    try:
        result = subprocess.run(cmd, timeout=timeout_seconds, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Completed {model_code} [{condition_name}]")
            return True
        else:
            print(f"Failed {model_code} [{condition_name}]: exit code {result.returncode}")
            if result.stderr:
                print(f"  Error: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"Timeout running {model_code} [{condition_name}]")
        return False
    except Exception as e:
        print(f"Exception running {model_code} [{condition_name}]: {e}")
        return False


def find_result_csv(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return list(folder.rglob("*.csv"))


def aggregate_results(root_out: Path, summary_path: Path):
    """Create two summary files: raw metrics and degradation metrics (properly separated)."""
    raw_rows = []
    degr_rows = []
    
    raw_col_map = {
        "Precision": "stressed_precision",
        "Recall": "stressed_recall",
        "F1": "stressed_f1",
        "Full_Brake_FNR": "stressed_fnr",
        "MTTD": "stressed_mttd",
        "NMR": "stressed_nmr",
    }
    degr_col_map = {
        "Precision": "precision_degradation_mean",
        "Recall": "recall_degradation_mean",
        "F1": "f1_degradation_mean",
        "Full_Brake_FNR": "fnr_degradation_mean",
        "MTTD": "mttd_degradation_mean",
        "NMR": "nmr_degradation_mean",
    }
    
    for model_code in MODELS:
        for cond in CONDITIONS:
            folder = root_out / model_code / cond
            csvs = find_result_csv(folder)
            if not csvs:
                continue
            
            # Find per-run and degradation CSVs.
            # Model scripts use different filenames, so prefer the newest non-summary CSV.
            per_run_candidates = [
                c for c in csvs
                if c.name.lower().endswith(".csv")
                and "degradation" not in c.name.lower()
                and "summary" not in c.name.lower()
            ]
            degr_candidates = [
                c for c in csvs
                if "degradation_summary" in c.name.lower()
            ]
            per_run_csv = max(per_run_candidates, key=lambda path: path.stat().st_mtime, default=None)
            degr_csv = max(degr_candidates, key=lambda path: path.stat().st_mtime, default=None)
            
            # Extract RAW metrics from per-run CSV (average of stressed_* columns)
            if per_run_csv:
                raw_metrics = {k: "" for k in METRICS}
                try:
                    with open(per_run_csv, newline='', encoding='utf-8') as fh:
                        rdr = csv.DictReader(fh)
                        data_rows = list(rdr)
                        if data_rows:
                            # Compute mean of stressed_* columns across scenarios
                            for metric_name, csv_col in raw_col_map.items():
                                values = []
                                for row in data_rows:
                                    if csv_col in row and row[csv_col] != "" and row[csv_col].lower() != "nan":
                                        try:
                                            values.append(float(row[csv_col]))
                                        except (ValueError, TypeError):
                                            pass
                                if values:
                                    raw_metrics[metric_name] = sum(values) / len(values)
                except Exception as e:
                    print(f"Failed to extract raw metrics from {per_run_csv}: {e}")
                
                raw_rows.append({"model": model_code, "condition": cond, **raw_metrics})
            
            # Extract DEGRADATION metrics from degradation summary
            if degr_csv:
                degr_metrics = {k: "" for k in METRICS}
                try:
                    with open(degr_csv, newline='', encoding='utf-8') as fh:
                        rdr = csv.DictReader(fh)
                        data_rows = list(rdr)
                        if data_rows:
                            row = data_rows[0]
                            for metric_name, csv_col in degr_col_map.items():
                                if csv_col in row and row[csv_col] != "" and row[csv_col].lower() != "nan":
                                    try:
                                        degr_metrics[metric_name] = float(row[csv_col])
                                    except (ValueError, TypeError):
                                        pass
                except Exception as e:
                    print(f"Failed to extract degradation metrics from {degr_csv}: {e}")
                
                degr_rows.append({"model": model_code, "condition": cond, **degr_metrics})
    
    # Write RAW metrics to summary_path
    if raw_rows:
        with open(summary_path, 'w', newline='', encoding='utf-8') as outfh:
            fieldnames = ["model", "condition"] + METRICS
            w = csv.DictWriter(outfh, fieldnames=fieldnames)
            w.writeheader()
            for r in raw_rows:
                w.writerow(r)
        print(f"Wrote {len(raw_rows)} raw metric rows to {summary_path}")
    
    # Write DEGRADATION metrics to parallel file with _degradation_mean column names
    if degr_rows:
        seed_id = summary_path.stem.split("_")[-1]
        degr_path = summary_path.parent / f"robustness_degradation_summary_seed_{seed_id}.csv"
        with open(degr_path, 'w', newline='', encoding='utf-8') as outfh:
            # Use _degradation_mean suffix for column names so aggregator can classify them
            metric_col_renames = {
                "Precision": "precision_degradation_mean",
                "Recall": "recall_degradation_mean",
                "F1": "f1_degradation_mean",
                "Full_Brake_FNR": "fnr_degradation_mean",
                "MTTD": "mttd_degradation_mean",
                "NMR": "nmr_degradation_mean",
            }
            fieldnames = ["model", "condition"] + [metric_col_renames[m] for m in METRICS]
            w = csv.DictWriter(outfh, fieldnames=fieldnames)
            w.writeheader()
            for r in degr_rows:
                # Rename keys in the row to match degradation column names
                renamed_row = {"model": r["model"], "condition": r["condition"]}
                for metric_name, col_name in metric_col_renames.items():
                    renamed_row[col_name] = r[metric_name]
                w.writerow(renamed_row)
        print(f"Wrote {len(degr_rows)} degradation metric rows to {degr_path}")


def build_parser():
    p = argparse.ArgumentParser(description="Run robustness benchmark across multiple model scripts")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-seeds", type=int, default=1, help="Number of consecutive seeds to run")
    p.add_argument("--models", type=str, default=','.join(MODELS.keys()),
                   help="Comma-separated model codes to run (RB,BF,DS,XGB,SMLP,TS,HB)")
    p.add_argument("--output-dir", type=str, default="results/robustness")
    p.add_argument("--timeout", type=int, default=300, help="Timeout per run (seconds)")
    return p


def build_seed_list(seed_start: int, num_seeds: int):
    if num_seeds < 1:
        raise ValueError("--num-seeds must be at least 1")
    return [seed_start + offset for offset in range(num_seeds)]


def run_single_seed(seed: int, chosen: list[str], out_root: Path, timeout: int):
    failures = []
    runs_log = []
    tasks = []

    for model_code in chosen:
        if model_code not in MODELS:
            print(f"Unknown model code: {model_code}, skipping")
            continue
        script = Path(MODELS[model_code])
        if not script.exists():
            print(f"Script not found: {script} (skipping {model_code})")
            continue

        for cond_name, params in CONDITIONS.items():
            run_out = out_root / model_code / cond_name
            tasks.append((model_code, cond_name, script, params, run_out))

    if tasks:
        max_workers = min(7, len(tasks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    run_model_headless,
                    model_code,
                    script,
                    seed,
                    cond_name,
                    params,
                    run_out,
                    timeout,
                ): (model_code, cond_name)
                for model_code, cond_name, script, params, run_out in tasks
            }

            for future in as_completed(future_map):
                model_code, cond_name = future_map[future]
                try:
                    success = bool(future.result())
                except Exception as e:
                    print(f"Exception running {model_code} [{cond_name}]: {e}")
                    success = False

                runs_log.append({
                    'seed': seed,
                    'model': model_code,
                    'condition': cond_name,
                    'success': success,
                })
                if not success:
                    failures.append((model_code, cond_name))

    summary = out_root / f"robustness_summary_seed_{seed}.csv"
    aggregate_results(out_root, summary)

    try:
        import json
        with open(out_root / 'benchmark_log.json', 'w', encoding='utf-8') as fh:
            json.dump({'seed': seed, 'runs': runs_log}, fh, indent=2)
    except Exception:
        pass

    return failures, runs_log, summary


def main():
    args = build_parser().parse_args()
    chosen = [m.strip() for m in args.models.split(',') if m.strip()]
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    seed_list = build_seed_list(args.seed, args.num_seeds)
    all_failures = []

    for seed in seed_list:
        seed_out_root = out_root if len(seed_list) == 1 else out_root / f"seed_{seed}"
        seed_out_root.mkdir(parents=True, exist_ok=True)
        failures, _, summary = run_single_seed(seed, chosen, seed_out_root, args.timeout)
        all_failures.extend([(seed, model_code, cond_name) for model_code, cond_name in failures])
        print(f"\nBenchmark complete for seed {seed}. Summary written to {summary}")

    if all_failures:
        print("\nSome runs failed:")
        for seed, model_code, cond_name in all_failures:
            print(f"  seed {seed}: {model_code} [{cond_name}]")
        sys.exit(2)


if __name__ == '__main__':
    main()
