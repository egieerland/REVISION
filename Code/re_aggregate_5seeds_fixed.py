#!/usr/bin/env python
"""Re-aggregate results with corrected metric mapping that includes precision/recall."""

import csv
import sys
from pathlib import Path

MODELS = {
    "RB": "can_bus_simulation_Rule Based copy 5.py",
    "BF": "can_bus_simulation_Bayesian fusion.py",
    "DS": "can_bus_simulation_Dempster-Shafer fusion.py",
    "XGB": "can_bus_simulation_XGBoost copy.py",
    "SMLP": "can_bus_simulation_SmallMLP.py",
    "TS": "can_bus_simulation_Sequence.py",
    "HB": "can_bus_simulation_hybrid.py",
}

CONDITIONS = ["baseline", "moderate", "severe"]
METRICS = ["Precision", "Recall", "F1", "Full_Brake_FNR", "MTTD", "NMR"]


def find_result_csv(folder: Path):
    """Find result CSV files in a condition folder."""
    if not folder.exists():
        return []
    return list(folder.glob("robustness_*.csv"))


def aggregate_results_corrected(root_out: Path, summary_path: Path):
    """Aggregate results with corrected metric column mapping."""
    # CORRECTED mapping that includes precision/recall degradation
    metric_col_map = {
        "Precision": "precision_degradation_mean",
        "Recall": "recall_degradation_mean",
        "F1": "f1_degradation_mean",
        "Full_Brake_FNR": "fnr_degradation_mean",
        "MTTD": "mttd_degradation_mean",
        "NMR": "nmr_degradation_mean",
    }
    
    rows = []
    for model_code in MODELS:
        for cond in CONDITIONS:
            folder = root_out / model_code / cond
            csvs = find_result_csv(folder)
            if not csvs:
                print(f"  Skipping {model_code}/{cond}: no CSVs found")
                continue
            
            # Prefer robustness_degradation_summary.csv
            chosen = None
            for c in csvs:
                if c.name == "robustness_degradation_summary.csv":
                    chosen = c
                    break
            if chosen is None:
                chosen = csvs[0]
            
            metrics = {k: "" for k in METRICS}
            try:
                with open(chosen, newline='', encoding='utf-8') as fh:
                    rdr = csv.DictReader(fh)
                    data_rows = list(rdr)
                    if data_rows:
                        row = data_rows[0]
                        for metric_name, csv_col in metric_col_map.items():
                            if csv_col in row and row[csv_col] != "" and row[csv_col].lower() != "nan":
                                try:
                                    metrics[metric_name] = float(row[csv_col])
                                except (ValueError, TypeError):
                                    metrics[metric_name] = ""
                print(f"  {model_code}/{cond}: {metrics}")
            except Exception as e:
                print(f"Failed to parse {chosen}: {e}")
                continue
            
            rows.append({"model": model_code, "condition": cond, **metrics})
    
    # Write summary CSV
    if rows:
        with open(summary_path, 'w', newline='', encoding='utf-8') as outfh:
            fieldnames = ["model", "condition"] + METRICS
            w = csv.DictWriter(outfh, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nWrote {len(rows)} rows to {summary_path}")
    else:
        print("No results found to aggregate.")


if __name__ == "__main__":
    # Process each seed
    root = Path("results/final_validation_5seeds_fixed")
    for seed_folder in sorted(root.glob("seed_*")):
        print(f"\nProcessing {seed_folder.name}...")
        seed_num = seed_folder.name.replace("seed_", "")
        summary_path = seed_folder / f"robustness_summary_seed_{seed_num}.csv"
        aggregate_results_corrected(seed_folder, summary_path)
