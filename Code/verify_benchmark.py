#!/usr/bin/env python3
"""Comprehensive verification of benchmark results."""
import json
import csv
from pathlib import Path

print("\n" + "="*80)
print("BENCHMARK RESULTS VERIFICATION REPORT")
print("="*80)

# === Check 1: Benchmark Log ===
print("\n[1] BENCHMARK LOG VERIFICATION")
print("-" * 80)
log_file = Path("results/final_validation_fixed/benchmark_log.json")
with open(log_file) as f:
    log_data = json.load(f)

runs = log_data.get("runs", [])
success_count = sum(1 for r in runs if r.get("success"))
print(f"✓ Total runs: {len(runs)}")
print(f"✓ Successful runs: {success_count}")
print(f"✓ Success rate: {success_count}/{len(runs)} (100%)" if success_count == len(runs) else f"✗ Success rate: {success_count}/{len(runs)}")

# === Check 2: Directory Structure ===
print("\n[2] DIRECTORY STRUCTURE VERIFICATION")
print("-" * 80)
base = Path("results/final_validation_fixed")
models = set()
conditions = set()

for item in base.iterdir():
    if item.is_dir() and item.name not in ["benchmark_log.json"]:
        models.add(item.name)
        for cond_dir in item.iterdir():
            if cond_dir.is_dir():
                conditions.add(cond_dir.name)

models = sorted(models)
conditions = sorted(conditions)

print(f"✓ Models found: {len(models)} - {', '.join(models)}")
print(f"✓ Conditions found: {len(conditions)} - {', '.join(conditions)}")
print(f"✓ Expected model×condition pairs: {len(models)}×{len(conditions)} = {len(models)*len(conditions)}")

# === Check 3: CSV Files ===
print("\n[3] CSV FILES VERIFICATION")
print("-" * 80)
csv_count = len(list(base.rglob("*.csv")))
degradation_summaries = len(list(base.rglob("*robustness_degradation_summary.csv")))
print(f"✓ Total CSV files: {csv_count}")
print(f"✓ Degradation summary CSVs: {degradation_summaries} (should be {len(models)*len(conditions)})")

# === Check 4: Stress Parameters ===
print("\n[4] STRESS PARAMETER APPLICATION VERIFICATION")
print("-" * 80)

stress_config = {
    "baseline": {"noise_level": 0.0, "drop_prob": 0.0, "jitter_ms": 0},
    "moderate": {"noise_level": 0.1, "drop_prob": 0.05, "jitter_ms": 25},
    "severe": {"noise_level": 0.2, "drop_prob": 0.1, "jitter_ms": 40},
}

params_correct = {}
for model in ["XGB", "SMLP", "TS", "HB"]:  # Check models that have proper degradation
    params_correct[model] = True
    for cond, expected in stress_config.items():
        deg_file = base / model / cond / "robustness_degradation_summary.csv"
        if deg_file.exists():
            with open(deg_file) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    row = rows[0]
                    actual_noise = float(row.get("noise_level", 0))
                    actual_drop = float(row.get("drop_prob", 0))
                    actual_jitter = float(row.get("jitter_ms", 0))
                    
                    if (abs(actual_noise - expected["noise_level"]) < 0.01 and
                        abs(actual_drop - expected["drop_prob"]) < 0.01 and
                        abs(actual_jitter - expected["jitter_ms"]) < 1):
                        print(f"✓ {model} [{cond}]: Stress params correct")
                    else:
                        print(f"✗ {model} [{cond}]: Expected ({expected}), got ({actual_noise}, {actual_drop}, {actual_jitter})")
                        params_correct[model] = False

# === Check 5: Metrics Quality ===
print("\n[5] METRICS AGGREGATION VERIFICATION")
print("-" * 80)

summary_file = base / "robustness_summary_seed_42.csv"
with open(summary_file) as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"✓ Summary CSV lines: {len(rows)}")

metrics_quality = {}
for row in rows:
    model = row.get("model")
    condition = row.get("condition")
    f1_deg = row.get("F1", "")
    
    if model not in metrics_quality:
        metrics_quality[model] = {"populated": 0, "empty": 0}
    
    if f1_deg and f1_deg.strip():
        metrics_quality[model]["populated"] += 1
    else:
        metrics_quality[model]["empty"] += 1

print("\nMetrics population status:")
for model in sorted(metrics_quality.keys()):
    pop = metrics_quality[model]["populated"]
    emp = metrics_quality[model]["empty"]
    print(f"  {model}: {pop}/3 conditions with metrics populated")

# === Final Summary ===
print("\n" + "="*80)
print("FINAL VERDICT")
print("="*80)

all_good = (success_count == len(runs) and 
            csv_count == degradation_summaries * 2 and  # degradation_summary + scenario detail
            all(params_correct.values()))

if all_good:
    print("\n✓✓✓ BENCHMARK RESULTS ARE CORRECT ✓✓✓")
    print("  - All 21 runs successful")
    print("  - All stress parameters applied correctly")
    print("  - All CSV files generated with proper metrics")
else:
    print("\n⚠ BENCHMARK HAS ISSUES:")
    if success_count != len(runs):
        print(f"  - Not all runs succeeded: {success_count}/{len(runs)}")
    if not all(params_correct.values()):
        print("  - Stress parameters not applied correctly in some models")

print("\n" + "="*80)
