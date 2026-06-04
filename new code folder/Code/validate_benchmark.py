#!/usr/bin/env python3
"""Validate benchmark outputs and print final report."""
import json
from pathlib import Path
from collections import defaultdict

benchmark_dir = Path('results/final_validation')
log_file = benchmark_dir / 'benchmark_log.json'

# Read the benchmark log
with open(log_file) as f:
    log_data = json.load(f)

runs = log_data.get('runs', [])
success_count = sum(1 for r in runs if r.get('success'))
total_count = len(runs)

print("\n" + "="*70)
print("FINAL ROBUSTNESS BENCHMARK VALIDATION RESULTS")
print("="*70)
print(f"\nTotal Runs: {total_count}")
print(f"Successful Runs: {success_count}")
print(f"Success Rate: {success_count}/{total_count} (100%)" if success_count == total_count else f"Success Rate: {success_count}/{total_count}")

# List CSV outputs by model
print("\nGenerated CSV Artifacts:")
print("-" * 70)

model_counts = defaultdict(lambda: defaultdict(int))

for csv_file in sorted(benchmark_dir.rglob('*.csv')):
    if 'robustness_summary_seed' not in csv_file.name:
        parts = csv_file.relative_to(benchmark_dir).parts
        if len(parts) >= 3:
            model = parts[0]
            condition = parts[1]
            model_counts[model][condition] += 1

print(f"{'Model':<8} {'Baseline':<12} {'Moderate':<12} {'Severe':<12} {'Total':<8}")
print("-" * 70)
for model in sorted(model_counts.keys()):
    baseline = model_counts[model].get('baseline', 0)
    moderate = model_counts[model].get('moderate', 0)
    severe = model_counts[model].get('severe', 0)
    total = baseline + moderate + severe
    print(f"{model:<8} {baseline:<12} {moderate:<12} {severe:<12} {total:<8}")

print("-" * 70)
total_csvs = sum(sum(c.values()) for c in model_counts.values())
print(f"{'TOTAL':<8} {'':<12} {'':<12} {'':<12} {total_csvs:<8}")

# Check for summary CSV
summary_file = benchmark_dir / 'robustness_summary_seed_42.csv'
print(f"\nAggregated Summary: {summary_file}")
print(f"  File exists: {summary_file.exists()}")
if summary_file.exists():
    with open(summary_file) as f:
        lines = f.readlines()
        print(f"  Total lines: {len(lines)}")
        if lines:
            print(f"  Header: {lines[0].strip()}")

print("\nOutput Directory:")
print(f"  {benchmark_dir.absolute()}")

# Per-condition file paths
print("\nSample Output Files:")
print("-" * 70)
for model in sorted(model_counts.keys())[:2]:  # Show first 2 models
    for cond in ['baseline', 'moderate', 'severe']:
        cond_dir = benchmark_dir / model / cond
        if cond_dir.exists():
            csvs = list(cond_dir.glob('*.csv'))
            if csvs:
                rel_path = csvs[0].relative_to(Path.cwd())
                print(f"  {rel_path}")

print("="*70 + "\n")
