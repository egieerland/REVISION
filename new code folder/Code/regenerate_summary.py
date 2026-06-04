#!/usr/bin/env python3
"""Regenerate the summary CSV with fixed aggregation logic."""
from pathlib import Path
from run_robustness_benchmark_all_models import aggregate_results
import csv

out_root = Path('results/final_validation')
summary = out_root / 'robustness_summary_seed_42.csv'

print("Regenerating summary with fixed column mapping...")
aggregate_results(out_root, summary)
print(f"✓ Summary regenerated: {summary}\n")

# Show results
if summary.exists():
    with open(summary) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        print(f"Total rows: {len(rows)}")
        print("\nFirst 3 rows:")
        for row in rows[:3]:
            print(f"  {row}")
        print("\nLast 3 rows:")
        for row in rows[-3:]:
            print(f"  {row}")
