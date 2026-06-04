Robustness benchmark wrapper
===========================

Quick run (all models, seed=42)
------------------------------

```bash
python run_robustness_benchmark_all_models.py --seed 42 --output-dir results/robustness
```

Example run for a single model (debug):

```bash
python run_robustness_benchmark_all_models.py --seed 42 --models XGB --output-dir results/robustness_quick
```

Options
-------
- `--seed`: integer seed used for all runs (default 42)
- `--models`: comma separated model codes (RB,BF,DS,XGB,SMLP,TS,HB)
- `--output-dir`: directory where per-run outputs and summary will be written

Expected output structure
-------------------------

After a full run the output directory will look like:

```
results/robustness/
├── RB/
│   ├── baseline/
│   │   └── ...per-run CSVs...
│   ├── moderate/
│   └── severe/
├── BF/...
├── DS/...
├── XGB/...
├── SMLP/...
├── TS/...
└── HB/...
robustness_summary_seed_<SEED>.csv
```

Each model/condition folder should contain the model's per-run CSV outputs. The wrapper
will try to locate `robustness_degradation_summary.csv` inside each folder; if not present
it will use the first CSV found and compute average values for the requested metrics.

Stress conditions
-----------------
- Baseline: `noise_level=0.00`, `drop_prob=0.00`, `jitter_ms=0`
- Moderate: `noise_level=0.10`, `drop_prob=0.05`, `jitter_ms=25`
- Severe:   `noise_level=0.20`, `drop_prob=0.10`, `jitter_ms=40`

Metrics collected/aggregated
----------------------------
- `Precision`, `Recall`, `F1`, `Full_Brake_FNR`, `MTTD`, `NMR`

Notes & troubleshooting
-----------------------
- The wrapper calls each simulator script as a subprocess in headless mode and passes
  stress CLI flags. If a simulator script uses different CLI names, run that model
  individually (see example above) and adapt the script to accept the flags or update
  the wrapper to match.
- To validate outputs locally, list CSV files and inspect the aggregated summary:

```bash
ls -R results/robustness | sed -n '1,200p'
python -c "import csv;print(next(csv.DictReader(open('results/robustness/robustness_summary_seed_42.csv'))))"
```
