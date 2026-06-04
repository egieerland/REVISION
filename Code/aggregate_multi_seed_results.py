"""Aggregate per-seed benchmark CSVs into explicit raw and degradation summaries."""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import re
import zlib
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

RAW_METRIC_FIELDS = {
    "precision": ("Precision", "precision_avg"),
    "recall": ("Recall", "recall_avg"),
    "f1": ("F1", "f1_avg"),
    "mttd": ("MTTD", "mttd_mean", "mttd_mean_s"),
    "fnr": ("Full_Brake_FNR", "fnr_full_brake"),
    "nmr": ("NMR", "nmr"),
}

DEGRADATION_METRIC_FIELDS = {
    "precision": ("precision_degradation_mean",),
    "recall": ("recall_degradation_mean",),
    "f1": ("f1_degradation_mean",),
    "mttd": ("mttd_degradation_mean",),
    "fnr": ("fnr_degradation_mean",),
    "nmr": ("nmr_degradation_mean",),
}

CONDITION_ALIASES = {
    (0.0, 0.0, 0.0): "baseline",
    (0.10, 0.05, 25.0): "moderate",
    (0.2, 0.1, 40.0): "severe",
}


def parse_float(value: object) -> float:
    if value is None:
        return math.nan
    text = str(value).strip().lower()
    if text in ("", "nan", "none"):
        return math.nan
    return float(text)


def derive_seed(base: int, *parts: str) -> int:
    payload = "|".join(parts).encode("utf-8")
    return base + (zlib.crc32(payload) & 0xFFFFFFFF)


def bootstrap_ci(values: List[float], n_boot: int, seed: int) -> Dict[str, float]:
    clean = [value for value in values if not math.isnan(value)]
    if not clean:
        return {"mean": math.nan, "std": math.nan, "ci_lower": math.nan, "ci_upper": math.nan, "n": 0}

    mean = float(np.mean(clean))
    std = float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0

    rng = random.Random(seed)
    boot_means = []
    for _ in range(n_boot):
        sample = [rng.choice(clean) for _ in range(len(clean))]
        boot_means.append(float(np.mean(sample)))

    boot_means.sort()
    lower_idx = int(0.025 * (len(boot_means) - 1))
    upper_idx = int(0.975 * (len(boot_means) - 1))

    return {
        "mean": mean,
        "std": std,
        "ci_lower": boot_means[lower_idx],
        "ci_upper": boot_means[upper_idx],
        "n": len(clean),
    }


def classify_row(row: Dict[str, str]) -> Optional[str]:
    keys = set(row.keys())
    if any(field in keys for field in DEGRADATION_METRIC_FIELDS["f1"]):
        return "degradation"
    if any(field in keys for field in RAW_METRIC_FIELDS["f1"]):
        return "raw"
    if any(field in keys for field in RAW_METRIC_FIELDS["precision"]):
        return "raw"
    return None


def condition_from_row(row: Dict[str, str]) -> str:
    condition = (row.get("condition") or row.get("scenario") or "").strip()
    if condition:
        return condition

    try:
        noise = round(parse_float(row.get("noise_level")), 2)
        drop = round(parse_float(row.get("drop_prob")), 2)
        jitter = round(parse_float(row.get("jitter_ms")), 1)
        return CONDITION_ALIASES.get((noise, drop, jitter), "unknown")
    except Exception:
        return "unknown"


def load_rows(input_dir: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if not filename.lower().endswith(".csv"):
                continue
            path = os.path.join(root, filename)
            with open(path, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or "model" not in reader.fieldnames:
                    continue
                for row in reader:
                    kind = classify_row(row)
                    if kind is None:
                        continue
                    row["_kind"] = kind
                    row["_source"] = filename
                    rows.append(row)
    return rows


def summarize_group(
    rows: Iterable[Dict[str, str]],
    metric_fields: Dict[str, Tuple[str, ...]],
    n_boot: int,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, List[float]] = {name: [] for name in metric_fields}

    for row in rows:
        for metric_name, candidates in metric_fields.items():
            value = math.nan
            for candidate in candidates:
                if candidate in row and str(row[candidate]).strip() not in ("", "nan", "none"):
                    try:
                        value = parse_float(row[candidate])
                        break
                    except (TypeError, ValueError):
                        continue
            metrics[metric_name].append(value)

    summary: Dict[str, Dict[str, float]] = {}
    for metric_name, values in metrics.items():
        summary[metric_name] = bootstrap_ci(values, n_boot, derive_seed(seed, metric_name))
    return summary


def group_rows(rows: List[Dict[str, str]], kind: str, use_condition: bool) -> Dict[str, List[Dict[str, str]]]:
    groups: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        if row.get("_kind") != kind:
            continue
        model = row.get("model", "")
        condition = condition_from_row(row) if use_condition else "overall"
        key = f"{model}::{condition}" if use_condition else model
        groups.setdefault(key, []).append(row)
    return groups


def write_summary_csv(
    rows: List[Dict[str, str]],
    output_path: str,
    kind: str,
    metric_fields: Dict[str, Tuple[str, ...]],
    use_condition: bool,
) -> None:
    groups = group_rows(rows, kind, use_condition)
    metric_prefix = "" if kind == "raw" else "degradation_"

    headers = ["model"]
    if use_condition:
        headers.append("condition")
    headers.extend([
        f"{metric_prefix}precision_mean",
        f"{metric_prefix}precision_std",
        f"{metric_prefix}precision_ci_lower",
        f"{metric_prefix}precision_ci_upper",
        f"{metric_prefix}precision_count",
        f"{metric_prefix}recall_mean",
        f"{metric_prefix}recall_std",
        f"{metric_prefix}recall_ci_lower",
        f"{metric_prefix}recall_ci_upper",
        f"{metric_prefix}recall_count",
        f"{metric_prefix}f1_mean",
        f"{metric_prefix}f1_std",
        f"{metric_prefix}f1_ci_lower",
        f"{metric_prefix}f1_ci_upper",
        f"{metric_prefix}f1_count",
        f"{metric_prefix}mttd_mean",
        f"{metric_prefix}mttd_std",
        f"{metric_prefix}mttd_ci_lower",
        f"{metric_prefix}mttd_ci_upper",
        f"{metric_prefix}mttd_count",
        f"{metric_prefix}fnr_mean",
        f"{metric_prefix}fnr_std",
        f"{metric_prefix}fnr_ci_lower",
        f"{metric_prefix}fnr_ci_upper",
        f"{metric_prefix}fnr_count",
        f"{metric_prefix}nmr_mean",
        f"{metric_prefix}nmr_std",
        f"{metric_prefix}nmr_ci_lower",
        f"{metric_prefix}nmr_ci_upper",
        f"{metric_prefix}nmr_count",
        "n",
    ])

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)

        for key, grouped_rows in sorted(groups.items()):
            if use_condition:
                model, condition = key.split("::", 1)
            else:
                model, condition = key, None
            summary = summarize_group(grouped_rows, metric_fields, 1000, derive_seed(0, kind, key))

            precision = summary["precision"]
            recall = summary["recall"]
            f1 = summary["f1"]
            mttd = summary["mttd"]
            fnr = summary["fnr"]
            nmr = summary["nmr"]

            row = [
                model,
            ]
            if use_condition:
                row.append(condition)
            row.extend([
                precision["mean"],
                precision["std"],
                precision["ci_lower"],
                precision["ci_upper"],
                precision["n"],
                recall["mean"],
                recall["std"],
                recall["ci_lower"],
                recall["ci_upper"],
                recall["n"],
                f1["mean"],
                f1["std"],
                f1["ci_lower"],
                f1["ci_upper"],
                f1["n"],
                mttd["mean"],
                mttd["std"],
                mttd["ci_lower"],
                mttd["ci_upper"],
                mttd["n"],
                fnr["mean"],
                fnr["std"],
                fnr["ci_lower"],
                fnr["ci_upper"],
                fnr["n"],
                nmr["mean"],
                nmr["std"],
                nmr["ci_lower"],
                nmr["ci_upper"],
                nmr["n"],
                f1["n"],
            ])
            writer.writerow(row)


def write_table_csv(summary_path: str, table_path: str, kind: str, use_condition: bool) -> None:
    with open(summary_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    metric_prefix = "" if kind == "raw" else "degradation_"
    headers = ["model"]
    if use_condition:
        headers.append("condition")
    headers.extend([
        f"{metric_prefix}precision_mean_ci",
        f"{metric_prefix}recall_mean_ci",
        f"{metric_prefix}f1_mean_ci",
        f"{metric_prefix}mttd_mean_ci",
        f"{metric_prefix}fnr_mean_ci",
        f"{metric_prefix}nmr_mean_ci",
    ])

    with open(table_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)

        for row in rows:
            def fmt(metric: str) -> str:
                mean = parse_float(row.get(f"{metric_prefix}{metric}_mean"))
                lo = parse_float(row.get(f"{metric_prefix}{metric}_ci_lower"))
                hi = parse_float(row.get(f"{metric_prefix}{metric}_ci_upper"))
                if math.isnan(mean):
                    return "nan"
                return f"{mean:.4f} ({lo:.4f}, {hi:.4f})"

            output_row = [row.get("model", "")]
            if use_condition:
                output_row.append(row.get("condition", ""))
            output_row.extend([fmt("precision"), fmt("recall"), fmt("f1"), fmt("mttd"), fmt("fnr"), fmt("nmr")])
            writer.writerow(output_row)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed metrics")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "multi_seed"),
        help="Directory with per-seed CSVs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "aggregated"),
        help="Directory for summary CSVs",
    )
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap resamples for CI")
    parser.add_argument("--seed", type=int, default=0, help="Seed for bootstrap sampling")
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    rows = load_rows(args.input_dir)
    if not rows:
        raise SystemExit(f"No metric rows found in {args.input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    raw_rows = [row for row in rows if row.get("_kind") == "raw"]
    degradation_rows = [row for row in rows if row.get("_kind") == "degradation"]

    raw_overall_path = os.path.join(args.output_dir, "summary_overall.csv")
    raw_table_path = os.path.join(args.output_dir, "summary_table.csv")
    write_summary_csv(raw_rows, raw_overall_path, "raw", RAW_METRIC_FIELDS, use_condition=True)
    write_table_csv(raw_overall_path, raw_table_path, "raw", use_condition=True)

    if degradation_rows:
        degradation_overall_path = os.path.join(args.output_dir, "summary_degradation_overall.csv")
        degradation_table_path = os.path.join(args.output_dir, "summary_degradation_table.csv")
        write_summary_csv(degradation_rows, degradation_overall_path, "degradation", DEGRADATION_METRIC_FIELDS, use_condition=True)
        write_table_csv(degradation_overall_path, degradation_table_path, "degradation", use_condition=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
