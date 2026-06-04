"""Generate a heatmap for Mean Time-To-Detect (MTTD).

Reads the aggregated statistics CSV and pivots the requested statistic (Mean by
default) into a scenario × model matrix, then renders an annotated heatmap and
optionally exports the matrix for downstream analysis.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_INPUT = Path("Laporan/ALL Mean Time-To-Detect (MTTD).csv")
DEFAULT_IMAGE_OUTPUT = Path("Laporan/heatmaps/mttd_overview.png")
DEFAULT_MATRIX_OUTPUT = Path("Laporan/heatmaps/mttd_overview.csv")
DEFAULT_METRIC = "Mean"

MODEL_ALIASES = {
	"Rule-Based": "Rule Based",
	"Rule-Based Model": "Rule Based",
}

MODEL_ORDER = [
	"Expected result",
	"Rule Based",
	"Bayesian",
	"Dempster-Shafer",
	"XGBoost",
	"SmallMLP",
	"Sequence",
	"SmHybrid",
]


def _read_text_with_fallback(path: Path, encodings: Iterable[str] | None = None) -> str:
	encodings = list(encodings or ("utf-8-sig", "utf-16", "cp1252", "latin-1"))
	for encoding in encodings:
		try:
			return path.read_text(encoding=encoding)
		except UnicodeDecodeError:
			continue
	raise UnicodeDecodeError("", b"", 0, 1, f"Unable to decode file: {path}")


def parse_mttd_file(input_path: Path) -> pd.DataFrame:
	if not input_path.exists():
		raise FileNotFoundError(f"MTTD file not found: {input_path}")

	text = _read_text_with_fallback(input_path)
	reader = csv.reader(io.StringIO(text))

	rows: List[dict[str, str]] = []
	current_scenario: str | None = None
	current_model: str | None = None
	value_header: List[str] | None = None
	metrics: dict[str, str] = {}

	def flush() -> None:
		nonlocal metrics
		if current_scenario and current_model and metrics:
			record = {"Scenario": current_scenario, "Model": current_model}
			record.update(metrics)
			rows.append(record)
			metrics = {}

	for raw_row in reader:
		row = [cell.strip() for cell in raw_row]

		if not any(row):
			continue

		first_cell = row[0]

		if first_cell.lower().startswith("scenario"):
			flush()
			current_scenario = first_cell.split(",")[0].strip()
			current_model = None
			value_header = None
			continue

		if first_cell and not any(row[1:]) and first_cell.lower() != "statistic":
			flush()
			current_model = MODEL_ALIASES.get(first_cell, first_cell)
			value_header = None
			continue

		if first_cell.lower() == "statistic":
			value_header = row
			metrics = {}
			continue

		if value_header and len(row) >= 2:
			metric_name = row[0]
			metric_value = row[1]
			metrics[metric_name] = metric_value

	flush()

	if not rows:
		raise ValueError("No data rows parsed from MTTD file.")

	df = pd.DataFrame(rows)
	df["Model"] = df["Model"].replace(MODEL_ALIASES)

	numeric_columns = [col for col in df.columns if col not in {"Scenario", "Model"}]
	df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

	return df


def _format_scenario_code(raw_name: str) -> str:
	raw_name = raw_name.strip()
	match = re.match(r"scenario\s+(.+)", raw_name, flags=re.IGNORECASE)
	suffix = match.group(1) if match else raw_name
	suffix = re.sub(r"[\s-]+", "", suffix)
	suffix = suffix.upper()
	return suffix if suffix.startswith("S") else f"S{suffix}"


def build_metric_matrix(df: pd.DataFrame, metric: str) -> pd.DataFrame:
	if metric not in df.columns:
		available = ", ".join(sorted(col for col in df.columns if col not in {"Scenario", "Model"}))
		raise KeyError(f"Metric '{metric}' not found. Available metrics: {available}")

	df = df.copy()
	scenario_labels = list(dict.fromkeys(df["Scenario"].tolist()))
	scenario_codes = [_format_scenario_code(str(label)) for label in scenario_labels]
	df["ScenarioCode"] = [
		_format_scenario_code(str(scenario)) for scenario in df["Scenario"]
	]
	df["ScenarioCode"] = pd.Categorical(
		df["ScenarioCode"], categories=scenario_codes, ordered=True
	)

	pivot = df.pivot_table(
		index="Model",
		columns="ScenarioCode",
		values=metric,
		aggfunc="mean",
		observed=False,
	)

	present_models = list(df["Model"].dropna().unique())
	ordered_models = [model for model in MODEL_ORDER if model in present_models]
	remaining_models = [model for model in present_models if model not in MODEL_ORDER]

	matrix = pivot.reindex(ordered_models + remaining_models).reindex(columns=scenario_codes)

	return matrix


def plot_metric_heatmap(matrix: pd.DataFrame, metric_label: str, output_path: Path) -> None:
	if matrix.empty:
		raise ValueError("Cannot render heatmap: the metric matrix has no data.")

	width = max(12.0, matrix.shape[1] * 0.9)
	height = max(4.0, matrix.shape[0] * 0.8)

	annotations = matrix.apply(
		lambda column: column.map(lambda value: f"{value:.3f}" if pd.notna(value) else "")
	)

	max_value = matrix.max().max()
	min_value = matrix.min().min()
	if pd.isna(max_value):
		max_value = 0.1
	if pd.isna(min_value):
		min_value = 0.0

	plt.figure(figsize=(width, height))
	sns.heatmap(
		matrix,
		annot=annotations,
		fmt="",
		cmap="YlOrRd",
		vmin=float(min_value),
		vmax=float(max_value),
		linewidths=0.4,
		linecolor="#f0f0f0",
		cbar_kws={"label": f"{metric_label} (seconds)"},
		square=False,
	)
	plt.xticks(rotation=45, ha="right")
	plt.yticks(rotation=0)
	plt.xlabel("Scenario")
	plt.ylabel("Model")
	plt.title(f"{metric_label} by Model and Scenario")
	plt.tight_layout()

	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, dpi=300)
	plt.close()


def save_matrix(matrix: pd.DataFrame, output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	matrix.to_csv(output_path, float_format="%.4f")


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Generate a heatmap for Mean Time-To-Detect statistics across scenarios."
	)
	parser.add_argument(
		"--input",
		type=Path,
		default=DEFAULT_INPUT,
		help=f"Path to the aggregated MTTD CSV (default: {DEFAULT_INPUT})",
	)
	parser.add_argument(
		"--metric",
		type=str,
		default=DEFAULT_METRIC,
		help="Statistic column to visualize (default: Mean)",
	)
	parser.add_argument(
		"--image-output",
		type=Path,
		default=DEFAULT_IMAGE_OUTPUT,
		help=f"Destination for the heatmap image (default: {DEFAULT_IMAGE_OUTPUT})",
	)
	parser.add_argument(
		"--matrix-output",
		type=Path,
		default=DEFAULT_MATRIX_OUTPUT,
		help=(
			"Optional CSV export of the pivoted metric matrix"
			f" (default: {DEFAULT_MATRIX_OUTPUT}). Set to '-' to skip."
		),
	)
	return parser


def main() -> None:
	parser = build_argument_parser()
	args = parser.parse_args()

	df = parse_mttd_file(args.input)
	metric_matrix = build_metric_matrix(df, args.metric)

	plot_metric_heatmap(metric_matrix, args.metric, args.image_output)

	if args.matrix_output != Path("-"):
		save_matrix(metric_matrix, args.matrix_output)

	print(
		"Generated MTTD heatmap for "
		f"{len(metric_matrix.index)} models × {len(metric_matrix.columns)} scenarios."
	)
	print(f"Image saved to: {args.image_output.resolve()}")
	if args.matrix_output != Path("-"):
		print(f"Matrix CSV saved to: {args.matrix_output.resolve()}")


if __name__ == "__main__":
	main()
