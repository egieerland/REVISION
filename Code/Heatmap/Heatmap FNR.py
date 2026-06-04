"""Visualize Full Brake false negative rate (FNR) per model and scenario.

This utility reads the aggregated CSV export that lists, for each scenario and
fusion model, the false negative rate for the safety-critical *Full Brake*
action. It produces a heatmap highlighting the miss rate as well as an optional
CSV of the pivoted matrix for further analysis.
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


DEFAULT_INPUT = Path("Laporan/ALL False Negative Rate (FNR) for Full_Brake (Safety Critical).csv")
DEFAULT_IMAGE_OUTPUT = Path("Laporan/heatmaps/fnr_full_brake_overview.png")
DEFAULT_MATRIX_OUTPUT = Path("Laporan/heatmaps/fnr_full_brake_overview.csv")

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


def parse_fnr_file(input_path: Path) -> pd.DataFrame:
	if not input_path.exists():
		raise FileNotFoundError(f"FNR file not found: {input_path}")

	text = _read_text_with_fallback(input_path)
	reader = csv.reader(io.StringIO(text))

	rows: List[dict[str, str]] = []
	current_scenario: str | None = None
	current_model: str | None = None
	metric_header: List[str] | None = None
	metric_values: dict[str, str] = {}

	def flush() -> None:
		nonlocal metric_values
		if current_scenario and current_model and metric_values:
			record = {"Scenario": current_scenario, "Model": current_model}
			record.update(metric_values)
			rows.append(record)
			metric_values = {}

	for raw_row in reader:
		row = [cell.strip() for cell in raw_row]

		if not any(row):
			continue

		first_cell = row[0]

		if first_cell.lower().startswith("scenario"):
			flush()
			current_scenario = first_cell.split(",")[0].strip()
			current_model = None
			metric_header = None
			continue

		if first_cell and not any(row[1:]) and first_cell.lower() != "metric":
			flush()
			current_model = MODEL_ALIASES.get(first_cell, first_cell)
			metric_header = None
			continue

		if first_cell.lower() == "metric":
			metric_header = row
			metric_values = {}
			continue

		if metric_header and len(row) >= len(metric_header):
			key = row[0]
			value = row[1] if len(row) > 1 else ""
			metric_values[key] = value

	flush()

	if not rows:
		raise ValueError("No data rows parsed from FNR file.")

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


def build_fnr_matrix(df: pd.DataFrame) -> pd.DataFrame:
	if df.empty:
		raise ValueError("Input dataframe is empty; cannot build FNR matrix.")

	df = df.copy()
	scenario_labels = list(dict.fromkeys(df["Scenario"].tolist()))
	scenario_codes = [_format_scenario_code(str(scenario)) for scenario in scenario_labels]
	df["ScenarioCode"] = [
		_format_scenario_code(str(scenario)) for scenario in df["Scenario"]
	]
	df["ScenarioCode"] = pd.Categorical(
		df["ScenarioCode"], categories=scenario_codes, ordered=True
	)

	pivot = df.pivot_table(
		index="Model",
		columns="ScenarioCode",
		values="FNR_Full_Brake",
		aggfunc="mean",
		observed=False,
	)

	present_models = list(df["Model"].dropna().unique())
	ordered_models = [model for model in MODEL_ORDER if model in present_models]
	remaining_models = [model for model in present_models if model not in MODEL_ORDER]

	matrix = pivot.reindex(ordered_models + remaining_models).reindex(columns=scenario_codes)

	return matrix


def plot_fnr_heatmap(matrix: pd.DataFrame, output_path: Path) -> None:
	if matrix.empty:
		raise ValueError("Cannot render heatmap: the FNR matrix has no data.")

	width = max(12.0, matrix.shape[1] * 0.9)
	height = max(4.0, matrix.shape[0] * 0.8)

	annotations = matrix.apply(
		lambda column: column.map(lambda value: f"{value:.3f}" if pd.notna(value) else "")
	)

	plt.figure(figsize=(width, height))
	max_value = matrix.max().max()
	if pd.isna(max_value) or max_value <= 0:
		max_value = 0.001
	sns.heatmap(
		matrix,
		annot=annotations,
		fmt="",
		cmap="Reds",
		vmin=0.0,
		vmax=float(max_value),
		linewidths=0.4,
		linecolor="#f0f0f0",
		cbar_kws={"label": "False Negative Rate (Full Brake)"},
		square=False,
	)
	plt.xticks(rotation=45, ha="right")
	plt.yticks(rotation=0)
	plt.xlabel("Scenario")
	plt.ylabel("Model")
	plt.title("Full Brake False Negative Rate by Model and Scenario")
	plt.tight_layout()

	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, dpi=300)
	plt.close()


def save_matrix(matrix: pd.DataFrame, output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	matrix.to_csv(output_path, float_format="%.4f")


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Generate a heatmap showing the Full Brake false negative rate across scenarios."
	)
	parser.add_argument(
		"--input",
		type=Path,
		default=DEFAULT_INPUT,
		help=f"Path to the aggregated FNR CSV (default: {DEFAULT_INPUT})",
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
			"Optional CSV export of the pivoted FNR matrix"
			f" (default: {DEFAULT_MATRIX_OUTPUT}). Set to '-' to skip."
		),
	)
	return parser


def main() -> None:
	parser = build_argument_parser()
	args = parser.parse_args()

	df = parse_fnr_file(args.input)
	fnr_matrix = build_fnr_matrix(df)

	plot_fnr_heatmap(fnr_matrix, args.image_output)

	if args.matrix_output != Path("-"):
		save_matrix(fnr_matrix, args.matrix_output)

	print(
		"Generated FNR heatmap for "
		f"{len(fnr_matrix.index)} models × {len(fnr_matrix.columns)} scenarios."
	)
	print(f"Image saved to: {args.image_output.resolve()}")
	if args.matrix_output != Path("-"):
		print(f"Matrix CSV saved to: {args.matrix_output.resolve()}")


if __name__ == "__main__":
	main()
