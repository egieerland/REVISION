"""Generate visualization artifacts from confusion matrix exports.

This module focuses on building a single, comprehensive F1-score heatmap where
rows correspond to fusion models and columns represent scenario-class
combinations (e.g. ``S1_NO_Action``). The color intensity communicates the
F1-score for each pair, making it easy to spot strengths and weaknesses across
the evaluated scenarios.
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


DEFAULT_INPUT = Path("Laporan/ALL Confusion Matrix per-class .csv")
DEFAULT_IMAGE_OUTPUT = Path("Laporan/heatmaps/f1_score_overview.png")
DEFAULT_MATRIX_OUTPUT = Path("Laporan/heatmaps/f1_score_overview.csv")

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
	"""Read *path* trying multiple encodings until one succeeds."""

	encodings = list(encodings or ("utf-8-sig", "utf-16", "cp1252", "latin-1"))

	for encoding in encodings:
		try:
			return path.read_text(encoding=encoding)
		except UnicodeDecodeError:
			continue

	raise UnicodeDecodeError("", b"", 0, 1, f"Unable to decode file: {path}")


def parse_confusion_matrix(input_path: Path) -> pd.DataFrame:
	"""Parse the aggregated confusion matrix CSV into a tidy dataframe."""

	if not input_path.exists():
		raise FileNotFoundError(f"Confusion matrix file not found: {input_path}")

	text = _read_text_with_fallback(input_path)
	reader = csv.reader(io.StringIO(text))

	rows: List[dict[str, str]] = []
	current_scenario: str | None = None
	current_model: str | None = None
	header: List[str] | None = None

	for raw_row in reader:
		row = [cell.strip() for cell in raw_row]

		# Skip completely empty rows.
		if not any(row):
			continue

		first_cell = row[0]

		if first_cell.lower().startswith("scenario"):
			current_scenario = first_cell.split(",")[0].strip()
			current_model = None
			header = None
			continue

		if first_cell and not any(row[1:]) and first_cell.lower() != "class":
			current_model = MODEL_ALIASES.get(first_cell, first_cell)
			header = None
			continue

		if first_cell.lower() == "class":
			header = row
			continue

		if current_scenario and current_model and header:
			# Align row length with the header to avoid key mismatches.
			padded_row = row + [""] * (len(header) - len(row))
			record = {column: value for column, value in zip(header, padded_row)}
			record["Scenario"] = current_scenario
			record["Model"] = current_model
			rows.append(record)

	if not rows:
		raise ValueError(
			"The confusion matrix file was parsed but no data rows were detected."
		)

	df = pd.DataFrame(rows)

	# Ensure model naming consistency.
	df["Model"] = df["Model"].replace(MODEL_ALIASES)

	numeric_columns = [
		column
		for column in df.columns
		if column not in {"Scenario", "Model", "Class"}
	]

	df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

	return df


def _format_scenario_code(raw_name: str) -> str:
	"""Convert scenario labels like ``Scenario 5B`` to a compact identifier ``S5B``."""

	raw_name = raw_name.strip()
	match = re.match(r"scenario\s+(.+)", raw_name, flags=re.IGNORECASE)
	if match:
		suffix = match.group(1)
	else:
		# Fallback to the raw string if it does not start with "Scenario".
		suffix = raw_name

	suffix = re.sub(r"[\s-]+", "", suffix)
	suffix = suffix.upper()

	if suffix.startswith("S"):
		return suffix
	return f"S{suffix}"


def build_f1_matrix(df: pd.DataFrame) -> pd.DataFrame:
	"""Pivot the dataframe into model rows and scenario-class F1-score columns."""

	if df.empty:
		raise ValueError("Input dataframe is empty; cannot build F1-score matrix.")

	scenario_classes: List[str] = []

	for scenario, class_name in zip(df["Scenario"], df["Class"]):
		code = _format_scenario_code(str(scenario))
		label = f"{code}_{class_name}"
		scenario_classes.append(label)

	# Preserve the encounter order for columns.
	scenario_class_order = list(dict.fromkeys(scenario_classes))
	df = df.copy()
	df["ScenarioClass"] = scenario_classes
	df["ScenarioClass"] = pd.Categorical(
		df["ScenarioClass"], categories=scenario_class_order, ordered=True
	)

	pivot = df.pivot_table(
		index="Model",
		columns="ScenarioClass",
		values="F1-Score",
		aggfunc="mean",
		observed=False,
	)

	present_models = list(df["Model"].dropna().unique())
	ordered_models = [model for model in MODEL_ORDER if model in present_models]
	remaining_models = [model for model in present_models if model not in MODEL_ORDER]

	matrix = pivot.reindex(ordered_models + remaining_models).reindex(
		columns=scenario_class_order
	)

	return matrix


def plot_f1_heatmap(matrix: pd.DataFrame, output_path: Path) -> None:
	"""Render and save the overview F1-score heatmap."""

	if matrix.empty:
		raise ValueError("Cannot render heatmap: the F1-score matrix has no data.")

	width = max(14.0, matrix.shape[1] * 0.45)
	height = max(4.0, matrix.shape[0] * 0.8)

	annotations = matrix.apply(
		lambda column: column.map(lambda value: f"{value:.2f}" if pd.notna(value) else "")
	)

	plt.figure(figsize=(width, height))
	sns.heatmap(
		matrix,
		annot=annotations,
		fmt="",
		cmap="YlGnBu",
		vmin=0.0,
		vmax=1.0,
		linewidths=0.4,
		linecolor="#e0e0e0",
		cbar_kws={"label": "F1-score"},
		square=False,
	)
	plt.xticks(rotation=45, ha="right")
	plt.yticks(rotation=0)
	plt.xlabel("Scenario · Class")
	plt.ylabel("Model")
	plt.title("F1-score Overview by Model and Scenario/Class")
	plt.tight_layout()

	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, dpi=300)
	plt.close()


def save_matrix(matrix: pd.DataFrame, output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	matrix.to_csv(output_path, float_format="%.4f")


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Generate an F1-score heatmap across scenarios and classes for each model.")
	parser.add_argument(
		"--input",
		type=Path,
		default=DEFAULT_INPUT,
		help=f"Path to the aggregated confusion matrix CSV (default: {DEFAULT_INPUT})",
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
			"Optional CSV export of the pivoted F1-score matrix"
			f" (default: {DEFAULT_MATRIX_OUTPUT}). Set to '-' to skip."
		),
	)
	return parser


def main() -> None:
	parser = build_argument_parser()
	args = parser.parse_args()

	df = parse_confusion_matrix(args.input)
	f1_matrix = build_f1_matrix(df)

	plot_f1_heatmap(f1_matrix, args.image_output)

	if args.matrix_output != Path("-"):
		save_matrix(f1_matrix, args.matrix_output)

	print(
		"Generated F1-score heatmap for "
		f"{len(f1_matrix.index)} models × {len(f1_matrix.columns)} scenario-class pairs."
	)
	print(f"Image saved to: {args.image_output.resolve()}")
	if args.matrix_output != Path("-"):
		print(f"Matrix CSV saved to: {args.matrix_output.resolve()}")


if __name__ == "__main__":
	main()
