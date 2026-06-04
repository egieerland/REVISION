"""Generate visualization artifacts from confusion matrix exports.

This module can build overview heatmaps for Precision, Recall, or F1-score
values. Rows correspond to fusion models and columns represent scenario-class
combinations (e.g. ``S1_NO_Action``). The color intensity communicates the
selected metric for each pair, making it easy to spot strengths and weaknesses
across the evaluated scenarios.
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
DEFAULT_METRIC = "F1-Score"
DEFAULT_IMAGE_TEMPLATE = "Laporan/heatmaps/{slug}_overview.png"
DEFAULT_MATRIX_TEMPLATE = "Laporan/heatmaps/{slug}_overview.csv"

SCENARIO_SPLITS = [
	("s1_s5a", "S1–S5A", ["S1", "S2", "S3", "S4", "S5A"]),
	("s5b_s6a", "S5B–S6A", ["S5B", "S5C", "S5D", "S5E", "S6A"]),
	("s6b_s6f", "S6B–S6F", ["S6B", "S6C", "S6D", "S6E", "S6F"]),
]

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


def build_metric_matrix(df: pd.DataFrame, metric: str) -> pd.DataFrame:
	"""Pivot the dataframe into model rows and scenario-class metric columns."""

	if df.empty:
		raise ValueError("Input dataframe is empty; cannot build metric matrix.")

	if metric not in df.columns:
		available = ", ".join(
			sorted(column for column in df.columns if column not in {"Scenario", "Model", "Class"})
		)
		raise KeyError(
			f"Metric '{metric}' not found in dataframe. Available metric columns: {available}"
		)

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
		values=metric,
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


def split_matrix_by_scenario(matrix: pd.DataFrame) -> List[tuple[str, str, pd.DataFrame]]:
	"""Split the matrix into predefined scenario ranges while preserving column order."""

	if matrix.empty:
		return []

	column_scenarios = {
		column: column.split("_", 1)[0]
		for column in matrix.columns
	}

	assigned_columns: set[str] = set()
	splits: List[tuple[str, str, pd.DataFrame]] = []

	for slug, label, scenario_codes in SCENARIO_SPLITS:
		selected_columns = [
			column
			for column in matrix.columns
			if column_scenarios.get(column) in scenario_codes
		]
		if selected_columns:
			splits.append((slug, label, matrix[selected_columns]))
			assigned_columns.update(selected_columns)

	remaining_columns = [
		column for column in matrix.columns if column not in assigned_columns
	]

	if remaining_columns:
		splits.append(("remaining", "Remaining scenarios", matrix[remaining_columns]))

	return splits


def _with_suffix(path: Path, slug: str) -> Path:
	"""Append _<slug> before the file extension of *path*."""

	suffix = path.suffix
	stem = path.stem
	return path.with_name(f"{stem}_{slug}{suffix}")


def plot_metric_heatmap(
	matrix: pd.DataFrame,
	metric: str,
	output_path: Path,
	title_suffix: str | None = None,
) -> None:
	"""Render and save the overview metric heatmap."""

	if matrix.empty:
		raise ValueError("Cannot render heatmap: the metric matrix has no data.")

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
		cbar_kws={"label": metric},
		square=False,
	)
	plt.xticks(rotation=45, ha="right")
	plt.yticks(rotation=0)
	plt.xlabel("Scenario · Class")
	plt.ylabel("Model")
	title = f"{metric} Overview by Model and Scenario/Class"
	if title_suffix:
		title = f"{title} {title_suffix}"
	plt.title(title)
	plt.tight_layout()

	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, dpi=300)
	plt.close()


def save_matrix(matrix: pd.DataFrame, output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	matrix.to_csv(output_path, float_format="%.4f")


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Generate a metric heatmap (Precision, Recall, or F1-score) across scenarios and classes for each model."
	)
	parser.add_argument(
		"--metric",
		type=str,
		default=DEFAULT_METRIC,
		help="Metric column to visualize. Expected values include Precision, Recall, or F1-Score (default: F1-Score).",
	)
	parser.add_argument(
		"--input",
		type=Path,
		default=DEFAULT_INPUT,
		help=f"Path to the aggregated confusion matrix CSV (default: {DEFAULT_INPUT})",
	)
	parser.add_argument(
		"--image-output",
		type=Path,
		default=None,
		help="Destination for the heatmap image. Defaults to a file named after the selected metric.",
	)
	parser.add_argument(
		"--matrix-output",
		type=Path,
		default=None,
		help=(
			"Optional CSV export of the pivoted metric matrix. Defaults to a file named after the selected metric."
			" Set to '-' to skip."
		),
	)
	return parser


def main() -> None:
	parser = build_argument_parser()
	args = parser.parse_args()

	metric = args.metric.strip()

	# Allow case-insensitive matches for the known metric names.
	known_metrics = {name.lower(): name for name in ("Precision", "Recall", "F1-Score")}
	metric_lookup = metric.lower()
	if metric_lookup in known_metrics:
		metric = known_metrics[metric_lookup]

	metric_slug = metric.lower().replace(" ", "_").replace("-", "_")

	image_output = args.image_output or Path(DEFAULT_IMAGE_TEMPLATE.format(slug=metric_slug))
	matrix_output = args.matrix_output or Path(DEFAULT_MATRIX_TEMPLATE.format(slug=metric_slug))

	df = parse_confusion_matrix(args.input)
	metric_matrix = build_metric_matrix(df, metric)

	split_matrices = split_matrix_by_scenario(metric_matrix)
	if not split_matrices:
		raise ValueError("No scenario columns available after splitting; check input data.")

	image_paths: List[Path] = []
	matrix_paths: List[Path] = []

	for slug, label, sub_matrix in split_matrices:
		part_image_path = _with_suffix(image_output, slug)

		plot_metric_heatmap(sub_matrix, metric, part_image_path, title_suffix=f"({label})")
		image_paths.append(part_image_path)

		if matrix_output != Path("-"):
			part_matrix_path = _with_suffix(matrix_output, slug)
			save_matrix(sub_matrix, part_matrix_path)
			matrix_paths.append(part_matrix_path)

	# Always produce the full overview heatmap using the base filename.
	plot_metric_heatmap(metric_matrix, metric, image_output, title_suffix="(All Scenarios)")
	image_paths.insert(0, image_output)

	# Always save the complete matrix for downstream processing if requested.
	if matrix_output != Path("-"):
		save_matrix(metric_matrix, matrix_output)

	print(
		f"Generated {metric} heatmaps across {len(split_matrices)} scenario ranges, covering "
		f"{len(metric_matrix.index)} models × {len(metric_matrix.columns)} scenario-class pairs."
	)
	print("Images saved to:")
	for path in image_paths:
		print(f"  - {path.resolve()}")
	if matrix_output != Path("-"):
		print("Matrices saved to:")
		# Include the full-matrix export first for clarity.
		print(f"  - {matrix_output.resolve()} (full matrix)")
		for path in matrix_paths:
			if path == matrix_output:
				continue
			print(f"  - {path.resolve()}")


if __name__ == "__main__":
	main()
