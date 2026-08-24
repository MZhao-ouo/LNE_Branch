from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
WEBSITE_DIR = ROOT / "website" / "branch_global_suvr"
OUTPUT_PATH = WEBSITE_DIR / "data.js"

DIAG_ORDER = [1, 2, 4]
DIAG_LABELS = {1: "CN", 2: "MCI", 4: "AD"}
BRANCH_ORDER = ["common_branch", "normal_branch", "AD_branch"]
BRANCH_META = {
    "common_branch": {
        "label": "Common / split point",
        "line_color": "#757575",
        "point_color": "#8C8C8C",
    },
    "normal_branch": {
        "label": "Normal branch",
        "line_color": "#1F5EFF",
        "point_color": "#1F77B4",
    },
    "AD_branch": {
        "label": "AD branch",
        "line_color": "#D94841",
        "point_color": "#D62728",
    },
}
DATASET_PATTERN = re.compile(r"^trial2_(?P<name>.+)_pre\.csv$")
WINDOW_METRIC_SPECS = [
    {
        "key": "global_suvr",
        "label": "Global SUVR",
        "source_column": None,
        "step": 0.001,
        "default_width": 0.08,
    },
    {
        "key": "age",
        "label": "Age",
        "source_column": "age",
        "step": 0.1,
        "default_width": 8.0,
    },
    {
        "key": "PHC_MEM",
        "label": "PHC_MEM",
        "source_column": "PHC_MEM",
        "step": 0.01,
        "default_width": 0.6,
    },
    {
        "key": "PHC_EXF",
        "label": "PHC_EXF",
        "source_column": "PHC_EXF",
        "step": 0.01,
        "default_width": 0.6,
    },
]


def rf(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def maybe_int(value) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def maybe_identifier(value) -> int | str | None:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        return value
    return int(value)


def maybe_float(value, digits: int = 6) -> float | None:
    if pd.isna(value):
        return None
    return rf(value, digits)


def format_dataset_label(dataset_name: str) -> str:
    parts = []
    for token in dataset_name.split("_"):
        if token.isupper() or any(char.isdigit() for char in token):
            parts.append(token)
        else:
            parts.append(token.capitalize())
    return " ".join(parts)


def discover_dataset_configs() -> list[dict]:
    analysis_dir = ROOT / "analysis"
    configs = []

    for csv_path in sorted(analysis_dir.glob("trial2_*_pre.csv")):
        match = DATASET_PATTERN.match(csv_path.name)
        if not match:
            continue

        dataset_name = match.group("name")
        configs.append(
            {
                "key": dataset_name,
                "label": format_dataset_label(dataset_name),
                "source_csv": csv_path,
                "description": "Auto-discovered dataset.",
            }
        )

    if not configs:
        raise FileNotFoundError(
            f"No files matching {analysis_dir / 'trial2_*_pre.csv'} were found."
        )

    return configs


def build_window_metric(series: pd.Series, spec: dict) -> dict | None:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return None

    min_value = float(valid.min())
    max_value = float(valid.max())
    span = max_value - min_value
    default_width = min(spec["default_width"], span) if span > 0 else 0.0

    if default_width == 0:
        default_min = min_value
        default_max = max_value
    else:
        median_value = float(valid.median())
        default_min = max(min_value, median_value - default_width / 2)
        default_max = min(max_value, median_value + default_width / 2)

    return {
        "key": spec["key"],
        "label": spec["label"],
        "min": rf(min_value),
        "max": rf(max_value),
        "mean": rf(valid.mean()),
        "median": rf(valid.median()),
        "std": rf(valid.std()),
        "step": spec["step"],
        "available_count": int(valid.shape[0]),
        "default_window": {
            "min": rf(default_min),
            "max": rf(default_max),
        },
    }


def build_dataset(config: dict) -> dict:
    csv_path = Path(config["source_csv"])
    df = pd.read_csv(csv_path)
    df = df[df["lb"].isin(DIAG_ORDER)].copy().reset_index(drop=True)

    suvr_cols = sorted(col for col in df.columns if col.startswith("CTX_"))
    df["global_suvr"] = df[suvr_cols].mean(axis=1)
    for spec in WINDOW_METRIC_SPECS:
        if spec["key"] == "global_suvr":
            continue
        if spec["source_column"] in df.columns:
            df[spec["key"]] = pd.to_numeric(df[spec["source_column"]], errors="coerce")
        else:
            df[spec["key"]] = pd.NA
    df["row_id"] = range(len(df))

    points = df[
        [
            "row_id",
            "RID",
            "lb",
            "branch",
            "cluster",
            "embedding1",
            "embedding2",
            "global_suvr",
            "age",
            "PHC_MEM",
            "PHC_EXF",
        ]
    ].copy()
    points = points.sort_values(["row_id"]).reset_index(drop=True)

    cluster_centers = (
        df.groupby(["cluster", "branch"], as_index=False)[["embedding1", "embedding2"]]
        .mean()
        .sort_values(["branch", "cluster"])
        .reset_index(drop=True)
    )

    global_suvr = points["global_suvr"]
    window_metrics = {}
    window_metric_order = []
    for spec in WINDOW_METRIC_SPECS:
        metric = build_window_metric(df[spec["key"]], spec)
        if metric is None:
            continue
        window_metrics[spec["key"]] = metric
        window_metric_order.append(spec["key"])

    global_suvr_metric = window_metrics["global_suvr"]

    point_records = []
    for row in points.itertuples(index=False):
        branch = row.branch
        point_records.append(
            {
                "row_id": int(row.row_id),
                "RID": maybe_identifier(row.RID),
                "lb": int(row.lb),
                "diag_label": DIAG_LABELS[int(row.lb)],
                "branch": branch,
                "branch_label": BRANCH_META[branch]["label"],
                "cluster": maybe_int(row.cluster),
                "embedding1": rf(row.embedding1),
                "embedding2": rf(row.embedding2),
                "global_suvr": rf(row.global_suvr),
                "age": maybe_float(row.age),
                "PHC_MEM": maybe_float(row.PHC_MEM),
                "PHC_EXF": maybe_float(row.PHC_EXF),
            }
        )

    center_records = []
    for row in cluster_centers.itertuples(index=False):
        center_records.append(
            {
                "cluster": maybe_int(row.cluster),
                "branch": row.branch,
                "embedding1": rf(row.embedding1),
                "embedding2": rf(row.embedding2),
            }
        )

    counts_by_branch = {
        branch: int((df["branch"] == branch).sum()) for branch in BRANCH_ORDER
    }
    counts_by_diagnosis = {
        str(diag): int((df["lb"] == diag).sum()) for diag in DIAG_ORDER
    }

    return {
        "key": config["key"],
        "label": config["label"],
        "description": config["description"],
        "source_csv": str(csv_path.relative_to(ROOT)),
        "rows": int(len(df)),
        "unique_rids": int(df["RID"].nunique()),
        "suvr_column_count": int(len(suvr_cols)),
        "branch_order": BRANCH_ORDER,
        "diagnosis_order": DIAG_ORDER,
        "branch_meta": BRANCH_META,
        "diagnosis_labels": {str(key): value for key, value in DIAG_LABELS.items()},
        "window_metric_order": window_metric_order,
        "window_metrics": window_metrics,
        "counts_by_branch": counts_by_branch,
        "counts_by_diagnosis": counts_by_diagnosis,
        "global_suvr": {
            "min": global_suvr_metric["min"],
            "max": global_suvr_metric["max"],
            "mean": global_suvr_metric["mean"],
            "median": global_suvr_metric["median"],
            "std": global_suvr_metric["std"],
        },
        "default_window": {
            "min": global_suvr_metric["default_window"]["min"],
            "max": global_suvr_metric["default_window"]["max"],
            "step": global_suvr_metric["step"],
        },
        "extents": {
            "embedding1": {
                "min": rf(df["embedding1"].min()),
                "max": rf(df["embedding1"].max()),
            },
            "embedding2": {
                "min": rf(df["embedding2"].min()),
                "max": rf(df["embedding2"].max()),
            },
        },
        "cluster_centers": center_records,
        "points": point_records,
    }


def main() -> None:
    dataset_configs = discover_dataset_configs()
    payload = {
        "datasetOrder": [config["key"] for config in dataset_configs],
        "datasets": {
            config["key"]: build_dataset(config) for config in dataset_configs
        },
    }
    OUTPUT_PATH.write_text(
        "window.BRANCH_GLOBAL_SUVR_DATASETS = "
        + json.dumps(payload, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_PATH} with {len(dataset_configs)} datasets")


if __name__ == "__main__":
    main()
