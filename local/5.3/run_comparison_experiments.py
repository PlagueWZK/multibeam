r"""Run the 5.3 partition-feature comparison experiments.

Usage from the project root on Windows PowerShell:

    chcp 65001
    $env:PYTHONUTF8 = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    .\.venv\Scripts\python.exe .\local\5.3\run_comparison_experiments.py

The script intentionally runs all combinations through the same frozen pipeline:

- adaptive secondary partition trigger: max-gap, minimum 0.10, area >= 5%
- start point strategy: geometric_center
- line retention: adaptive max(30%, parent_gain * 80%) by default, or fixed threshold via CLI
- planning scope: plan_all
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import os
import sys
from dataclasses import dataclass
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
EXPECTED_VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

from multibeam.Coverage import calculate_coverage_matrix_with_ml
from multibeam.GridCell import calculate_optimal_mesh_size
from multibeam.Partition import partition_coverage_matrix
from multibeam.Planner import SurveyPlanner


@dataclass(frozen=True)
class ComparisonCombo:
    label: str
    primary_feature_mode: str
    display_name: str


COMBOS: tuple[ComparisonCombo, ...] = (
    ComparisonCombo(
        label="xy-depth-cov_g",
        primary_feature_mode="xy_coverage_depth",
        display_name="xy-depth-cov/g",
    ),
    ComparisonCombo(
        label="depthcov_g",
        primary_feature_mode="coverage_depth",
        display_name="depthcov/g",
    ),
    ComparisonCombo(
        label="cov_g",
        primary_feature_mode="coverage_only",
        display_name="cov/g",
    ),
    ComparisonCombo(
        label="xy-cov_g",
        primary_feature_mode="xy_coverage",
        display_name="xy-cov/g",
    ),
)

X_MIN, X_MAX = 0, 4 * 1852
Y_MIN, Y_MAX = 0, 5 * 1852

START_POINT_STRATEGY = "geometric_center"
START_POINT_LABEL = "geocenter"
SECONDARY_DIRECTION_DISPERSION_MIN_THRESHOLD = 0.10
SECONDARY_MIN_AREA_RATIO_FOR_SECONDARY = 0.05
SECONDARY_RULE_LABEL = "sec-adaptive-min0p1-area5pct"
CHILD_LINE_PARENT_GAIN_FACTOR = 0.80
CHILD_LINE_MIN_GAIN_THRESHOLD = 0.30
PLANNING_SCOPE_LABEL = "plan-all"


def configure_utf8() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def ensure_project_venv() -> None:
    actual = Path(sys.executable).resolve()
    expected = EXPECTED_VENV_PYTHON.resolve()
    if actual != expected:
        raise RuntimeError(
            "请使用当前项目虚拟环境运行对比实验：\n"
            f"  期望: {expected}\n"
            f"  当前: {actual}"
        )


def parse_args() -> argparse.Namespace:
    combo_choices = ["all", *(combo.label for combo in COMBOS)]
    parser = argparse.ArgumentParser(
        description="Run frozen 5.3 comparison experiments for partition feature combos."
    )
    parser.add_argument(
        "--combo",
        choices=combo_choices,
        default="all",
        help="Run one combo or all combos. Default: all.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "multibeam" / "output",
        help="Output root directory. Default: multibeam/output.",
    )
    parser.add_argument(
        "--line-gain-mode",
        choices=("adaptive", "fixed"),
        default="adaptive",
        help=(
            "Line-retention gain threshold mode. "
            "adaptive keeps max(floor, parent_gain * factor); fixed uses --line-gain-threshold. "
            "Default: adaptive."
        ),
    )
    parser.add_argument(
        "--line-gain-threshold",
        type=float,
        default=0.50,
        help="Fixed line-retention gain threshold used when --line-gain-mode fixed. Default: 0.50.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.line_gain_threshold <= 1.0:
        parser.error("--line-gain-threshold must be in [0, 1].")
    return args


def percent_label(value: float) -> str:
    percent = float(value) * 100.0
    if abs(percent - round(percent)) < 1e-9:
        text = str(int(round(percent)))
    else:
        text = f"{percent:.2f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"{text}pct"


def build_line_gain_rule_label(line_gain_mode: str, line_gain_threshold: float) -> str:
    if line_gain_mode == "fixed":
        return f"gain-fixed{percent_label(line_gain_threshold)}"
    return "gain-adaptive-parent80pct-floor30pct"


def select_combos(combo_arg: str) -> list[ComparisonCombo]:
    if combo_arg == "all":
        return list(COMBOS)
    return [combo for combo in COMBOS if combo.label == combo_arg]


def build_output_dir(
    output_root: Path,
    run_timestamp: str,
    combo: ComparisonCombo,
    line_gain_rule_label: str,
) -> Path:
    output_name = (
        f"{run_timestamp}_comparison_{combo.label}_{START_POINT_LABEL}_"
        f"{SECONDARY_RULE_LABEL}_{line_gain_rule_label}_{PLANNING_SCOPE_LABEL}"
    )
    return output_root / output_name


def load_global_metrics(metrics_path: Path) -> dict[str, object]:
    if not metrics_path.exists():
        return {"metrics_status": "missing", "metrics_path": str(metrics_path)}

    import pandas as pd

    df = pd.read_excel(metrics_path, sheet_name="全局统计")
    metrics = {"metrics_status": "ok", "metrics_path": str(metrics_path)}
    for _, row in df.iterrows():
        metric_name = str(row.get("指标", "")).strip()
        if not metric_name:
            continue
        metrics[metric_name] = row.get("值")
    return metrics


def write_summary(rows: list[dict[str, object]], summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    configure_utf8()
    ensure_project_venv()
    args = parse_args()
    combos = select_combos(args.combo)
    if not combos:
        raise RuntimeError(f"未找到要运行的组合: {args.combo}")

    output_root = args.output_root.resolve()
    data_path = PROJECT_ROOT / "data" / "data.xlsx"
    run_timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    line_gain_rule_label = build_line_gain_rule_label(
        args.line_gain_mode, args.line_gain_threshold
    )

    print("[对比实验] 冻结配置")
    print(f"  combos={[combo.display_name for combo in combos]}")
    print(f"  start_point_strategy={START_POINT_STRATEGY}")
    print(
        "  secondary_trigger=adaptive-maxgap | "
        f"min_dispersion={SECONDARY_DIRECTION_DISPERSION_MIN_THRESHOLD:.2f} | "
        f"min_area_ratio={SECONDARY_MIN_AREA_RATIO_FOR_SECONDARY:.0%}"
    )
    if args.line_gain_mode == "fixed":
        print(
            "  line_gain=fixed-threshold | "
            f"threshold={args.line_gain_threshold:.0%}"
        )
    else:
        print(
            "  line_gain=adaptive-parent-gain | "
            f"parent_factor={CHILD_LINE_PARENT_GAIN_FACTOR:.0%} | "
            f"floor={CHILD_LINE_MIN_GAIN_THRESHOLD:.0%}"
        )
    print("  planning_scope=plan_all")
    print(f"  output_root={output_root}")
    print("  建议外部单组合超时: 2小时")

    print("[对比实验] Phase 1: 计算最优网格")
    d_optimal = calculate_optimal_mesh_size(data_path)
    print(f"[对比实验] d_optimal={d_optimal}")

    print("[对比实验] Phase 2: 计算覆盖矩阵与梯度矩阵")
    (
        xs,
        ys,
        depth_matrix,
        coverage_matrix,
        coarse_boundary_mask,
        coarse_cell_effective_area,
        coarse_cell_area_ratio,
        gx_matrix,
        gy_matrix,
    ) = calculate_coverage_matrix_with_ml(
        x_min=X_MIN,
        x_max=X_MAX,
        y_min=Y_MIN,
        y_max=Y_MAX,
        d=d_optimal,
    )

    summary_rows: list[dict[str, object]] = []
    for combo_index, combo in enumerate(combos, start=1):
        combo_output_dir = build_output_dir(
            output_root, run_timestamp, combo, line_gain_rule_label
        )
        print("\n" + "=" * 80)
        print(
            f"[对比实验] 组合 {combo_index}/{len(combos)}: "
            f"{combo.display_name} -> {combo.primary_feature_mode}"
        )
        print(f"[对比实验] 输出目录: {combo_output_dir}")

        final_cluster_matrix, final_u = partition_coverage_matrix(
            xs,
            ys,
            coverage_matrix,
            output_dir=str(combo_output_dir),
            boundary_mask=coarse_boundary_mask,
            gx_matrix=gx_matrix,
            gy_matrix=gy_matrix,
            depth_matrix=depth_matrix,
            primary_feature_mode=combo.primary_feature_mode,
            direction_dispersion_threshold=SECONDARY_DIRECTION_DISPERSION_MIN_THRESHOLD,
            min_partition_area_ratio_for_secondary=SECONDARY_MIN_AREA_RATIO_FOR_SECONDARY,
            cell_effective_area=coarse_cell_effective_area,
        )

        planner = SurveyPlanner(
            xs,
            ys,
            final_cluster_matrix,
            depth_matrix=depth_matrix,
            x_min=X_MIN,
            x_max=X_MAX,
            y_min=Y_MIN,
            y_max=Y_MAX,
            boundary_mask=coarse_boundary_mask,
            cell_effective_area=coarse_cell_effective_area,
            grid_cell_size=d_optimal,
            start_point_strategy=START_POINT_STRATEGY,
            line_gain_threshold_mode=args.line_gain_mode,
            jump_line_gain_threshold=args.line_gain_threshold,
            child_line_parent_gain_factor=CHILD_LINE_PARENT_GAIN_FACTOR,
            child_line_min_gain_threshold=CHILD_LINE_MIN_GAIN_THRESHOLD,
        )
        planner.plan_all(output_dir=str(combo_output_dir))

        metrics_path = combo_output_dir / "metrics" / "metrics.xlsx"
        summary_row = {
            "combo": combo.display_name,
            "combo_label": combo.label,
            "primary_feature_mode": combo.primary_feature_mode,
            "secondary_feature": "gradient-direction",
            "final_partition_u": int(final_u),
            "start_point_strategy": START_POINT_STRATEGY,
            "line_gain_rule_label": line_gain_rule_label,
            "line_gain_threshold_mode": args.line_gain_mode,
            "line_gain_fixed_threshold": (
                args.line_gain_threshold if args.line_gain_mode == "fixed" else ""
            ),
            "line_gain_parent_factor": (
                CHILD_LINE_PARENT_GAIN_FACTOR if args.line_gain_mode == "adaptive" else ""
            ),
            "line_gain_floor": (
                CHILD_LINE_MIN_GAIN_THRESHOLD if args.line_gain_mode == "adaptive" else ""
            ),
            "output_dir": str(combo_output_dir),
        }
        summary_row.update(load_global_metrics(metrics_path))
        summary_rows.append(summary_row)

    summary_name = f"{run_timestamp}_comparison_summary.csv"
    if args.line_gain_mode == "fixed":
        summary_name = f"{run_timestamp}_comparison_{line_gain_rule_label}_summary.csv"
    summary_path = output_root / summary_name
    write_summary(summary_rows, summary_path)
    print("\n" + "=" * 80)
    print(f"[对比实验] 完成。汇总文件: {summary_path}")


if __name__ == "__main__":
    main()
