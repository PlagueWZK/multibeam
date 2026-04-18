import argparse
import csv
import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from multibeam.GridCell import calculate_mesh_size_search_trace


def export_d_search_errors(data_path, min_error=0.001):
    result = calculate_mesh_size_search_trace(data_path, min_error=min_error)

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("./multibeam/output") / current_time
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "d_search_error_trace.csv"

    fieldnames = [
        "d_m",
        "radius_grid_count",
        "relative_error",
        "relative_error_percent",
        "patent_error",
        "meets_threshold",
        "grid_points_in_circle",
        "covered_area_B",
        "theoretical_area_A",
        "is_raw_spacing_reference",
        "is_search_candidate",
    ]

    with output_file.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result["trace_records"])

    figure_file = output_dir / "d_search_error_trace.png"
    plot_d_search_errors(result, figure_file)

    print(f"输出文件: {output_file.resolve()}")
    print(f"统计图文件: {figure_file.resolve()}")
    print(
        f"原始网格边长: {result['raw_spacing_nm']:.4f} 海里 = {result['raw_spacing_m']:.2f} m"
    )
    print(f"邻域覆盖半径(ξ)对应的理论起点 d: {result['xi_start_d']:.2f} m")
    print(f"受原始网格上界约束的搜索起始 d: {result['start_d']:.2f} m")
    print(f"相对面积误差阈值: {result['relative_error_threshold'] * 100:.4f}%")
    if result["optimal_d"] is not None:
        print(f"满足条件的最优 d: {result['optimal_d']:.2f} m")
    else:
        print("未找到满足条件的最优 d，但已导出完整搜索轨迹。")

    return output_file


def plot_d_search_errors(result, figure_file):
    trace_records = result["trace_records"]
    d_values = [record["d_m"] for record in trace_records]
    relative_error_percent = [
        record["relative_error_percent"] for record in trace_records
    ]
    patent_error = [record["patent_error"] for record in trace_records]

    relative_threshold_percent = result["relative_error_threshold"] * 100
    patent_threshold = result["patent_error_threshold"]

    optimal_record = next(
        (record for record in trace_records if record["meets_threshold"]),
        None,
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)

    axes[0].plot(d_values, relative_error_percent, marker="o", linewidth=1.8)
    axes[0].axhline(
        relative_threshold_percent,
        color="red",
        linestyle="--",
        label=f"Relative error threshold = {relative_threshold_percent:.4f}%",
    )
    if optimal_record is not None:
        axes[0].scatter(
            [optimal_record["d_m"]],
            [optimal_record["relative_error_percent"]],
            color="green",
            s=80,
            zorder=3,
            label=f"Matched d = {optimal_record['d_m']:.2f} m",
        )
    axes[0].set_title("Relative area error during d search")
    axes[0].set_xlabel("d (m)")
    axes[0].set_ylabel("Relative Error (%)")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].legend()
    axes[0].invert_xaxis()

    axes[1].plot(d_values, patent_error, marker="o", linewidth=1.8, color="#ff7f0e")
    axes[1].axhline(
        patent_threshold,
        color="red",
        linestyle="--",
        label=f"Patent error threshold = {patent_threshold:.4f}",
    )
    if optimal_record is not None:
        axes[1].scatter(
            [optimal_record["d_m"]],
            [optimal_record["patent_error"]],
            color="green",
            s=80,
            zorder=3,
            label=f"Matched d = {optimal_record['d_m']:.2f} m",
        )
    axes[1].set_title("Patent error during d search")
    axes[1].set_xlabel("d (m)")
    axes[1].set_ylabel("Patent Error")
    axes[1].grid(True, linestyle=":", alpha=0.5)
    axes[1].legend()
    axes[1].invert_xaxis()

    fig.suptitle("D-search error trace", fontsize=14)
    fig.savefig(figure_file, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="导出最优 d 搜索误差轨迹")
    parser.add_argument(
        "--data-path",
        default=str(Path(__file__).resolve().parent / "data" / "data.xlsx"),
        help="输入数据 Excel 路径",
    )
    parser.add_argument(
        "--min-error",
        type=float,
        default=0.001,
        help="允许的相对面积误差上限",
    )
    args = parser.parse_args()

    export_d_search_errors(Path(args.data_path), min_error=args.min_error)


if __name__ == "__main__":
    main()
