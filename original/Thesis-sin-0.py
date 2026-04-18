"""
脚本用途：第四问(6)。
主要包含数据处理、计算与可视化步骤。
"""

import copy
import datetime
import sys
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from joblib import load
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
DATA_DIR = REPO_ROOT / "data"

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multibeam.GridCell import calculate_optimal_mesh_size, generate_coordinate_array
from multibeam.coverage_state_grid import PartitionCoverageStateGrid

X_MIN, X_MAX = 0, 4 * 1852
Y_MIN, Y_MAX = 0, 5 * 1852
SEA_AREA = float((X_MAX - X_MIN) * (Y_MAX - Y_MIN))

FINE_GRID_STATE_STYLES = {
    "full": {"color": "#4CAF50", "alpha": 0.18, "label": "Fine grid - full"},
    "partial": {
        "color": "#FFB300",
        "alpha": 0.22,
        "label": "Fine grid - partial",
    },
    "uncovered": {
        "color": "#E53935",
        "alpha": 0.18,
        "label": "Fine grid - uncovered",
    },
}


def create_timestamped_output_dir():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = BASE_DIR / "output" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def ensure_models():  # 确保所需的机器学习模型文件存在
    height_path = DATA_DIR / "height_random_forest_model.pkl"
    gx_path = DATA_DIR / "gx_random_forest_model.pkl"
    gy_path = DATA_DIR / "gy_random_forest_model.pkl"
    missing = [
        str(path) for path in (height_path, gx_path, gy_path) if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "缺少共享随机森林模型文件，请先确认 data/ 下模型已生成: "
            + ", ".join(missing)
        )


@lru_cache(maxsize=None)
def get_alpha_with_microstep(x, y, microstep):
    tx1 = x + microstep * get_gx(x, y)
    ty1 = y + microstep * get_gy(x, y)
    h1 = get_height(tx1, ty1)
    tx2 = x - microstep * get_gx(x, y)
    ty2 = y - microstep * get_gy(x, y)
    h2 = get_height(tx2, ty2)
    return float(np.arctan((abs(h1 - h2)) / (2 * microstep)) * 180 / np.pi)


@lru_cache(maxsize=None)
def get_shared_w_left(x, y, theta=120, microstep=35.0):
    depth = get_height(x, y)
    alpha = get_alpha_with_microstep(x, y, microstep)
    return (depth * sin(theta / 2) / sin(90 - theta / 2 - alpha)) * cos(alpha)


@lru_cache(maxsize=None)
def get_shared_w_right(x, y, theta=120, microstep=35.0):
    depth = get_height(x, y)
    alpha = get_alpha_with_microstep(x, y, microstep)
    return (depth * sin(theta / 2) / sin(90 - theta / 2 + alpha)) * cos(alpha)


def build_line_with_shared_width(line_xy, theta=120, microstep=35.0):
    line_arr = np.asarray(line_xy, dtype=float)
    if line_arr.ndim != 2 or len(line_arr) < 2:
        return None

    widths = []
    for x, y in line_arr:
        widths.append(
            get_shared_w_left(float(x), float(y), theta=theta, microstep=microstep)
            + get_shared_w_right(float(x), float(y), theta=theta, microstep=microstep)
        )
    return np.column_stack(
        [line_arr[:, 0], line_arr[:, 1], np.asarray(widths, dtype=float)]
    )


def compute_parent_child_overlap_rate(parent_point, child_point, theta=120):
    parent_arr = np.asarray(parent_point, dtype=float)
    child_arr = np.asarray(child_point, dtype=float)
    x_i, y_i = float(parent_arr[0]), float(parent_arr[1])
    x_j, y_j = float(child_arr[0]), float(child_arr[1])

    d_i = get_height(x_i, y_i)
    d_j = get_height(x_j, y_j)
    alpha_i = np.radians(get_alpha(x_i, y_i))
    alpha_j = np.radians(get_alpha(x_j, y_j))
    theta_rad = np.radians(theta)
    sin_half = np.sin(theta_rad / 2.0)

    denom_i_left = np.sin((np.pi - theta_rad) / 2.0 + alpha_i)
    denom_i_right = np.sin((np.pi - theta_rad) / 2.0 - alpha_i)
    denom_j_right = np.sin((np.pi - theta_rad) / 2.0 - alpha_j)
    cos_alpha_i = np.cos(alpha_i)

    if (
        abs(denom_i_left) <= 1e-12
        or abs(denom_i_right) <= 1e-12
        or abs(denom_j_right) <= 1e-12
        or abs(cos_alpha_i) <= 1e-12
    ):
        return 0.0

    overlap_width = (
        d_i * sin_half / denom_i_left
        + d_j * sin_half / denom_j_right
        - np.hypot(x_i - x_j, y_i - y_j) / cos_alpha_i
    )
    coverage_width_i = d_i * sin_half * (1.0 / denom_i_left + 1.0 / denom_i_right)

    if not np.isfinite(overlap_width) or not np.isfinite(coverage_width_i):
        return 0.0
    if coverage_width_i <= 1e-12:
        return 0.0
    return float(max(overlap_width / coverage_width_i, 0.0))


def compute_overlap_excess_length(child_points, overlap_rates, threshold=0.2):
    line_arr = np.asarray(child_points, dtype=float)
    rates = np.asarray(overlap_rates, dtype=float)
    if line_arr.ndim != 2 or len(line_arr) < 2 or len(rates) < 2:
        return 0.0

    total_length = 0.0
    for point0, point1, rate0, rate1 in zip(
        line_arr[:-1], line_arr[1:], rates[:-1], rates[1:]
    ):
        segment_length = float(np.hypot(point1[0] - point0[0], point1[1] - point0[1]))
        if segment_length <= 0 or (not np.isfinite(rate0)) or (not np.isfinite(rate1)):
            continue

        if rate0 <= threshold and rate1 <= threshold:
            continue
        if rate0 > threshold and rate1 > threshold:
            total_length += segment_length
            continue
        if abs(rate1 - rate0) <= 1e-12:
            if rate0 > threshold:
                total_length += segment_length
            continue

        crossing_ratio = (threshold - rate0) / (rate1 - rate0)
        crossing_ratio = float(np.clip(crossing_ratio, 0.0, 1.0))
        if rate0 > threshold and rate1 <= threshold:
            total_length += segment_length * crossing_ratio
        elif rate0 <= threshold and rate1 > threshold:
            total_length += segment_length * (1.0 - crossing_ratio)

    return float(total_length)


def save_fine_grid_overlay_figure(output_dir, state_grid, rendered_lines):
    state_code_matrix = np.asarray(state_grid.state_code_matrix, dtype=float)
    display_matrix = np.ma.masked_where(state_code_matrix < 0, state_code_matrix)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Thesis Survey Lines - Fine Grid View")
    ax.grid(True, alpha=0.3)

    cmap = ListedColormap(
        [
            FINE_GRID_STATE_STYLES["uncovered"]["color"],
            FINE_GRID_STATE_STYLES["partial"]["color"],
            FINE_GRID_STATE_STYLES["full"]["color"],
        ]
    )
    ax.imshow(
        display_matrix,
        extent=[X_MIN, X_MAX, Y_MAX, Y_MIN],
        cmap=cmap,
        vmin=0,
        vmax=2,
        aspect="auto",
        alpha=0.25,
        interpolation="nearest",
        zorder=0,
    )

    for line in rendered_lines:
        line_arr = np.asarray(line, dtype=float)
        if line_arr.ndim != 2 or len(line_arr) < 2:
            continue
        ax.plot(
            line_arr[:, 0],
            line_arr[:, 1],
            color="#1f77b4",
            linewidth=0.9,
            alpha=0.85,
            zorder=2,
        )

    ax.legend(
        handles=[
            Line2D([0], [0], color="#1f77b4", linewidth=1.5, label="Survey lines"),
            Patch(
                facecolor=FINE_GRID_STATE_STYLES["full"]["color"],
                alpha=FINE_GRID_STATE_STYLES["full"]["alpha"],
                edgecolor="none",
                label=FINE_GRID_STATE_STYLES["full"]["label"],
            ),
            Patch(
                facecolor=FINE_GRID_STATE_STYLES["partial"]["color"],
                alpha=FINE_GRID_STATE_STYLES["partial"]["alpha"],
                edgecolor="none",
                label=FINE_GRID_STATE_STYLES["partial"]["label"],
            ),
            Patch(
                facecolor=FINE_GRID_STATE_STYLES["uncovered"]["color"],
                alpha=FINE_GRID_STATE_STYLES["uncovered"]["alpha"],
                edgecolor="none",
                label=FINE_GRID_STATE_STYLES["uncovered"]["label"],
            ),
        ],
        loc="best",
    )

    fine_grid_view_path = output_dir / "sin-0-fine-grid-view.png"
    fig.savefig(fine_grid_view_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"细网格状态图已保存到: {fine_grid_view_path}")
    return fine_grid_view_path


def compute_polyline_hit_mask(state_grid, line_with_width):
    line_arr = np.asarray(line_with_width, dtype=float)
    hit_mask = np.zeros_like(state_grid.partition_sample_mask, dtype=bool)
    if len(line_arr) < 2:
        return hit_mask

    for start, end in zip(line_arr[:-1], line_arr[1:]):
        radius = max(float(start[2]), float(end[2])) / 2.0
        if radius <= 0:
            continue

        x_low = min(float(start[0]), float(end[0])) - radius
        x_high = max(float(start[0]), float(end[0])) + radius
        y_low = min(float(start[1]), float(end[1])) - radius
        y_high = max(float(start[1]), float(end[1])) + radius

        row_start, row_end = state_grid._candidate_index_range(
            state_grid.y_centers, y_low, y_high
        )
        col_start, col_end = state_grid._candidate_index_range(
            state_grid.x_centers, x_low, x_high
        )

        if row_end < row_start or col_end < col_start:
            continue

        for i in range(row_start, row_end + 1):
            for j in range(col_start, col_end + 1):
                if state_grid.partition_sample_counts[i, j] == 0:
                    continue

                sample_indices = np.where(state_grid.partition_sample_mask[i, j])[0]
                if len(sample_indices) == 0:
                    continue

                covered_now = state_grid._points_within_segment_swath(
                    state_grid.sample_x[i, j, sample_indices],
                    state_grid.sample_y[i, j, sample_indices],
                    start,
                    end,
                )
                if np.any(covered_now):
                    hit_mask[i, j, sample_indices[covered_now]] = True

    return hit_mask


def compute_repeat_area_for_polyline(state_grid, line_with_width):
    line_hit_mask = compute_polyline_hit_mask(state_grid, line_with_width)
    repeated_mask = (
        line_hit_mask & state_grid.covered_sample_mask & state_grid.partition_sample_mask
    )
    repeated_counts = np.sum(repeated_mask, axis=2)
    valid_domain_cells = state_grid.domain_sample_counts > 0
    repeated_ratio_against_domain = np.zeros_like(
        state_grid.partition_sample_counts, dtype=float
    )
    repeated_ratio_against_domain[valid_domain_cells] = (
        repeated_counts[valid_domain_cells] / state_grid.domain_sample_counts[valid_domain_cells]
    )
    return float(
        np.sum(repeated_ratio_against_domain * state_grid.cell_domain_area)
    )


def initialize_metrics_context(theta=120, overlap_rate=0.1):
    d_optimal = calculate_optimal_mesh_size(DATA_DIR / "data.xlsx")
    xs, ys, boundary_mask, cell_effective_area, _ = generate_coordinate_array(
        X_MIN, X_MAX, Y_MIN, Y_MAX, d_optimal
    )
    cluster_matrix = np.ones(boundary_mask.shape, dtype=int)
    state_grid = PartitionCoverageStateGrid(
        1,
        xs,
        ys,
        cluster_matrix,
        x_min=X_MIN,
        x_max=X_MAX,
        y_min=Y_MIN,
        y_max=Y_MAX,
        boundary_mask=boundary_mask,
        cell_effective_area=cell_effective_area,
        cell_size=d_optimal,
        sampling_points=9,
        full_threshold=1.0,
    )

    return {
        "theta": theta,
        "overlap_rate": overlap_rate,
        "d_optimal": d_optimal,
        "state_grid": state_grid,
        "rendered_lines": [],
        "line_rows": [],
        "total_length": 0.0,
        "total_old_proxy_coverage": 0.0,
        "line_counter": 0,
        "repeated_area_total": 0.0,
        "max_overlap_eta_by_group": [0.0, 0.0, 0.0, 0.0, 0.0],
        "overlap_excess_length_by_group": [0.0, 0.0, 0.0, 0.0, 0.0],
    }


def register_rendered_line(metrics_context, line, group_index):
    line_arr = np.asarray(line, dtype=float)
    if line_arr.ndim != 2 or len(line_arr) < 2:
        return 0.0, 0.0

    line_length = figure_length(line_arr)
    old_proxy_area = figure_width(line_arr)
    metrics_context["total_length"] += line_length
    metrics_context["total_old_proxy_coverage"] += old_proxy_area
    metrics_context["line_counter"] += 1
    metrics_context["rendered_lines"].append(line_arr.copy())

    shared_line = build_line_with_shared_width(
        line_arr,
        theta=metrics_context["theta"],
        microstep=metrics_context["state_grid"].microstep,
    )
    if shared_line is not None:
        metrics_context["repeated_area_total"] += compute_repeat_area_for_polyline(
            metrics_context["state_grid"], shared_line
        )
        metrics_context["state_grid"].update_polyline(shared_line)

    metrics_context["line_rows"].append(
        {
            "测线ID": metrics_context["line_counter"],
            "线组ID": int(group_index) + 1,
            "点数": len(line_arr),
            "测线长度(m)": round(line_length, 2),
            "积分覆盖面积-旧口径(m²)": round(old_proxy_area, 2),
        }
    )
    return line_length, old_proxy_area


def add_overlap_excess_length(
    metrics_context, group_index, child_points, overlap_rates, threshold=0.2
):
    rate_arr = np.asarray(overlap_rates, dtype=float)
    if len(rate_arr) > 0:
        metrics_context["max_overlap_eta_by_group"][group_index] = max(
            metrics_context["max_overlap_eta_by_group"][group_index],
            float(np.max(rate_arr)),
        )
    overlap_excess = compute_overlap_excess_length(
        child_points, rate_arr, threshold=threshold
    )
    metrics_context["overlap_excess_length_by_group"][group_index] += overlap_excess


def save_metrics_report(output_dir, metrics_context):
    state_grid = metrics_context["state_grid"]
    d_optimal = metrics_context["d_optimal"]
    line_rows = metrics_context["line_rows"]
    total_length = metrics_context["total_length"]
    total_old_proxy_coverage = metrics_context["total_old_proxy_coverage"]
    repeated_area_total = metrics_context["repeated_area_total"]
    max_overlap_eta_by_group = metrics_context["max_overlap_eta_by_group"]
    overlap_excess_length_by_group = metrics_context["overlap_excess_length_by_group"]

    summary = state_grid.summarize()
    old_effective_coverage = total_old_proxy_coverage * (1 - metrics_context["overlap_rate"])
    old_coverage_rate = old_effective_coverage / SEA_AREA if SEA_AREA > 0 else 0.0
    new_coverage_rate = summary.coverage_rate
    new_miss_rate = summary.miss_rate
    overlap_excess_total = float(sum(overlap_excess_length_by_group))
    max_overlap_eta = float(max(max_overlap_eta_by_group)) if max_overlap_eta_by_group else 0.0
    repeated_rate = repeated_area_total / SEA_AREA if SEA_AREA > 0 else 0.0

    global_rows = [
        {"指标": "海域定义", "值": "x=0-4海里, y=0-5海里"},
        {"指标": "矩形海域总面积(m²)", "值": round(SEA_AREA, 2)},
        {"指标": "测线条数", "值": len(line_rows)},
        {"指标": "测线总长度(m)", "值": round(total_length, 2)},
        {"指标": "积分总覆盖面积-旧口径(m²)", "值": round(total_old_proxy_coverage, 2)},
        {"指标": "积分有效覆盖面积-旧口径(m²)", "值": round(old_effective_coverage, 2)},
        {"指标": "积分覆盖率-旧口径", "值": f"{old_coverage_rate:.4%}"},
        {"指标": "积分漏测率-旧口径(可能为负)", "值": f"{1 - old_coverage_rate:.4%}"},
        {"指标": "重叠率超过20%部分总长度(m)", "值": round(overlap_excess_total, 2)},
        {"指标": "parent-child公式下最大η(诊断)", "值": round(max_overlap_eta, 4)},
        {
            "指标": "重叠率阈值长度折算规则",
            "值": "父子对应点η按线性插值折算到相邻子测线段，累计η>20%的长度",
        },
        {"指标": "统一细网格边长d(m)", "值": round(d_optimal, 4)},
        {"指标": "统一细网格总面积(m²)", "值": round(summary.total_area, 2)},
        {
            "指标": "细网格待测面积偏差-诊断值(m²)",
            "值": round(summary.total_area - SEA_AREA, 2),
        },
        {"指标": "细网格真实覆盖面积-新口径(m²)", "值": round(summary.covered_area, 2)},
        {"指标": "细网格重复测量面积-新口径(m²)", "值": round(repeated_area_total, 2)},
        {"指标": "累积重复率-新口径", "值": f"{repeated_rate:.4%}"},
        {"指标": "细网格覆盖率-新口径", "值": f"{new_coverage_rate:.4%}"},
        {"指标": "细网格漏测率-新口径", "值": f"{new_miss_rate:.4%}"},
        {"指标": "9点full判定阈值", "值": "1.0"},
        {"指标": "细网格规划步长step(m)", "值": round(summary.planning_step, 2)},
        {"指标": "细网格坡度差分microstep(m)", "值": round(summary.microstep, 2)},
        {"指标": "full_cells", "值": summary.full_cells},
        {"指标": "partial_cells", "值": summary.partial_cells},
        {"指标": "uncovered_cells", "值": summary.uncovered_cells},
    ]

    for group_index, group_length in enumerate(overlap_excess_length_by_group, start=1):
        global_rows.append(
            {
                "指标": f"第{group_index}组-重叠率超过20%部分长度(m)",
                "值": round(group_length, 2),
            }
        )
        global_rows.append(
            {
                "指标": f"第{group_index}组-parent-child最大η(诊断)",
                "值": round(max_overlap_eta_by_group[group_index - 1], 4),
            }
        )

    summary_dict = asdict(summary)
    for key, value in summary_dict.items():
        if key == "partition_id":
            continue
        if key == "repeated_area":
            value = repeated_area_total
        global_rows.append(
            {
                "指标": f"coverage_summary.{key}",
                "值": round(value, 4) if isinstance(value, float) else value,
            }
        )

    metrics_path = output_dir / "metrics.xlsx"
    with pd.ExcelWriter(metrics_path, engine="openpyxl") as writer:
        pd.DataFrame(line_rows).to_excel(writer, sheet_name="测线统计", index=False)
        pd.DataFrame(global_rows).to_excel(writer, sheet_name="全局统计", index=False)

    fine_grid_view_path = save_fine_grid_overlay_figure(
        output_dir, state_grid, metrics_context["rendered_lines"]
    )
    print(f"指标统计已保存到: {metrics_path}")
    return metrics_path, fine_grid_view_path


# 确保模型文件存在并加载模型
ensure_models()  # 调用函数确保模型存在
OUTPUT_DIR = create_timestamped_output_dir()
print(f"本次输出目录: {OUTPUT_DIR}")
# 加载深度预测模型
height_rf = load(
    str(DATA_DIR / "height_random_forest_model.pkl")
)  # 加载深度预测随机森林模型
# 加载x方向梯度预测模型
gx_rf = load(
    str(DATA_DIR / "gx_random_forest_model.pkl")
)  # 加载x方向梯度预测随机森林模型
# 加载y方向梯度预测模型
gy_rf = load(
    str(DATA_DIR / "gy_random_forest_model.pkl")
)  # 加载y方向梯度预测随机森林模型


# 定义获取指定位置深度的函数
@lru_cache(maxsize=None)
def get_height(x, y):  # 输入x、y坐标，返回该位置的深度
    return float(height_rf.predict([[x, y]])[0])  # 使用深度预测模型预测深度


# 定义获取指定位置x方向梯度的函数
@lru_cache(maxsize=None)
def get_gx(x, y):  # 输入x、y坐标，返回该位置的x方向梯度
    return float(gx_rf.predict([[x, y]])[0])  # 使用x方向梯度预测模型预测梯度


# 定义获取指定位置y方向梯度的函数
@lru_cache(maxsize=None)
def get_gy(x, y):  # 输入x、y坐标，返回该位置的y方向梯度
    return float(gy_rf.predict([[x, y]])[0])  # 使用y方向梯度预测模型预测梯度


# 定义计算指定位置坡度角的函数
@lru_cache(maxsize=None)
def get_alpha(x, y):  # 输入x、y坐标，返回该位置的坡度角
    # 使用与原始网格分辨率一致的步长，避免随机森林在极小位移下输出不变
    step = 0.0001  # 约等于 1 海里 / 50，对应原始数据网格间距（米）
    # 计算沿梯度方向前进一步后的坐标
    tx1 = x + step * get_gx(x, y)  # 沿x方向梯度前进
    ty1 = y + step * get_gy(x, y)  # 沿y方向梯度前进
    # 获取前进后位置的深度
    h1 = get_height(tx1, ty1)  # 前进后的深度
    # 计算沿梯度方向后退一步后的坐标
    tx2 = x - step * get_gx(x, y)  # 沿x方向梯度后退
    ty2 = y - step * get_gy(x, y)  # 沿y方向梯度后退
    # 获取后退后位置的深度
    h2 = get_height(tx2, ty2)  # 后退后的深度
    # 计算坡度角（使用反正切函数）
    return float(
        np.arctan((abs(h1 - h2)) / (2 * step)) * 180 / np.pi
    )  # 坡度角计算（弧度转角度）


# 定义正弦函数（角度转弧度）
def sin(a):  # 输入角度，返回正弦值
    return np.sin(np.radians(a))  # 将角度转换为弧度后计算正弦值


# 定义余弦函数（角度转弧度）
def cos(a):  # 输入角度，返回余弦值
    return np.cos(np.radians(a))  # 将角度转换为弧度后计算余弦值


# 定义正切函数（角度转弧度）
def tan(a):  # 输入角度，返回正切值
    return np.tan(np.radians(a))  # 将角度转换为弧度后计算正切值


# 定义计算指定位置左侧覆盖宽度的函数
@lru_cache(maxsize=None)
def get_Wleft(x, y):  # 输入x、y坐标，返回该位置的左侧覆盖宽度
    # 获取该位置的深度
    D = get_height(x, y)  # 获取深度
    # 获取该位置的坡度角
    alpha = get_alpha(x, y)  # 获取坡度角
    # 计算左侧覆盖宽度
    return (D * sin(theta / 2) / sin(90 - theta / 2 - alpha)) * cos(
        alpha
    )  # 左侧覆盖宽度计算公式(转化为水平标量)


# 定义计算指定位置右侧覆盖宽度的函数
@lru_cache(maxsize=None)
def get_WRight(x, y):  # 输入x、y坐标，返回该位置的右侧覆盖宽度
    # 获取该位置的深度
    D = get_height(x, y)  # 获取深度
    # 获取该位置的坡度角
    alpha = get_alpha(x, y)  # 获取坡度角
    # 计算右侧覆盖宽度
    return (D * sin(theta / 2) / sin(90 - theta / 2 + alpha)) * cos(
        alpha
    )  # 右侧覆盖宽度计算公式(转化为水平标量)


# 定义计算前进方向的函数
def forward_direction(gx, gy):  # 输入梯度分量，返回垂直于梯度的前进方向
    return (-gy, gx)  # 返回垂直于梯度方向的向量


# 定义计算测线长度的函数
def figure_length(line):  # 输入测线路径点列表，返回测线总长度
    total = 0.0
    for i in range(len(line) - 1):
        total += np.sqrt(
            (line[i][0] - line[i + 1][0]) ** 2 + (line[i][1] - line[i + 1][1]) ** 2
        )
    return total


# 定义计算测线覆盖面积的函数
def figure_width(line):  # 输入测线路径点列表，返回测线覆盖的总面积
    total = 0.0
    for i in range(len(line) - 1):
        total += np.sqrt(
            (line[i][0] - line[i + 1][0]) ** 2 + (line[i][1] - line[i + 1][1]) ** 2
        ) * (
            get_WRight(line[i][0], line[i][1]) + get_Wleft(line[i][0], line[i][1])
        )
    return total


# 初始化测量点数组，用于存储所有测量点坐标
dot = np.empty((0, 2))  # 创建空数组，避免导出伪原点
# 设置重叠率参数
n = 0.1  # 测线重叠率设为10%
# 创建固定大小的单一画布，确保所有测线绘制在同一张图上
plt.figure(figsize=(10, 8))

# 第一条测线：起始位置(1700, 10)
# 初始化第一条测线路径列表
line = []
# 设置起始位置x坐标
loc_x = 1700
# 设置起始位置y坐标
loc_y = 10
# 设置前进步长
step = 50  # 每步前进50米
# 设置换能器开角
theta = 120  # 多波束换能器开角120度
metrics_context = initialize_metrics_context(theta=theta, overlap_rate=n)
# 生成第一条主测线路径
while True:  # 无限循环直到满足退出条件
    # 获取当前位置的x方向梯度
    gx = get_gx(loc_x, loc_y)  # 调用函数获取x方向梯度
    # 获取当前位置的y方向梯度
    gy = get_gy(loc_x, loc_y)  # 调用函数获取y方向梯度
    # 计算前进方向向量
    dx, dy = forward_direction(gx, gy)  # 获取垂直于梯度的前进方向
    # 更新当前位置x坐标
    loc_x += step * dx  # 新x坐标 = 原x坐标 + 步长 * x方向分量
    # 更新当前位置y坐标
    loc_y += step * dy  # 新y坐标 = 原y坐标 + 步长 * y方向分量
    # 检查是否超出测量区域边界
    if (
        loc_x > 4 * 1852 or loc_y > 5 * 1852 or loc_y < 0 or loc_x < 0
    ):  # 边界检查：4海里×5海里区域
        break  # 超出边界则退出循环
    # 将当前位置添加到测线路径
    line.append([loc_x, loc_y])  # 记录路径点坐标
# 将路径列表转换为numpy数组
line = np.array(line)  # 转换为numpy数组便于后续处理
# 绘制第一条主测线
plt.plot(line[:-1, 0], line[:-1, 1], color="b")  # 用蓝色绘制主测线路径
# 为第一条主测线生成垂直测线
while True:  # 无限循环直到所有垂直测线生成完毕
    # 初始化下一轮测线点列表
    t1 = []  # 存储下一轮垂直测线的点坐标
    t1_overlap_rates = []
    # 遍历当前测线上的每个点
    for index, i in enumerate(line):  # 对测线上每个点进行处理
        # 获取当前点的x坐标
        x = i[0]  # 提取x坐标
        # 获取当前点的y坐标
        y = i[1]  # 提取y坐标
        # 计算当前位置的坡度角
        alpha = get_alpha(x, y)  # 获取海底坡度角
        # 获取当前位置的深度
        h = get_height(x, y)  # 获取海底深度
        # 根据坡度角大小选择不同的计算方法
        if alpha <= 0.005:  # 如果坡度角很小（近似平坦海底）
            # 使用简化公式计算测线间距
            d = 2 * h * tan(theta / 2) * (1 - n)  # 平坦海底的测线间距公式
            # 计算x方向的偏移量
            tx = d * get_gx(x, y)  # x偏移 = 间距 * x方向梯度
            # 计算y方向的偏移量
            ty = d * get_gy(x, y)  # y偏移 = 间距 * y方向梯度
            # 更新x坐标
            x = x + tx  # 新x坐标 = 原x坐标 + x偏移
            # 更新y坐标
            y = y + ty  # 新y坐标 = 原y坐标 + y偏移
        else:  # 如果坡度角较大（倾斜海底）
            # 计算几何参数A
            A = sin(90 - theta / 2 + alpha)  # 几何参数A
            # 计算几何参数B
            B = sin(90 - theta / 2 - alpha)  # 几何参数B
            # 计算迭代参数C
            C = sin(theta / 2) / A - 1 / sin(alpha)  # 迭代公式中的参数C
            # 计算迭代参数D（注意换行符的处理）
            D = (
                n * sin(theta / 2) * (1 / A + 1 / B)
                - sin(theta / 2) / B
                - 1 / sin(alpha)
            )  # 迭代公式中的参数D

            # 计算下一个测线位置的深度
            next_h = h * C / D  # 根据迭代公式计算新深度
            # 计算x方向的偏移量
            tx = (h - next_h) / tan(alpha) * get_gx(x, y)
            # 计算y方向的偏移量
            ty = (h - next_h) / tan(alpha) * get_gy(x, y)
            # 更新x坐标
            x = x + tx  # 新x坐标 = 原x坐标 + x偏移
            # 更新y坐标
            y = y + ty  # 新y坐标 = 原y坐标 + y偏移
        # 检查新位置是否在有效测量范围内
        if (
            y < 0 or x < 0 or get_height(x, y) < 21 or get_height(x, y) > 39
        ):  # 边界和深度范围检查
            pass
        else:  # 如果在有效范围内
            # 将新位置添加到下一轮测线点列表
            child_point = [x, y]
            t1.append(child_point)  # 记录有效的测线点
            t1_overlap_rates.append(
                compute_parent_child_overlap_rate(i, child_point, theta=theta)
            )
    add_overlap_excess_length(metrics_context, 0, t1, t1_overlap_rates, threshold=0.2)
    # 如果当前测线长度足够，进行统计和绘制
    if len(line) > 5:  # 测线点数大于5时才进行统计
        plotted_line = line[:-1]
        # 绘制当前垂直测线
        plt.plot(plotted_line[:, 0], plotted_line[:, 1], color="lightgray")  # 用浅灰色绘制垂直测线
        # 记录测线长度统计
        register_rendered_line(metrics_context, plotted_line, 0)
    dot = np.concatenate((dot, line), axis=0)  # 合并所有测量点
    if len(t1) == 0:  # 检查是否还有测线点
        break  # 退出垂直测线生成循环
    # 深拷贝下一轮测线点
    line = copy.deepcopy(t1)  # 深拷贝避免引用问题
    # 从最后一个测线点继续生成主测线
    loc_x = line[-1][0]  # 获取最后一个点的x坐标作为新起点
    # 获取最后一个测线点的y坐标
    loc_y = line[-1][1]  # 获取最后一个点的y坐标作为新起点
    # 调整步长为更小值
    step = 25  # 减小步长以提高精度
    # 继续生成主测线路径
    while True:  # 继续主测线生成循环
        # 获取当前位置的x方向梯度
        gx = get_gx(loc_x, loc_y)  # 计算x方向梯度
        # 获取当前位置的y方向梯度
        gy = get_gy(loc_x, loc_y)  # 计算y方向梯度
        # 计算前进方向向量
        dx, dy = forward_direction(gx, gy)  # 获取前进方向
        # 更新当前位置x坐标
        loc_x += step * dx  # 新x坐标 = 原x坐标 + 步长 * x方向分量
        # 更新当前位置y坐标
        loc_y += step * dy  # 新y坐标 = 原y坐标 + 步长 * y方向分量
        # 检查是否超出测量区域边界
        if loc_x > 4 * 1852 or loc_y > 5 * 1852 or loc_y < 0 or loc_x < 0:  # 边界检查
            break  # 超出边界则退出主测线生成
        # 将当前位置添加到测线路径
        line.append([loc_x, loc_y])  # 记录新的路径点
    # 将路径列表转换为numpy数组
    line = np.array(line)  # 转换为numpy数组

# 第二条测线：起始位置(3000, 10)
# 重新初始化测线路径列表
line = []  # 清空路径列表，准备生成第二条测线
# 设置第二条测线的起始x坐标
loc_x = 3000  # 起始x坐标为3000米
# 设置第二条测线的起始y坐标
loc_y = 10  # 起始y坐标为10米
# 设置前进步长
step = 50  # 每步前进50米
# 设置换能器开角
theta = 120  # 多波束换能器开角120度

# 生成第二条主测线路径
while True:  # 无限循环直到满足退出条件
    # 获取当前位置的x方向梯度
    gx = get_gx(loc_x, loc_y)  # 调用函数获取x方向梯度
    # 获取当前位置的y方向梯度
    gy = get_gy(loc_x, loc_y)  # 调用函数获取y方向梯度
    # 计算前进方向向量
    dx, dy = forward_direction(gx, gy)  # 获取垂直于梯度的前进方向
    # 更新当前位置x坐标
    loc_x += step * dx  # 新x坐标 = 原x坐标 + 步长 * x方向分量
    # 更新当前位置y坐标
    loc_y += step * dy  # 新y坐标 = 原y坐标 + 步长 * y方向分量
    # 检查是否超出测量区域边界
    if (
        loc_x > 4 * 1852 or loc_y > 5 * 1852 or loc_y < 0 or loc_x < 0
    ):  # 边界检查：4海里×5海里区域
        break  # 超出边界则退出循环
    # 将当前位置添加到测线路径
    line.append([loc_x, loc_y])  # 记录路径点坐标
# 将路径列表转换为numpy数组
line = np.array(line)  # 转换为numpy数组便于后续处理
# 绘制第二条主测线
plt.plot(line[:-1, 0], line[:-1, 1], color="r")  # 用红色绘制第二条主测线路径
# 循环迭代
# 为第二条主测线生成垂直测线
while True:  # 无限循环直到所有垂直测线生成完毕
    # 初始化下一轮测线点列表
    t1 = []  # 存储下一轮垂直测线的点坐标
    t1_overlap_rates = []
    # 遍历当前测线上的每个点
    for index, i in enumerate(line):  # 对测线上每个点进行处理
        # 获取当前点的x坐标
        x = i[0]  # 提取x坐标
        # 获取当前点的y坐标
        y = i[1]  # 提取y坐标
        # 计算当前位置的坡度角
        alpha = get_alpha(x, y)  # 获取海底坡度角
        # 获取当前位置的深度
        h = get_height(x, y)  # 获取海底深度
        # 根据坡度角大小选择不同的计算方法
        if alpha <= 0.005:  # 如果坡度角很小（近似平坦海底）
            # 使用简化公式计算测线间距
            d = 2 * h * tan(theta / 2) * (1 - n)  # 平坦海底的测线间距公式
            # 计算x方向的偏移量
            tx = d * get_gx(x, y)  # x偏移 = 间距 * x方向梯度
            # 计算y方向的偏移量
            ty = d * get_gy(x, y)  # y偏移 = 间距 * y方向梯度
            # 更新x坐标
            x = x + tx  # 新x坐标 = 原x坐标 + x偏移
            # 更新y坐标
            y = y + ty  # 新y坐标 = 原y坐标 + y偏移
        else:  # 如果坡度角较大（倾斜海底）
            # 计算几何参数A
            A = sin(90 - theta / 2 + alpha)  # 几何参数A
            # 计算几何参数B
            B = sin(90 - theta / 2 - alpha)  # 几何参数B
            # 计算迭代参数C
            C = sin(theta / 2) / A - 1 / sin(alpha)  # 迭代公式中的参数C
            # 计算迭代参数D（注意换行符的处理）
            D = (
                n * sin(theta / 2) * (1 / A + 1 / B)
                - sin(theta / 2) / B
                - 1 / sin(alpha)
            )  # 迭代公式中的参数D

            # 计算下一个测线位置的深度
            next_h = h * C / D  # 根据迭代公式计算新深度
            # 计算x方向的偏移量
            tx = (h - next_h) / tan(alpha) * get_gx(x, y)  # x偏移 = 深度差 * x方向梯度
            # 计算y方向的偏移量
            ty = (h - next_h) / tan(alpha) * get_gy(x, y)  # y偏移 = 深度差 * y方向梯度
            # 更新x坐标（注意方向相反）
            x = x + tx  # 新x坐标 = 原x坐标 - x偏移
            # 更新y坐标（注意方向相反）
            y = y + ty  # 新y坐标 = 原y坐标 - y偏移
        # 检查新位置是否在有效测量范围内（第二条测线的边界条件）
        if (
            x > 4 * 1852
            or y > 8100
            or y < 0
            or x < 0
            or get_height(x, y) > 75
            or 2500 * y - 9260 * (x - 400) < -37040000
        ):  # 复杂边界和深度检查
            pass
        else:  # 如果在有效范围内
            # 将新位置添加到下一轮测线点列表
            child_point = [x, y]
            t1.append(child_point)  # 记录有效的测线点
            t1_overlap_rates.append(
                compute_parent_child_overlap_rate(i, child_point, theta=theta)
            )
    add_overlap_excess_length(metrics_context, 1, t1, t1_overlap_rates, threshold=0.2)
    # 如果当前测线长度足够，进行统计和绘制
    if len(line) > 10:  # 测线点数大于10时才进行统计
        plotted_line = line[:-1]
        # 绘制当前垂直测线
        plt.plot(plotted_line[:, 0], plotted_line[:, 1], color="silver")  # 用深灰色绘制垂直测线
        # 记录测线长度统计
        register_rendered_line(metrics_context, plotted_line, 1)
    dot = np.concatenate((dot, line), axis=0)  # 合并所有测量点
    # 如果没有有效的下一轮测线点，退出循环
    if len(t1) == 0:  # 检查是否还有测线点
        break  # 退出垂直测线生成循环
    line = copy.deepcopy(t1)  # 深拷贝避免引用问题
    # 从最后一个测线点继续生成主测线
    loc_x = line[-1][0]  # 获取最后一个点的x坐标作为新起点
    # 获取最后一个测线点的y坐标
    loc_y = line[-1][1]  # 获取最后一个点的y坐标作为新起点
    # 设置步长
    step = 50  # 设置步长为50米
    # 继续生成主测线路径
    while True:  # 继续主测线生成循环
        # 获取当前位置的x方向梯度
        gx = get_gx(loc_x, loc_y)  # 计算x方向梯度
        # 获取当前位置的y方向梯度
        gy = get_gy(loc_x, loc_y)  # 计算y方向梯度
        # 计算前进方向向量
        dx, dy = forward_direction(gx, gy)  # 获取前进方向
        # 更新当前位置x坐标
        loc_x += step * dx  # 新x坐标 = 原x坐标 + 步长 * x方向分量
        # 更新当前位置y坐标
        loc_y += step * dy  # 新y坐标 = 原y坐标 + 步长 * y方向分量
        # 检查是否超出测量区域边界
        if (
            loc_x > 4 * 1852 or loc_y > 8000 or loc_y < 0 or loc_x < 0
        ):  # 边界检查（y上限为8000米）
            break  # 超出边界则退出主测线生成
        # 将当前位置添加到测线路径
        line.append([loc_x, loc_y])  # 记录新的路径点
    # 将路径列表转换为numpy数组
    line = np.array(line)  # 转换为numpy数组

# 第三条测线：起始位置(4500, 10)
# 重新初始化测线路径列表
line = []  # 清空路径列表，准备生成第三条测线
# 设置第三条测线的起始x坐标
loc_x = 4500  # 起始x坐标为4500米
# 设置第三条测线的起始y坐标
loc_y = 10  # 起始y坐标为10米
# 设置前进步长
step = 50  # 每步前进50米
# 设置换能器开角
theta = 120  # 多波束换能器开角120度
# 生成第三条主测线路径
while True:  # 无限循环直到满足退出条件
    # 获取当前位置的x方向梯度
    gx = get_gx(loc_x, loc_y)  # 调用函数获取x方向梯度
    # 获取当前位置的y方向梯度
    gy = get_gy(loc_x, loc_y)  # 调用函数获取y方向梯度
    # 计算前进方向向量
    dx, dy = forward_direction(gx, gy)  # 获取垂直于梯度的前进方向
    # 更新当前位置x坐标
    loc_x += step * dx  # 新x坐标 = 原x坐标 + 步长 * x方向分量
    # 更新当前位置y坐标
    loc_y += step * dy  # 新y坐标 = 原y坐标 + 步长 * y方向分量
    # 检查是否超出测量区域边界
    if (
        loc_x > 4 * 1852 or loc_y > 5 * 1852 or loc_y < 0 or loc_x < 0
    ):  # 边界检查：4海里×5海里区域
        break  # 超出边界则退出循环
    # 将当前位置添加到测线路径
    line.append([loc_x, loc_y])  # 记录路径点坐标
# 将路径列表转换为numpy数组
line = np.array(line)  # 转换为numpy数组便于后续处理
# 绘制第三条主测线
plt.plot(line[:-1, 0], line[:-1, 1], color="g")  # 用绿色绘制第三条主测线路径
# 为第三条主测线生成垂直测线
while True:  # 无限循环直到所有垂直测线生成完毕
    # 初始化下一轮测线点列表
    t1 = []  # 存储下一轮垂直测线的点坐标
    t1_overlap_rates = []
    # 遍历当前测线上的每个点
    for index, i in enumerate(line):  # 对测线上每个点进行处理
        # 获取当前点的x坐标
        x = i[0]  # 提取x坐标
        # 获取当前点的y坐标
        y = i[1]  # 提取y坐标
        # 计算当前位置的坡度角
        alpha = get_alpha(x, y)  # 获取海底坡度角
        # 获取当前位置的深度
        h = get_height(x, y)  # 获取海底深度
        # 根据坡度角大小选择不同的计算方法
        if alpha <= 0.005:  # 如果坡度角很小（近似平坦海底）
            # 使用简化公式计算测线间距
            d = 2 * h * tan(theta / 2) * (1 - n)  # 平坦海底的测线间距公式
            # 计算x方向的偏移量
            tx = d * get_gx(x, y)  # x偏移 = 间距 * x方向梯度
            # 计算y方向的偏移量
            ty = d * get_gy(x, y)  # y偏移 = 间距 * y方向梯度
            # 更新x坐标
            x = x + tx  # 新x坐标 = 原x坐标 + x偏移
            # 更新y坐标
            y = y + ty  # 新y坐标 = 原y坐标 + y偏移
        else:  # 如果坡度角较大（倾斜海底）
            # 计算几何参数A
            A = sin(90 - theta / 2 + alpha)  # 几何参数A
            # 计算几何参数B
            B = sin(90 - theta / 2 - alpha)  # 几何参数B
            # 计算迭代参数C
            C = sin(theta / 2) / A - 1 / sin(alpha)  # 迭代公式中的参数C
            # 计算迭代参数D（注意换行符的处理）
            D = (
                n * sin(theta / 2) * (1 / A + 1 / B)
                - sin(theta / 2) / B
                - 1 / sin(alpha)
            )  # 迭代公式中的参数D

            # 计算下一个测线位置的深度
            next_h = h * C / D  # 根据迭代公式计算新深度
            # 计算x方向的偏移量
            tx = (h - next_h) / tan(alpha) * get_gx(x, y)  # x偏移 = 深度差 * x方向梯度
            # 计算y方向的偏移量
            ty = (h - next_h) / tan(alpha) * get_gy(x, y)  # y偏移 = 深度差 * y方向梯度
            # 更新x坐标（注意方向相反）
            x = x + tx  # 新x坐标 = 原x坐标 - x偏移
            # 更新y坐标（注意方向相反）
            y = y + ty  # 新y坐标 = 原y坐标 - y偏移
        # 检查新位置是否在有效测量范围内（第三条测线的边界条件）
        if (
            x > 4 * 1852
            or y > 5 * 1852
            or y < 0
            or x < 0
            or 2500 * y - 9260 * x > -37040000
        ):  # 复杂边界检查（包含特殊线性约束）
            pass
        else:  # 如果在有效范围内
            # 将新位置添加到下一轮测线点列表
            child_point = [x, y]
            t1.append(child_point)  # 记录有效的测线点
            t1_overlap_rates.append(
                compute_parent_child_overlap_rate(i, child_point, theta=theta)
            )
    add_overlap_excess_length(metrics_context, 2, t1, t1_overlap_rates, threshold=0.2)
    # 绘制当前垂直测线
    plotted_line = line[:-1]
    plt.plot(plotted_line[:, 0], plotted_line[:, 1], color="darkgray")  # 用深灰色绘制垂直测线
    # 记录测线长度统计
    register_rendered_line(metrics_context, plotted_line, 2)
    dot = np.concatenate((dot, line), axis=0)  # 合并所有测量点
    # 深拷贝下一轮测线点
    line = copy.deepcopy(t1)  # 深拷贝避免引用问题
    # 将测线点列表转换为numpy数组
    line = np.array(line)  # 转换为numpy数组便于处理
    # 如果没有有效的下一轮测线点，退出循环
    if len(t1) == 0:  # 检查是否还有测线点
        break  # 退出垂直测线生成循环

# 第四条测线：起始位置(6500, 9000)
# 重新初始化测线路径列表
line = []  # 清空路径列表，准备生成第四条测线
# 设置第四条测线的起始x坐标
loc_x = 6500  # 起始x坐标为6500米
# 设置第四条测线的起始y坐标
loc_y = 9000  # 起始y坐标为9000米
# 设置前进步长
step = 50  # 每步前进50米
# 设置换能器开角
theta = 120  # 多波束换能器开角120度
# 生成第四条主测线路径
while True:  # 无限循环直到满足退出条件
    # 获取当前位置的x方向梯度
    gx = get_gx(loc_x, loc_y)  # 调用函数获取x方向梯度
    # 获取当前位置的y方向梯度
    gy = get_gy(loc_x, loc_y)  # 调用函数获取y方向梯度
    # 计算前进方向向量
    dx, dy = forward_direction(gx, gy)  # 获取垂直于梯度的前进方向
    # 更新当前位置x坐标
    loc_x += step * dx  # 新x坐标 = 原x坐标 + 步长 * x方向分量
    # 更新当前位置y坐标
    loc_y += step * dy  # 新y坐标 = 原y坐标 + 步长 * y方向分量
    # 检查是否超出测量区域边界
    if (
        loc_x > 4 * 1852 or loc_y > 5 * 1852 or loc_y < 0 or loc_x < 0
    ):  # 边界检查：4海里×5海里区域
        break  # 超出边界则退出循环
    # 将当前位置添加到测线路径
    line.append([loc_x, loc_y])  # 记录路径点坐标
# 将路径列表转换为numpy数组
line = np.array(line)  # 转换为numpy数组便于后续处理
# 绘制第四条主测线
plt.plot(line[:-1, 0], line[:-1, 1], color="c")  # 用青色绘制第四条主测线路径
# 为第四条主测线生成垂直测线
while True:  # 无限循环直到所有垂直测线生成完毕
    # 初始化下一轮测线点列表
    t1 = []  # 存储下一轮垂直测线的点坐标
    t1_overlap_rates = []
    # 遍历当前测线上的每个点
    for index, i in enumerate(line):  # 对测线上每个点进行处理
        # 获取当前点的x坐标
        x = i[0]  # 提取x坐标
        # 获取当前点的y坐标
        y = i[1]  # 提取y坐标
        # 计算当前位置的坡度角
        alpha = get_alpha(x, y)  # 获取海底坡度角
        # 获取当前位置的深度
        h = get_height(x, y)  # 获取海底深度
        # 根据坡度角大小选择不同的计算方法
        if alpha <= 0.005:  # 如果坡度角很小（近似平坦海底）
            # 使用简化公式计算测线间距
            d = 2 * h * tan(theta / 2) * (1 - n)  # 平坦海底的测线间距公式
            # 计算x方向的偏移量
            tx = d * get_gx(x, y)  # x偏移 = 间距 * x方向梯度
            # 计算y方向的偏移量
            ty = d * get_gy(x, y)  # y偏移 = 间距 * y方向梯度
            # 更新x坐标
            x = x + tx  # 新x坐标 = 原x坐标 + x偏移
            # 更新y坐标
            y = y + ty  # 新y坐标 = 原y坐标 + y偏移
        else:  # 如果坡度角较大（倾斜海底）
            # 计算几何参数A
            A = sin(90 - theta / 2 + alpha)  # 几何参数A
            # 计算几何参数B
            B = sin(90 - theta / 2 - alpha)  # 几何参数B
            # 计算迭代参数C
            C = sin(theta / 2) / A - 1 / sin(alpha)  # 迭代公式中的参数C
            # 计算迭代参数D（注意换行符的处理）
            D = (
                n * sin(theta / 2) * (1 / A + 1 / B)
                - sin(theta / 2) / B
                - 1 / sin(alpha)
            )  # 迭代公式中的参数D

            # 计算下一个测线位置的深度
            next_h = h * C / D  # 根据迭代公式计算新深度
            # 计算x方向的偏移量
            tx = (h - next_h) / tan(alpha) * get_gx(x, y)  # x偏移 = 深度差 * x方向梯度
            # 计算y方向的偏移量
            ty = (h - next_h) / tan(alpha) * get_gy(x, y)  # y偏移 = 深度差 * y方向梯度
            # 更新x坐标（注意方向相反）
            x = x + tx  # 新x坐标 = 原x坐标 - x偏移
            # 更新y坐标（注意方向相反）
            y = y + ty  # 新y坐标 = 原y坐标 - y偏移
        # 检查新位置是否在有效测量范围内（第四条测线的边界条件）
        if (
            x > 4 * 1852
            or y > 5 * 1852
            or y < 0
            or x < 0
            or 4408 * y + 2260 * x < 47598080
            or get_height(x, y) > 75
        ):  # 复杂边界检查（包含特殊线性约束和深度限制）
            pass
        else:  # 如果在有效范围内
            # 将新位置添加到下一轮测线点列表
            child_point = [x, y]
            t1.append(child_point)  # 记录有效的测线点
            t1_overlap_rates.append(
                compute_parent_child_overlap_rate(i, child_point, theta=theta)
            )
    add_overlap_excess_length(metrics_context, 3, t1, t1_overlap_rates, threshold=0.2)
    # 如果当前测线长度足够，进行统计和绘制
    if len(line) > 20:  # 测线点数大于20时才进行统计
        plotted_line = line
        # 绘制当前垂直测线
        plt.plot(plotted_line[:, 0], plotted_line[:, 1], color="darkgrey")  # 用银色绘制垂直测线
        # 记录测线长度统计
        register_rendered_line(metrics_context, plotted_line, 3)
    dot = np.concatenate((dot, line), axis=0)  # 合并所有测量点
    # 深拷贝下一轮测线点
    line = copy.deepcopy(t1)  # 深拷贝避免引用问题
    # 将测线点列表转换为numpy数组
    line = np.array(line)  # 转换为numpy数组便于处理
    # 如果没有有效的下一轮测线点，退出循环
    if len(t1) == 0:  # 检查是否还有测线点
        break  # 退出垂直测线生成循环

# 第五条测线：起始位置(5700, 7900)
# 重新初始化测线路径列表
line = []  # 清空路径列表，准备生成第五条测线
# 设置第五条测线的起始x坐标
loc_x = 5700  # 起始x坐标为5700米
# 设置第五条测线的起始y坐标
loc_y = 7900  # 起始y坐标为7900米
# 设置前进步长
step = 50  # 每步前进50米
# 设置换能器开角
theta = 120  # 多波束换能器开角120度
# 生成第五条主测线路径
while True:  # 无限循环直到满足退出条件
    # 获取当前位置的x方向梯度
    gx = get_gx(loc_x, loc_y)  # 调用函数获取x方向梯度
    # 获取当前位置的y方向梯度
    gy = get_gy(loc_x, loc_y)  # 调用函数获取y方向梯度
    # 计算前进方向向量
    dx, dy = forward_direction(gx, gy)  # 获取垂直于梯度的前进方向
    # 更新当前位置x坐标
    loc_x += step * dx  # 新x坐标 = 原x坐标 + 步长 * x方向分量
    # 更新当前位置y坐标
    loc_y += step * dy  # 新y坐标 = 原y坐标 + 步长 * y方向分量
    # 检查是否超出测量区域边界
    if (
        loc_x > 4 * 1852 or loc_y > 5 * 1852 or loc_y < 0 or loc_x < 0
    ):  # 边界检查：4海里×5海里区域
        break  # 超出边界则退出循环
    # 将当前位置添加到测线路径
    line.append([loc_x, loc_y])  # 记录路径点坐标
# 将路径列表转换为numpy数组
line = np.array(line)  # 转换为numpy数组便于后续处理
# 绘制第五条主测线
plt.plot(line[:-1, 0], line[:-1, 1], color="m")  # 用洋红色绘制第五条主测线路径
# 为第五条主测线生成垂直测线
while True:  # 无限循环直到所有垂直测线生成完毕
    # 初始化下一轮测线点列表
    t1 = []  # 存储下一轮垂直测线的点坐标
    t1_overlap_rates = []
    # 遍历当前测线上的每个点
    for index, i in enumerate(line):  # 对测线上每个点进行处理
        # 获取当前点的x坐标
        x = i[0]  # 提取x坐标
        # 获取当前点的y坐标
        y = i[1]  # 提取y坐标
        # 计算当前位置的坡度角
        alpha = get_alpha(x, y)  # 获取海底坡度角
        # 获取当前位置的深度
        h = get_height(x, y)  # 获取海底深度
        # 根据坡度角大小选择不同的计算方法
        if alpha <= 0.005:  # 如果坡度角很小（近似平坦海底）
            # 使用简化公式计算测线间距
            d = 2 * h * tan(theta / 2) * (1 - n)  # 平坦海底的测线间距公式
            # 计算x方向的偏移量
            tx = d * get_gx(x, y)  # x偏移 = 间距 * x方向梯度
            # 计算y方向的偏移量
            ty = d * get_gy(x, y)  # y偏移 = 间距 * y方向梯度
            # 更新x坐标
            x = x + tx  # 新x坐标 = 原x坐标 + x偏移
            # 更新y坐标
            y = y + ty  # 新y坐标 = 原y坐标 + y偏移
        else:  # 如果坡度角较大（倾斜海底）
            # 计算几何参数A
            A = sin(90 - theta / 2 + alpha)  # 几何参数A
            # 计算几何参数B
            B = sin(90 - theta / 2 - alpha)  # 几何参数B
            # 计算迭代参数C
            C = sin(theta / 2) / A - 1 / sin(alpha)  # 迭代公式中的参数C
            # 计算迭代参数D（注意换行符的处理）
            D = (
                n * sin(theta / 2) * (1 / A + 1 / B)
                - sin(theta / 2) / B
                - 1 / sin(alpha)
            )  # 迭代公式中的参数D

            # 计算下一个测线位置的深度
            next_h = h * C / D  # 根据迭代公式计算新深度
            # 计算x方向的偏移量
            tx = (h - next_h) / tan(alpha) * get_gx(x, y)  # x偏移 = 深度差 * x方向梯度
            # 计算y方向的偏移量
            ty = (h - next_h) / tan(alpha) * get_gy(x, y)  # y偏移 = 深度差 * y方向梯度
            # 更新x坐标
            x = x + tx  # 新x坐标 = 原x坐标 + x偏移
            # 更新y坐标
            y = y + ty  # 新y坐标 = 原y坐标 + y偏移
        # 检查新位置是否在有效测量范围内（第五条测线的边界条件）
        if (
            x > 4 * 1852 or y > 1852 * 5 or x < 0
        ):  # 简化的边界检查（只检查x和y的基本范围）
            pass
        else:  # 如果在有效范围内
            # 将新位置添加到下一轮测线点列表
            child_point = [x, y]
            t1.append(child_point)  # 记录有效的测线点
            t1_overlap_rates.append(
                compute_parent_child_overlap_rate(i, child_point, theta=theta)
            )
    add_overlap_excess_length(metrics_context, 4, t1, t1_overlap_rates, threshold=0.2)
    # 如果当前测线长度足够，进行统计和绘制
    if len(line) > 50:  # 测线点数大于50时才进行统计
        plotted_line = line[:-1]
        # 绘制当前垂直测线
        plt.plot(plotted_line[:, 0], plotted_line[:, 1], color="gray")  # 用灰色绘制垂直测线
        # 记录测线长度统计
        register_rendered_line(metrics_context, plotted_line, 4)
    dot = np.concatenate((dot, line), axis=0)  # 合并所有测量点
    # 深拷贝下一轮测线点
    line = copy.deepcopy(t1)  # 深拷贝避免引用问题
    # 将测线点列表转换为numpy数组
    line = np.array(line)  # 转换为numpy数组便于处理
    # 如果没有有效的下一轮测线点，退出循环
    if len(t1) == 0:  # 检查是否还有测线点
        break  # 退出垂直测线生成循环

# 保存最终的测线分布图
result_path = OUTPUT_DIR / "sin-0.png"
plt.savefig(result_path, dpi=300, bbox_inches="tight")  # 保存包含所有测线的完整图表
print(f"结果图已保存到: {result_path}")

plt.close()  # 默认关闭图像；如需交互展示可另行开启

metrics_path, fine_grid_view_path = save_metrics_report(OUTPUT_DIR, metrics_context)
print(f"统一统计指标已保存到: {metrics_path}")
print(f"细网格状态图已保存到: {fine_grid_view_path}")

# 设置测量点数据输出文件路径
path = OUTPUT_DIR / "dot-sin-0.xlsx"
# 将所有测量点坐标保存为Excel文件
pd.DataFrame(dot).to_excel(path, index=False)  # 保存所有测量点坐标，不包含行索引
print(f"测量点坐标已保存到: {path}")
