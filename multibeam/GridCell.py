from pathlib import Path
import numpy as np

from tool import Data
from tool import Tool

NAUTICAL_MILE_TO_METER = 1852.0


def infer_raw_grid_spacing_m(x, y):
    """根据原始 Excel 坐标推断原始网格边长（米）。"""

    x_unique = np.sort(np.unique(np.asarray(x, dtype=float)))
    y_unique = np.sort(np.unique(np.asarray(y, dtype=float)))
    x_diff = np.diff(x_unique)
    y_diff = np.diff(y_unique)
    x_diff = x_diff[x_diff > 1e-12]
    y_diff = y_diff[y_diff > 1e-12]

    if len(x_diff) == 0 and len(y_diff) == 0:
        raise ValueError("无法从原始坐标推断网格边长。")

    candidate_steps = []
    if len(x_diff) > 0:
        candidate_steps.append(float(np.min(x_diff)))
    if len(y_diff) > 0:
        candidate_steps.append(float(np.min(y_diff)))

    raw_spacing_nm = min(candidate_steps)
    raw_spacing_m = raw_spacing_nm * NAUTICAL_MILE_TO_METER
    return raw_spacing_nm, raw_spacing_m


def compute_axis_effective_lengths(centers, cell_size, lower, upper):
    """计算一维均匀网格在矩形边界内的有效长度。"""
    half = cell_size / 2
    left = centers - half
    right = centers + half
    return np.maximum(0.0, np.minimum(right, upper) - np.maximum(left, lower))


def compute_effective_cell_area_matrix(xs, ys, d, x_min, x_max, y_min, y_max):
    """计算二维均匀网格在矩形海域内的有效面积矩阵。"""
    x_lengths = compute_axis_effective_lengths(
        np.asarray(xs, dtype=float), d, x_min, x_max
    )
    y_lengths = compute_axis_effective_lengths(
        np.asarray(ys, dtype=float), d, y_min, y_max
    )
    cell_effective_area = np.outer(y_lengths, x_lengths)
    cell_area_ratio = cell_effective_area / (d**2)
    return x_lengths, y_lengths, cell_effective_area, cell_area_ratio


def calculate_mesh_size_search_trace(
    data_path, min_error=0.001, search_step=0.5, stop_on_first_match=False
):
    """计算最优 d 搜索过程的完整误差轨迹。"""

    x, y, z = Tool.read_grid(data_path)
    raw_spacing_nm, raw_spacing_m = infer_raw_grid_spacing_m(x, y)

    max_depth = np.max(z)
    theta = 120
    w_max = max_depth * Data.tan(theta / 2) * 2
    xi = w_max / 2
    A = np.pi * xi**2

    relative_error_threshold = min_error
    patent_error_threshold = (1 - relative_error_threshold) * (xi**2)

    xi_start_d = float(xi)
    start_d = float(max(raw_spacing_m - search_step, 0.0))
    d_candidates = np.arange(start_d, 0.0, -search_step)

    def build_record(d, *, is_raw_spacing_reference, is_search_candidate):
        max_idx = int(xi / d)
        m = np.arange(-max_idx, max_idx + 1)
        n = np.arange(-max_idx, max_idx + 1)
        M, N = np.meshgrid(m, n)
        N_points = int(np.sum(M**2 + N**2 <= (xi / d) ** 2))
        B = N_points * (d**2)
        relative_error = abs(A - B) / A
        patent_error = (1 - relative_error) * (xi**2)
        meets_threshold = patent_error >= patent_error_threshold
        return {
            "d_m": float(d),
            "relative_error": float(relative_error),
            "relative_error_percent": float(relative_error * 100),
            "patent_error": float(patent_error),
            "meets_threshold": bool(meets_threshold),
            "grid_points_in_circle": N_points,
            "covered_area_B": float(B),
            "theoretical_area_A": float(A),
            "is_raw_spacing_reference": bool(is_raw_spacing_reference),
            "is_search_candidate": bool(is_search_candidate),
            "radius_grid_count": int(max_idx),
        }

    trace_records = [
        build_record(
            raw_spacing_m,
            is_raw_spacing_reference=True,
            is_search_candidate=False,
        )
    ]

    optimal_d = None
    final_relative_error = None
    final_patent_E = None

    for d in d_candidates:
        record = build_record(
            d,
            is_raw_spacing_reference=False,
            is_search_candidate=True,
        )
        trace_records.append(record)
        if optimal_d is None and record["meets_threshold"]:
            optimal_d = float(d)
            final_relative_error = record["relative_error"]
            final_patent_E = record["patent_error"]
            if stop_on_first_match:
                break

    return {
        "max_depth": float(max_depth),
        "theta": float(theta),
        "w_max": float(w_max),
        "xi": float(xi),
        "theoretical_area_A": float(A),
        "raw_spacing_nm": float(raw_spacing_nm),
        "raw_spacing_m": float(raw_spacing_m),
        "xi_start_d": float(xi_start_d),
        "start_d": float(start_d),
        "search_step": float(search_step),
        "relative_error_threshold": float(relative_error_threshold),
        "patent_error_threshold": float(patent_error_threshold),
        "optimal_d": optimal_d,
        "final_relative_error": final_relative_error,
        "final_patent_E": final_patent_E,
        "trace_records": trace_records,
    }


def calculate_optimal_mesh_size(data_path, min_error=0.001):
    trace_result = calculate_mesh_size_search_trace(
        data_path, min_error=min_error, stop_on_first_match=True
    )
    max_depth = trace_result["max_depth"]
    print("最大深度:", max_depth)

    xi = trace_result["xi"]
    A = trace_result["theoretical_area_A"]

    print(f"xi: {xi:.2f}, A: {A:.2f}")
    relative_error_threshold = trace_result["relative_error_threshold"]
    patent_error_threshold = trace_result["patent_error_threshold"]
    optimal_d = trace_result["optimal_d"]
    final_relative_error = trace_result["final_relative_error"]
    final_patent_E = trace_result["final_patent_E"]
    w_max = trace_result["w_max"]
    raw_spacing_nm = trace_result["raw_spacing_nm"]
    raw_spacing_m = trace_result["raw_spacing_m"]
    xi_start_d = trace_result["xi_start_d"]
    start_d = trace_result["start_d"]

    # 输出结果
    print(f"最大有效测深宽度 (w_max): {w_max:.2f}")
    print(f"邻域覆盖半径 (ξ): {xi:.2f}")
    print(f"原始网格边长: {raw_spacing_nm:.4f} 海里 = {raw_spacing_m:.2f} m")
    print(f"邻域覆盖半径(ξ)对应的理论起点 d: {xi_start_d:.2f} m")
    print(f"受原始网格上界约束的搜索起始 d: {start_d:.2f} m")
    print(f"设定相对面积误差阈值: {relative_error_threshold * 100}%")
    print(f"对应专利误差项下限: {patent_error_threshold:.4f}")

    if optimal_d:
        print(f"满足精度的最优网格边长 (d): {optimal_d:.2f}")
        print(f"最终专利误差项 E(d, ξ): {final_patent_E:.4f}")
        print(f"最终相对面积误差: {final_relative_error * 100:.4f}%")
    else:
        print("在指定范围内未找到满足精度的 d，请尝试减小搜索下限或增大误差阈值。")
    return optimal_d


def generate_coordinate_array(x_min, x_max, y_min, y_max, d):
    """
    使用固定步长 d 生成覆盖整个海域的坐标数组。

    采用 arange 策略：以 d 为步长从起点生成坐标，末端自然延伸
    确保覆盖整个海域。超出海域边界的网格在后续计算中通过掩码过滤。

    优势:
        - 网格间距严格等于 d，几何精度不受损
        - 天然覆盖整个海域，无需妥协 d 的值（不受质数问题影响）
        - 末端超出的网格通过掩码自动过滤，不影响面积计算

    参数:
        x_min, x_max: X 方向海域边界
        y_min, y_max: Y 方向海域边界
        d: 最优网格边长（直接使用，不做妥协调整）

    返回:
        xs: X 坐标数组 (一维, 间距严格等于 d)
        ys: Y 坐标数组 (一维, 间距严格等于 d)
        boundary_mask: 二维布尔掩码 (rows x cols), True 表示网格与海域有重叠面积
        cell_effective_area: 二维有效面积矩阵
        cell_area_ratio: 二维面积比例矩阵，相对于满格 d² 的比例
    """
    # arange 右端 +d 确保末端网格中心能覆盖到边界
    xs = np.arange(x_min, x_max + d, d)
    ys = np.arange(y_min, y_max + d, d)

    (
        x_effective_lengths,
        y_effective_lengths,
        cell_effective_area,
        cell_area_ratio,
    ) = compute_effective_cell_area_matrix(xs, ys, d, x_min, x_max, y_min, y_max)

    # 边界掩码：只要该格与矩形海域有正面积重叠，就视为有效格
    boundary_mask = cell_effective_area > 0

    # 数值保护：边界上的极小浮点误差直接截断
    cell_effective_area = np.where(boundary_mask, cell_effective_area, 0.0)
    cell_area_ratio = np.where(boundary_mask, cell_area_ratio, 0.0)

    effective_total_area = float(np.sum(cell_effective_area))
    target_total_area = float((x_max - x_min) * (y_max - y_min))
    area_error = effective_total_area - target_total_area

    _ = x_effective_lengths, y_effective_lengths

    effective_cells = np.sum(boundary_mask)
    total_cells = len(xs) * len(ys)
    print(
        f"坐标数组: X方向{len(xs)}点, Y方向{len(ys)}点, "
        f"有效网格{effective_cells}/{total_cells} "
        f"(末端超出{total_cells - effective_cells}个网格被裁剪)"
    )
    print(
        f"边界修正后总面积: {effective_total_area:.2f} m² | "
        f"理论矩形面积: {target_total_area:.2f} m² | 偏差: {area_error:.6f}"
    )

    return xs, ys, boundary_mask, cell_effective_area, cell_area_ratio


if __name__ == "__main__":
    d_optimal = calculate_optimal_mesh_size(
        Path(__file__).parents[1] / "data" / "data.xlsx"
    )

    if d_optimal:
        # 测试新坐标生成方案
        X_MIN, X_MAX = 0, 4 * 1852
        Y_MIN, Y_MAX = 0, 5 * 1852
        xs, ys, mask, area_mat, area_ratio = generate_coordinate_array(
            X_MIN, X_MAX, Y_MIN, Y_MAX, d_optimal
        )
        print(f"xs 范围: [{xs[0]:.2f}, {xs[-1]:.2f}], 间距: {xs[1] - xs[0]:.2f}")
        print(f"ys 范围: [{ys[0]:.2f}, {ys[-1]:.2f}], 间距: {ys[1] - ys[0]:.2f}")
        print(
            f"有效面积总和: {np.sum(area_mat):.2f}, 平均面积比例: {np.mean(area_ratio):.4f}"
        )
