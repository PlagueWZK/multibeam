from pathlib import Path
import numpy as np

from tool import Data
from tool import Tool


def calculate_optimal_mesh_size(data_path, min_error=0.001):
    # 读取高程/深度数据
    x, y, z = Tool.read_grid(data_path)

    max_depth = np.max(z)
    print("最大深度:", max_depth)

    theta = 120

    # 计算最大有效测深宽度
    w_max = max_depth * Data.tan(theta / 2) * 2

    # 1. 计算邻域覆盖半径 ξ 和圆的理论面积 A
    xi = w_max / 2
    A = np.pi * xi**2

    print(f"xi: {xi:.2f}, A: {A:.2f}")
    # 2. 设定控制精度的目标误差阈值 (例如 1% 的相对误差)
    error_threshold = min_error

    # 3. 初始化寻优参数
    optimal_d = None
    # 设定 d 的搜索范围，从较大的网格逐渐细化到较小的网格
    # 步长可以根据实际精度需求调整，这里以 0.5 为步长递减
    d_candidates = np.arange(xi, 0.1, -0.5)

    final_E = None

    for d in d_candidates:
        # 计算在当前网格边长 d 下，X轴和Y轴方向的最大网格索引
        max_idx = int(xi / d)

        # 生成 m 和 n 的二维坐标网格
        m = np.arange(-max_idx, max_idx + 1)
        n = np.arange(-max_idx, max_idx + 1)
        M, N = np.meshgrid(m, n)

        # 统计落在圆内的格点数 N(d, ξ)
        # 判定条件：(m*d)^2 + (n*d)^2 <= xi^2，即 m^2 + n^2 <= (xi/d)^2
        N_points = np.sum(M**2 + N**2 <= (xi / d) ** 2)

        # 计算被判定为邻域覆盖的区域面积 B
        B = N_points * (d**2)

        # 计算相对误差 E(d, ξ)
        E = abs(A - B) / A

        # 根据专利中的误差项指数拟合逻辑，误差会随着 d 的减小趋近于0
        # 当误差首次小于设定的阈值时，认为找到了满足精度的最优且最经济的网格边长
        if E <= error_threshold:
            optimal_d = d
            final_E = E
            break

    # 输出结果
    print(f"最大有效测深宽度 (w_max): {w_max:.2f}")
    print(f"邻域覆盖半径 (ξ): {xi:.2f}")
    print(f"设定误差阈值: {error_threshold * 100}%")

    if optimal_d:
        print(f"满足精度的最优网格边长 (d): {optimal_d:.2f}")
        print(f"最终拟合误差 (E): {final_E * 100:.4f}%")
    else:
        print("在指定范围内未找到满足精度的 d，请尝试减小搜索下限或增大误差阈值。")
    return optimal_d


def adjust_mesh_size_to_fit_domain(d_optimal, x_range, y_range, tolerance=0.05):
    """
    找到最接近 d_optimal 且能整除海域尺寸的网格边长。

    通过调整分段数，使网格边长恰好整除 X 和 Y 方向的海域范围，
    保证 xs[-1] == x_max, ys[-1] == y_max，面积计算精确。

    参数:
        d_optimal: 原始最优网格边长（仅考虑几何精度）
        x_range: X 方向海域宽度 (x_max - x_min)
        y_range: Y 方向海域长度 (y_max - y_min)
        tolerance: 允许偏离 d_optimal 的最大比例（默认 5%）

    返回:
        d_aligned: 对齐后的网格边长
    """
    n_x_ideal = x_range / d_optimal
    n_y_ideal = y_range / d_optimal

    candidates = []
    for n_x in range(max(1, int(n_x_ideal) - 10), int(n_x_ideal) + 11):
        for n_y in range(max(1, int(n_y_ideal) - 10), int(n_y_ideal) + 11):
            # 取较小的 d 保证两个方向都能恰好覆盖
            d = min(x_range / n_x, y_range / n_y)
            deviation = abs(d - d_optimal) / d_optimal
            if deviation <= tolerance:
                candidates.append((deviation, d, n_x, n_y))

    if not candidates:
        # 放宽 tolerance 重试
        return adjust_mesh_size_to_fit_domain(
            d_optimal, x_range, y_range, tolerance + 0.05
        )

    # 选偏离 optimal_d 最小的候选
    candidates.sort()
    _, d_best, n_x, n_y = candidates[0]
    print(
        f"网格对齐: d_optimal={d_optimal:.2f} → d_aligned={d_best:.2f}m "
        f"(X分{n_x}段, Y分{n_y}段, 偏差{abs(d_best - d_optimal) / d_optimal:.2%})"
    )
    return d_best


if __name__ == "__main__":
    print(calculate_optimal_mesh_size(Path(__file__).parents[1] / "data" / "data.xlsx"))
