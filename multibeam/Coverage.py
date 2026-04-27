import numpy as np

from tool import Data


def generate_resampled_grid(x_min, x_max, y_min, y_max, d):
    """
    根据最优网格边长 d 和预测模型，重新生成指定范围的海域深度矩阵。
    使用 arange 策略：以 d 为固定步长生成坐标，末端自然延伸覆盖整个海域。

    优势：
        - 网格间距严格等于 d，几何精度不受损
        - 天然覆盖整个海域，不受质数/整除问题影响
        - 末端超出海域的网格通过 boundary_mask 过滤

    返回:
        xs: X 坐标数组 (间距严格等于 d)
        ys: Y 坐标数组 (间距严格等于 d)
        Z: 深度矩阵 (rows x cols)
        gx_matrix: x方向梯度矩阵 (rows x cols)
        gy_matrix: y方向梯度矩阵 (rows x cols)
        boundary_mask: 二维布尔掩码，True 表示网格与海域有正面积重叠
        cell_effective_area: 二维有效面积矩阵
        cell_area_ratio: 二维面积比例矩阵
    """
    from multibeam.GridCell import generate_coordinate_array

    xs, ys, boundary_mask, cell_effective_area, cell_area_ratio = (
        generate_coordinate_array(x_min, x_max, y_min, y_max, d)
    )

    rows = len(ys)
    cols = len(xs)

    print(f"正在使用模型生成重采样网格 (维度: {rows} x {cols}, 网格边长: {d:.2f}m)...")
    print(
        f"  X 范围: [{xs[0]:.2f}, {xs[-1]:.2f}] (实际海域: [{x_min:.2f}, {x_max:.2f}])"
    )
    print(
        f"  Y 范围: [{ys[0]:.2f}, {ys[-1]:.2f}] (实际海域: [{y_min:.2f}, {y_max:.2f}])"
    )

    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    print(f"正在批量预测重采样网格深度与梯度，共 {len(points)} 个点...")
    predictions = Data.predict_model_fields(
        points,
        include_height=True,
        include_gradient=True,
    )
    Z = predictions["height"].reshape(rows, cols)
    gx_matrix = predictions["gx"].reshape(rows, cols)
    gy_matrix = predictions["gy"].reshape(rows, cols)

    return (
        xs,
        ys,
        Z,
        gx_matrix,
        gy_matrix,
        boundary_mask,
        cell_effective_area,
        cell_area_ratio,
    )


def calculate_coverage_matrix_with_ml(x_min, x_max, y_min, y_max, d, theta=120):
    """
    结合机器学习深度预测模型和最优网格 d，计算覆盖次数矩阵。
    """
    # 1. 调用模型重采样，获取全新的 Z 矩阵和边界掩码
    xs, ys, Z, gx_matrix, gy_matrix, boundary_mask, cell_effective_area, cell_area_ratio = (
        generate_resampled_grid(x_min, x_max, y_min, y_max, d)
    )

    rows, cols = Z.shape
    times_mat = np.zeros((rows, cols), dtype=int)

    # 2. 获取全局最大水深，计算覆盖邻域边界
    # 仅使用海域内的有效深度
    valid_depths = Z[boundary_mask]
    if len(valid_depths) == 0:
        raise ValueError("边界掩码内无有效网格，请检查坐标范围与网格边长 d 的设置。")
    max_depth = np.max(valid_depths)
    # 根据现有测量标准考虑先验海图最大水深计算最大有效测深宽度 [cite: 38]
    w_max = max_depth * np.tan(np.radians(theta / 2)) * 2

    # 计算需要向外扩张多少个网格才能覆盖完 w_max/2 的半径
    r_cells = int(np.ceil((w_max / 2) / d))

    print(f"全局最大水深: {max_depth:.2f}m, 最大有效测深宽度: {w_max:.2f}m")
    print(f"动态邻域搜索半径: {r_cells} 个网格")

    print("正在进行全海域邻域覆盖判定并生成矩阵...")
    # 3. 遍历每一个网格进行邻域覆盖判定
    for i in range(rows):
        for j in range(cols):
            # 跳过海域范围外的网格
            if not boundary_mask[i, j]:
                continue

            D_center = Z[i, j]

            # 限定局部邻域边界 (防止搜出数组越界)
            min_row = max(0, i - r_cells)
            max_row = min(rows - 1, i + r_cells)
            min_col = max(0, j - r_cells)
            max_col = min(cols - 1, j + r_cells)

            # 截取邻域内的深度子矩阵
            Z_neighbor = Z[min_row : max_row + 1, min_col : max_col + 1]

            # 生成局部坐标网格 (相对于中心点)
            grid_x, grid_y = np.meshgrid(
                np.arange(min_col - j, max_col - j + 1) * d,
                np.arange(min_row - i, max_row - i + 1) * d,
            )

            # 4. 计算水平面投影距离 dis_xoy
            dis_xoy = np.sqrt(grid_x**2 + grid_y**2)

            # 剔除超过最大有效测深宽度一半的范围外网格，形成圆形邻域 [cite: 38]
            in_circle_mask = dis_xoy <= (w_max / 2)

            # 5. 计算三维空间距离 dis_xyz
            dis_xyz = np.sqrt(dis_xoy**2 + (D_center - Z_neighbor) ** 2)

            # 忽略中心点距离为0带来的除零警告
            with np.errstate(invalid="ignore", divide="ignore"):
                # 判定以 P 为中心的邻域覆盖区域内 Q 点所在网格是否被覆盖
                # 1. 换能器开角转弧度
                theta_rad = np.radians(theta)
                # 2. 计算带符号的真实坡度角 alpha_signed
                # 使用反正切 arctan(垂直高度差 / 水平距离)
                # 注意：前提是 Z 矩阵代表"深度"(越大越深)。若 Z 代表"海拔高度"(越小越深)，则需写成 (Z_neighbor - D_center)
                alpha_signed = np.arctan((D_center - Z_neighbor) / dis_xoy)

                # 3. 统一使用一套公式，符号会自动调节加减
                dis_sp = (np.sin(theta_rad / 2) * D_center) / np.sin(
                    np.pi / 2 - theta_rad / 2 + alpha_signed
                )

                # 4. 判定被覆盖 (同样需要判定 dis_sp > 0 防止极其陡峭的下坡导致分母为负)
                valid_mask = (dis_sp > 0) & (dis_xyz <= dis_sp) & in_circle_mask

            # 中心点(测量船自身)始终被覆盖，修复除零产生的 nan 掩码
            valid_mask[i - min_row, j - min_col] = True

            # 6. 计算每个网格的覆盖次数，累加到全局矩阵 [cite: 41]
            times_mat[min_row : max_row + 1, min_col : max_col + 1] += valid_mask

    print("覆盖次数矩阵计算完成！")
    return (
        xs,
        ys,
        Z,
        times_mat,
        boundary_mask,
        cell_effective_area,
        cell_area_ratio,
        gx_matrix,
        gy_matrix,
    )
