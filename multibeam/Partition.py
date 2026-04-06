import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def find_optimal_k_elbow(features, k_min=2, k_max=10):
    """
    根据 Elbow 准则自动寻找最优的簇数目 U
    使用"点到直线最大距离法"从数学上确定拐点
    """
    n_samples = len(features)
    # KMeans 要求 n_clusters < n_samples
    k_max = min(k_max, n_samples - 1)
    if k_max < k_min:
        print(f"  [警告] 数据点过少 (n={n_samples})，直接返回 k_min={k_min}")
        return k_min

    inertias = []
    k_range = range(k_min, k_max + 1)

    print("正在计算不同簇数下的 Inertia (簇内误差平方和)...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features)
        inertias.append(kmeans.inertia_)

    # 自动寻找拐点 (Elbow)
    # 建立一条连接起点 (k_min, inertia_min) 和终点 (k_max, inertia_max) 的直线
    p1 = np.array([k_range[0], inertias[0]])
    p2 = np.array([k_range[-1], inertias[-1]])

    distances = []
    for i, k in enumerate(k_range):
        p0 = np.array([k, inertias[i]])
        # 计算该点到直线的垂直距离
        distance = _point_line_distance(p0, p1, p2)
        distances.append(distance)

    # 距离直线最远的点即为最优拐点
    optimal_index = np.argmax(distances)
    optimal_u = k_range[optimal_index]

    # 绘制 Elbow 曲线图供验证
    plt.figure(figsize=(8, 4))
    plt.plot(k_range, inertias, marker="o", linestyle="-")
    plt.plot(
        k_range[optimal_index],
        inertias[optimal_index],
        marker="X",
        color="red",
        markersize=12,
        label=f"Optimal U = {optimal_u}",
    )
    plt.title("Elbow Method For Optimal U")
    plt.xlabel("Number of clusters (U)")
    plt.ylabel("Inertia (WCSS)")
    plt.legend()
    plt.grid(True)
    output_dir = Path("./multibeam/output/elbow")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "elbow_method.png", dpi=300, bbox_inches="tight")
    plt.close()
    return optimal_u


def partition_coverage_matrix(xs, ys, coverage_matrix, U=None, k_max=10):
    """
    对覆盖次数矩阵进行 K-means 空间聚类并可视化边界

    参数:
        xs, ys: 坐标数组
        coverage_matrix: 覆盖次数矩阵
        U: 指定分区数量（可选）。若传入则直接使用，若为 None 则使用 Elbow 法则自动确定
        k_max: Elbow 法则搜索的最大簇数（仅在 U=None 时生效）
    """
    rows, cols = coverage_matrix.shape
    n_samples = rows * cols

    # 1. 构建特征矩阵 (X坐标, Y坐标, 覆盖次数)
    # 注意：这里引入空间坐标是为了保证聚类结果在地理空间上的连续性
    grid_x, grid_y = np.meshgrid(xs, ys)

    # 将二维矩阵展平为一维特征向量
    feature_x = grid_x.flatten().reshape(-1, 1)
    feature_y = grid_y.flatten().reshape(-1, 1)
    feature_c = coverage_matrix.flatten().reshape(-1, 1)

    # 将空间坐标与覆盖次数拼接 (N, 3)
    features = np.hstack((feature_x, feature_y, feature_c))

    # 2. 特征缩放（修复：在标准化前计算权重，避免权重失效）
    # 先计算原始特征的尺度差异，用于后续权重调节
    std_xy_raw = np.std(features[:, :2])
    std_c_raw = np.std(features[:, 2])

    # 计算覆盖次数相对于空间坐标的权重比
    # 目的：使覆盖次数维度的尺度与空间坐标维度相当
    w_cover = std_xy_raw / (std_c_raw + 1e-6)

    # 应用权重：放大覆盖次数维度
    weighted_features = features.copy().astype(float)
    weighted_features[:, 2] *= w_cover

    # 再进行标准化，使各维度均值为0、方差为1
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(weighted_features)

    # 3. 确定簇的数目 U
    if U is not None:
        # 验证 U 的合法性
        if U < 2:
            raise ValueError(f"分区数 U 必须 >= 2，当前传入 U={U}")
        if U > n_samples - 1:
            raise ValueError(
                f"分区数 U={U} 超过数据点数 (n={n_samples})，"
                f"KMeans 要求 n_clusters < n_samples"
            )
        optimal_u = U
        print(f"[分区] 使用指定分区数量 U = {optimal_u}")
    else:
        # 使用 Elbow 法则自动确定
        optimal_u = find_optimal_k_elbow(scaled_features, k_min=2, k_max=k_max)
        print(f"[分区] 根据 Elbow 准则自动确定 U = {optimal_u} ")

    # 4. 对覆盖次数矩阵进行 K-means 空间聚类
    print(f"正在执行 K-means 聚类 (U={optimal_u})...")
    final_kmeans = KMeans(n_clusters=optimal_u, random_state=42, n_init=10)
    cluster_labels_1d = final_kmeans.fit_predict(scaled_features)

    # 5. 将一维的聚类标签还原为二维矩阵
    cluster_matrix = cluster_labels_1d.reshape(rows, cols)

    # 6. 可视化分区结果并确定每个区域的边界线
    plt.figure(figsize=(10, 8))

    # 绘制底图分区颜色
    plt.imshow(
        cluster_matrix,
        extent=[xs.min(), xs.max(), ys.max(), ys.min()],
        cmap="Set3",
        aspect="auto",
    )

    # 修复：轮廓线级别只需在簇之间绘制，避免冗余
    levels = np.arange(0.5, optimal_u - 0.5)
    if len(levels) > 0:  # U >= 3 时才需要边界线
        plt.contour(
            grid_x,
            grid_y,
            cluster_matrix,
            levels=levels,
            colors="black",
            linewidths=1.5,
            linestyles="dashed",
        )

    plt.title(f"K-means Spatial Partitioning (U={optimal_u}) with Boundary Lines")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")

    # 防止坐标轴倒置（根据海图习惯，通常北向上，如果需要反转去掉此行即可）
    plt.gca().invert_yaxis()
    plt.colorbar(label="Cluster ID")
    output_dir = Path("./multibeam/output/partition")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_dir / f"partition_U={optimal_u}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return cluster_matrix, optimal_u


def _point_line_distance(p0, p1, p2):
    return np.abs(
        (p2[1] - p1[1]) * p0[0]
        - (p2[0] - p1[0]) * p0[1]
        + p2[0] * p1[1]
        - p2[1] * p1[0]
    ) / np.linalg.norm(p2 - p1)


def is_point_in_partition(
    x,
    y,
    target_partition_id,
    xs,
    ys,
    cluster_matrix,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
):
    """
    判断点 (x, y) 是否属于指定的分区

    参数:
        x: 点的x坐标 (米)
        y: 点的y坐标 (米)
        target_partition_id: 目标分区ID (0 到 U-1)
        xs: x坐标数组 (一维)
        ys: y坐标数组 (一维)
        cluster_matrix: 分区矩阵 (二维，维度为 len(ys) x len(xs))
        x_min, x_max, y_min, y_max: 真实海域边界（arange 策略下 xs[-1] 可能超出真实边界）

    返回:
        tuple: (bool, int)
            - bool: True表示点属于目标分区，False表示不属于或点在网格范围外
            - int: 点实际所属的分区ID，如果在范围外则返回-1
    """
    actual_partition_id = get_partition_for_point(
        x,
        y,
        xs,
        ys,
        cluster_matrix,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )

    is_in_target = actual_partition_id == target_partition_id

    return is_in_target, actual_partition_id


def get_partition_for_point(
    x,
    y,
    xs,
    ys,
    cluster_matrix,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
):
    """
    获取点 (x, y) 所属的分区ID

    参数:
        x: 点的x坐标 (米)
        y: 点的y坐标 (米)
        xs: x坐标数组 (一维)
        ys: y坐标数组 (一维)
        cluster_matrix: 分区矩阵
        x_min, x_max, y_min, y_max: 真实海域边界（arange 策略下 xs[-1] 可能超出真实边界）

    返回:
        int: 分区ID (0 到 U-1)，如果点在网格范围外则返回 -1
    """
    # 使用真实海域边界（arange 策略下 xs[-1] 可能超出真实边界）
    real_x_min = x_min if x_min is not None else xs.min()
    real_x_max = x_max if x_max is not None else xs.max()
    real_y_min = y_min if y_min is not None else ys.min()
    real_y_max = y_max if y_max is not None else ys.max()

    # 检查点是否在真实海域范围内
    if x < real_x_min or x > real_x_max or y < real_y_min or y > real_y_max:
        return -1

    # 找到最近的网格索引
    x_idx = np.argmin(np.abs(xs - x))
    y_idx = np.argmin(np.abs(ys - y))

    return cluster_matrix[y_idx, x_idx]


def get_points_in_partition(target_partition_id, xs, ys, cluster_matrix):
    """
    获取指定分区内的所有网格点坐标

    参数:
        target_partition_id: 目标分区ID
        xs: x坐标数组 (一维)
        ys: y坐标数组 (一维)
        cluster_matrix: 分区矩阵

    返回:
        tuple: (points_x, points_y) 两个数组，包含该分区内所有点的坐标
    """
    # 找到属于目标分区的所有网格索引
    rows, cols = np.where(cluster_matrix == target_partition_id)

    # 将索引转换为实际坐标
    points_x = xs[cols]
    points_y = ys[rows]

    return points_x, points_y
