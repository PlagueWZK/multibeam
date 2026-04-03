import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def find_optimal_k_elbow(features, k_min=2, k_max=10):
    """
    根据 Elbow 准则自动寻找最优的簇数目 U
    使用“点到直线最大距离法”从数学上确定拐点
    """
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
    plt.show()

    return optimal_u


def partition_coverage_matrix(xs, ys, coverage_matrix, k_max=10):
    """
    对覆盖次数矩阵进行 K-means 空间聚类并可视化边界
    """
    rows, cols = coverage_matrix.shape

    # 1. 构建特征矩阵 (X坐标, Y坐标, 覆盖次数)
    # 注意：这里引入空间坐标是为了保证聚类结果在地理空间上的连续性
    grid_x, grid_y = np.meshgrid(xs, ys)

    # 将二维矩阵展平为一维特征向量
    feature_x = grid_x.flatten().reshape(-1, 1)
    feature_y = grid_y.flatten().reshape(-1, 1)
    feature_c = coverage_matrix.flatten().reshape(-1, 1)

    # 将空间坐标与覆盖次数拼接 (N, 3)
    features = np.hstack((feature_x, feature_y, feature_c))

    # 2. 特征标准化 (极其重要！)
    # 因为坐标值在几千 (米)，而覆盖次数只有几十，不缩放会导致 K-means 完全忽略覆盖次数
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # 自动计算权重（核心）
    std_xy = np.std(scaled_features[:, :2])
    std_c = np.std(scaled_features[:, 2])

    w_space = 1.0
    w_cover = std_xy / (std_c + 1e-6)

    # 应用权重
    scaled_features[:, :2] *= w_space
    scaled_features[:, 2] *= w_cover

    # 3. 根据 Elbow 准则确定簇的数目 U
    optimal_u = find_optimal_k_elbow(scaled_features, k_min=2, k_max=k_max)
    print(f"根据 Elbow 准则确定的最优簇数目 U = {optimal_u} ")

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

    levels = np.arange(optimal_u)
    plt.contour(
        grid_x,
        grid_y,
        cluster_matrix,
        levels=levels + 0.5,
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
    plt.show()

    return cluster_matrix, optimal_u


def _point_line_distance(p0, p1, p2):
    return np.abs(
        (p2[1] - p1[1]) * p0[0]
        - (p2[0] - p1[0]) * p0[1]
        + p2[0] * p1[1]
        - p2[1] * p1[0]
    ) / np.linalg.norm(p2 - p1)


def is_point_in_partition(x, y, target_partition_id, xs, ys, cluster_matrix):
    """
    判断点 (x, y) 是否属于指定的分区

    参数:
        x: 点的x坐标 (米)
        y: 点的y坐标 (米)
        target_partition_id: 目标分区ID (0 到 U-1)
        xs: x坐标数组 (一维)
        ys: y坐标数组 (一维)
        cluster_matrix: 分区矩阵 (二维，维度为 len(ys) x len(xs))

    返回:
        tuple: (bool, int)
            - bool: True表示点属于目标分区，False表示不属于或点在网格范围外
            - int: 点实际所属的分区ID，如果在范围外则返回-1

    示例:
        >>> is_in_partition, actual_id = is_point_in_partition(
        ...     x=1000, y=2000, target_partition_id=2,
        ...     xs=xs, ys=ys, cluster_matrix=final_cluster_matrix
        ... )
        >>> print(f"点是否在分区2中: {is_in_partition}, 实际分区: {actual_id}")
    """
    # 复用 get_partition_for_point 获取实际分区ID
    actual_partition_id = get_partition_for_point(x, y, xs, ys, cluster_matrix)

    # 判断是否属于目标分区
    is_in_target = actual_partition_id == target_partition_id

    return is_in_target, actual_partition_id


def get_partition_for_point(x, y, xs, ys, cluster_matrix):
    """
    获取点 (x, y) 所属的分区ID

    参数:
        x: 点的x坐标 (米)
        y: 点的y坐标 (米)
        xs: x坐标数组 (一维)
        ys: y坐标数组 (一维)
        cluster_matrix: 分区矩阵

    返回:
        int: 分区ID (0 到 U-1)，如果点在网格范围外则返回 -1
    """
    # 检查点是否在网格范围内
    if x < xs.min() or x > xs.max() or y < ys.min() or y > ys.max():
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
