import numpy as np
import matplotlib.pyplot as plt
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
        distance = np.abs(np.cross(p2 - p1, p1 - p0)) / np.linalg.norm(p2 - p1)
        distances.append(distance)

    # 距离直线最远的点即为最优拐点
    optimal_index = np.argmax(distances)
    optimal_u = k_range[optimal_index]

    # 绘制 Elbow 曲线图供验证
    plt.figure(figsize=(8, 4))
    plt.plot(k_range, inertias, marker='o', linestyle='-')
    plt.plot(k_range[optimal_index], inertias[optimal_index], marker='X', color='red', markersize=12,
             label=f'Optimal U = {optimal_u}')
    plt.title('Elbow Method For Optimal U')
    plt.xlabel('Number of clusters (U)')
    plt.ylabel('Inertia (WCSS)')
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
    plt.imshow(cluster_matrix, extent=[xs.min(), xs.max(), ys.max(), ys.min()],
               cmap='Set3', aspect='auto')

    # 使用 contour 提取并绘制区域边界线
    plt.contour(grid_x, grid_y, cluster_matrix, levels=optimal_u - 1,
                colors='black', linewidths=1.5, linestyles='dashed')

    plt.title(f'K-means Spatial Partitioning (U={optimal_u}) with Boundary Lines')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')

    # 防止坐标轴倒置（根据海图习惯，通常北向上，如果需要反转去掉此行即可）
    plt.gca().invert_yaxis()
    plt.colorbar(label='Cluster ID')
    plt.show()

    return cluster_matrix, optimal_u