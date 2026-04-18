import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from tool.Data import get_gx, get_gy


PRIMARY_FEATURES_BY_MODE = {
    "xy_coverage": ("X", "Y", "coverage_count"),
    "coverage_only": ("coverage_count",),
}
SECONDARY_FEATURE_NAMES = ("ux", "uy")


def find_optimal_k_elbow(features, k_min=2, k_max=10, output_dir=None):
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
    if output_dir is not None:
        elbow_dir = Path(output_dir) / "partition"
        elbow_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(elbow_dir / "elbow_method.png", dpi=300, bbox_inches="tight")
    plt.close()
    return optimal_u


def _build_feature_stacks(
    xs,
    ys,
    coverage_matrix,
    gx_matrix=None,
    gy_matrix=None,
    primary_feature_mode="xy_coverage",
):
    rows, cols = coverage_matrix.shape
    grid_x, grid_y = np.meshgrid(xs, ys)

    if primary_feature_mode not in PRIMARY_FEATURES_BY_MODE:
        raise ValueError(
            "不支持的一级分区特征模式："
            f"{primary_feature_mode}，可选={tuple(PRIMARY_FEATURES_BY_MODE)}"
        )

    if gx_matrix is not None or gy_matrix is not None:
        if gx_matrix is None or gy_matrix is None:
            raise ValueError("gx_matrix 和 gy_matrix 必须同时提供，或同时为 None。")
        gx_matrix = np.asarray(gx_matrix, dtype=float)
        gy_matrix = np.asarray(gy_matrix, dtype=float)
        if gx_matrix.shape != (rows, cols) or gy_matrix.shape != (rows, cols):
            raise ValueError(
                "提供的梯度矩阵尺寸与 coverage_matrix 不一致："
                f"coverage={coverage_matrix.shape}, gx={gx_matrix.shape}, gy={gy_matrix.shape}"
            )
        print("[分区] 复用 Coverage 阶段预计算的网格梯度矩阵...")
        gx_flat = gx_matrix.reshape(-1, 1)
        gy_flat = gy_matrix.reshape(-1, 1)
    else:
        print("[分区] 正在计算网格点梯度方向...")
        flat_xy = list(zip(grid_x.flatten(), grid_y.flatten()))
        gx_flat = np.array([get_gx(x, y) for x, y in flat_xy], dtype=float).reshape(
            -1, 1
        )
        gy_flat = np.array([get_gy(x, y) for x, y in flat_xy], dtype=float).reshape(
            -1, 1
        )

    feature_c = coverage_matrix.flatten().reshape(-1, 1)

    gx_matrix = gx_flat.reshape(rows, cols)
    gy_matrix = gy_flat.reshape(rows, cols)
    gradient_magnitude = np.hypot(gx_matrix, gy_matrix)
    direction_valid_mask = np.isfinite(gradient_magnitude) & (gradient_magnitude > 1e-8)
    unit_gx_matrix = np.zeros_like(gx_matrix, dtype=float)
    unit_gy_matrix = np.zeros_like(gy_matrix, dtype=float)
    unit_gx_matrix[direction_valid_mask] = (
        gx_matrix[direction_valid_mask] / gradient_magnitude[direction_valid_mask]
    )
    unit_gy_matrix[direction_valid_mask] = (
        gy_matrix[direction_valid_mask] / gradient_magnitude[direction_valid_mask]
    )

    if primary_feature_mode == "xy_coverage":
        feature_x = grid_x.flatten().reshape(-1, 1)
        feature_y = grid_y.flatten().reshape(-1, 1)
        primary_features = np.hstack((feature_x, feature_y, feature_c))
    else:
        primary_features = feature_c

    secondary_features = np.hstack(
        (
            unit_gx_matrix.reshape(-1, 1),
            unit_gy_matrix.reshape(-1, 1),
        )
    )

    return {
        "rows": rows,
        "cols": cols,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "gx_matrix": gx_matrix,
        "gy_matrix": gy_matrix,
        "gradient_magnitude": gradient_magnitude,
        "direction_valid_mask": direction_valid_mask,
        "unit_gx_matrix": unit_gx_matrix,
        "unit_gy_matrix": unit_gy_matrix,
        "primary_feature_names": PRIMARY_FEATURES_BY_MODE[primary_feature_mode],
        "primary_features": primary_features,
        "secondary_features": secondary_features,
    }


def _scale_features_for_kmeans(valid_features):
    weighted_features = valid_features.copy().astype(float)

    if valid_features.shape[1] >= 3:
        std_xy_raw = float(np.std(valid_features[:, :2]))
        weighted_features[:, 2] *= std_xy_raw / (
            float(np.std(valid_features[:, 2])) + 1e-6
        )

        if valid_features.shape[1] >= 5:
            weighted_features[:, 3:5] *= std_xy_raw / (
                float(np.std(valid_features[:, 3:5])) + 1e-6
            )

    scaler = StandardScaler()
    return scaler.fit_transform(weighted_features)


def _cluster_with_selected_mask(
    features,
    selected_mask_1d,
    rows,
    cols,
    U=None,
    k_max=10,
    output_dir=None,
    elbow_subdir=None,
):
    valid_features = features[selected_mask_1d]
    n_samples = len(valid_features)
    if n_samples == 0:
        raise ValueError("无有效海域网格可用于聚类，请检查有效掩码。")
    if n_samples < 3:
        raise ValueError(f"可用于聚类的数据点过少 (n={n_samples})，至少需要 3 个点。")

    scaled_features = _scale_features_for_kmeans(valid_features)

    if U is not None:
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
        elbow_output_dir = None
        if output_dir is not None:
            elbow_output_dir = Path(output_dir)
            if elbow_subdir is not None:
                elbow_output_dir = elbow_output_dir / elbow_subdir
        optimal_u = find_optimal_k_elbow(
            scaled_features,
            k_min=2,
            k_max=k_max,
            output_dir=elbow_output_dir,
        )
        print(f"[分区] 根据 Elbow 准则自动确定 U = {optimal_u}")

    print(f"正在执行 K-means 聚类 (U={optimal_u})...")
    final_kmeans = KMeans(n_clusters=optimal_u, random_state=42, n_init=10)
    cluster_labels_valid = final_kmeans.fit_predict(scaled_features)

    cluster_labels_1d = np.full(rows * cols, -1, dtype=int)
    cluster_labels_1d[selected_mask_1d] = cluster_labels_valid
    return cluster_labels_1d.reshape(rows, cols), optimal_u


def _detect_partition_boundaries(cluster_matrix):
    valid_matrix = cluster_matrix >= 0
    boundary = np.zeros_like(cluster_matrix, dtype=bool)
    vertical_diff = cluster_matrix[:-1, :] != cluster_matrix[1:, :]
    vertical_valid = valid_matrix[:-1, :] | valid_matrix[1:, :]
    horizontal_diff = cluster_matrix[:, :-1] != cluster_matrix[:, 1:]
    horizontal_valid = valid_matrix[:, :-1] | valid_matrix[:, 1:]
    boundary[:-1, :] |= vertical_diff & vertical_valid
    boundary[1:, :] |= vertical_diff & vertical_valid
    boundary[:, :-1] |= horizontal_diff & horizontal_valid
    boundary[:, 1:] |= horizontal_diff & horizontal_valid
    return boundary


def _save_partition_plot(cluster_matrix, xs, ys, output_dir, file_name, title):
    plt.figure(figsize=(10, 8))
    display_matrix = np.ma.masked_where(cluster_matrix < 0, cluster_matrix)
    plt.imshow(
        display_matrix,
        extent=[xs.min(), xs.max(), ys.max(), ys.min()],
        cmap="Set3",
        aspect="auto",
    )

    boundary = _detect_partition_boundaries(cluster_matrix)
    if np.any(boundary):
        by, bx = np.where(boundary)
        plt.scatter(xs[bx], ys[by], c="black", s=0.3, marker="s", linewidths=0, zorder=2)

    plt.title(title)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.gca().invert_yaxis()
    plt.colorbar(label="Cluster ID")

    if output_dir is not None:
        partition_dir = Path(output_dir) / "partition"
        partition_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(partition_dir / file_name, dpi=300, bbox_inches="tight")
    plt.close()


def _renumber_cluster_matrix(cluster_matrix):
    renumbered = np.full_like(cluster_matrix, -1, dtype=int)
    next_id = 0
    for pid in sorted(np.unique(cluster_matrix).astype(int)):
        if pid < 0:
            continue
        renumbered[cluster_matrix == pid] = next_id
        next_id += 1
    return renumbered, next_id


def _summarize_partition_gradient_complexity(
    partition_id,
    partition_mask,
    gx_matrix,
    gy_matrix,
    unit_gx_matrix,
    unit_gy_matrix,
    direction_valid_mask,
    min_partition_size_for_secondary=12,
    direction_dispersion_threshold=0.35,
):
    cell_count = int(np.count_nonzero(partition_mask))

    gx_values = gx_matrix[partition_mask]
    gy_values = gy_matrix[partition_mask]
    magnitudes = np.hypot(gx_values, gy_values)
    finite_magnitudes = magnitudes[np.isfinite(magnitudes)]
    mean_magnitude = float(np.mean(finite_magnitudes)) if finite_magnitudes.size else 0.0
    std_magnitude = float(np.std(finite_magnitudes)) if finite_magnitudes.size else 0.0
    unit_like_gradient = bool(
        finite_magnitudes.size > 0
        and np.allclose(finite_magnitudes, 1.0, atol=0.05, rtol=0.05)
    )

    direction_points = int(np.count_nonzero(partition_mask & direction_valid_mask))
    direction_dispersion = 0.0
    mean_resultant_length = 1.0
    if direction_points > 0:
        ux = unit_gx_matrix[partition_mask & direction_valid_mask]
        uy = unit_gy_matrix[partition_mask & direction_valid_mask]
        mean_resultant_length = float(np.hypot(np.mean(ux), np.mean(uy)))
        direction_dispersion = float(max(0.0, 1.0 - mean_resultant_length))

    trigger_reasons = []
    if cell_count < min_partition_size_for_secondary:
        trigger_reasons.append("too_small_for_secondary")
        triggered = False
    else:
        if direction_dispersion >= direction_dispersion_threshold:
            trigger_reasons.append("direction_dispersion")
        triggered = bool(trigger_reasons)

    return {
        "partition_id": int(partition_id),
        "cell_count": cell_count,
        "direction_points": direction_points,
        "mean_gradient_magnitude": mean_magnitude,
        "std_gradient_magnitude": std_magnitude,
        "unit_like_gradient": unit_like_gradient,
        "mean_resultant_length": mean_resultant_length,
        "direction_dispersion": direction_dispersion,
        "triggered_secondary_partition": triggered,
        "trigger_reasons": trigger_reasons,
    }


def _write_secondary_partition_diagnostics(
    output_dir,
    first_stage_u,
    final_u,
    primary_feature_names,
    diagnostics,
    direction_dispersion_threshold,
    min_partition_size_for_secondary,
    secondary_k_max,
):
    if output_dir is None:
        return

    partition_dir = Path(output_dir) / "partition"
    partition_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = partition_dir / "secondary_partition_diagnostics.json"
    payload = {
        "primary_features": list(primary_feature_names),
        "secondary_features": list(SECONDARY_FEATURE_NAMES),
        "first_stage_u": int(first_stage_u),
        "final_u": int(final_u),
        "secondary_detection_thresholds": {
            "direction_dispersion_threshold": float(direction_dispersion_threshold),
            "min_partition_size_for_secondary": int(min_partition_size_for_secondary),
            "secondary_k_max": int(secondary_k_max),
        },
        "partitions": diagnostics,
    }
    with diagnostics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def partition_coverage_matrix(
    xs,
    ys,
    coverage_matrix,
    U=None,
    k_max=10,
    output_dir=None,
    boundary_mask=None,
    gx_matrix=None,
    gy_matrix=None,
    primary_feature_mode="xy_coverage",
    secondary_k_max=6,
    min_partition_size_for_secondary=12,
    direction_dispersion_threshold=0.35,
):
    """
    对覆盖次数矩阵进行 K-means 空间聚类并可视化边界

    参数:
        xs, ys: 坐标数组
        coverage_matrix: 覆盖次数矩阵
        U: 指定分区数量（可选）。若传入则直接使用，若为 None 则使用 Elbow 法则自动确定
        k_max: Elbow 法则搜索的最大簇数（仅在 U=None 时生效）
        output_dir: 输出目录路径。若传入则保存分区图到 output_dir/partition/，否则保存到默认路径
        boundary_mask: 有效海域掩码。若提供，则仅对有效格聚类，其余位置标记为 -1
    """
    feature_bundle = _build_feature_stacks(
        xs,
        ys,
        coverage_matrix,
        gx_matrix=gx_matrix,
        gy_matrix=gy_matrix,
        primary_feature_mode=primary_feature_mode,
    )
    rows = feature_bundle["rows"]
    cols = feature_bundle["cols"]
    gx_matrix = feature_bundle["gx_matrix"]
    gy_matrix = feature_bundle["gy_matrix"]
    gradient_magnitude = feature_bundle["gradient_magnitude"]
    direction_valid_mask = feature_bundle["direction_valid_mask"]
    unit_gx_matrix = feature_bundle["unit_gx_matrix"]
    unit_gy_matrix = feature_bundle["unit_gy_matrix"]
    primary_feature_names = feature_bundle["primary_feature_names"]
    primary_features = feature_bundle["primary_features"]
    secondary_features = feature_bundle["secondary_features"]

    if boundary_mask is None:
        valid_mask_1d = np.ones(rows * cols, dtype=bool)
    else:
        valid_mask_1d = np.asarray(boundary_mask, dtype=bool).reshape(-1)

    if not np.any(valid_mask_1d):
        raise ValueError("无有效海域网格可用于聚类，请检查 boundary_mask。")

    print(
        "[分区] 一级分区特征 = "
        f"{primary_feature_names} | 二级分区特征 = {SECONDARY_FEATURE_NAMES}"
    )
    first_stage_cluster_matrix, first_stage_u = _cluster_with_selected_mask(
        primary_features,
        valid_mask_1d,
        rows,
        cols,
        U=U,
        k_max=k_max,
        output_dir=output_dir,
        elbow_subdir="stage1",
    )
    _save_partition_plot(
        first_stage_cluster_matrix,
        xs,
        ys,
        output_dir,
        f"partition_stage1_U={first_stage_u}.png",
        f"Stage-1 Partitioning (U={first_stage_u})",
    )

    merged_cluster_matrix = np.full_like(first_stage_cluster_matrix, -1, dtype=int)
    diagnostics = []
    next_global_id = 0
    partition_ids = [
        int(pid)
        for pid in sorted(np.unique(first_stage_cluster_matrix).astype(int))
        if pid >= 0
    ]
    for partition_id in partition_ids:
        partition_mask = first_stage_cluster_matrix == partition_id
        diagnostic = _summarize_partition_gradient_complexity(
            partition_id,
            partition_mask,
            gx_matrix,
            gy_matrix,
            unit_gx_matrix,
            unit_gy_matrix,
            direction_valid_mask,
            min_partition_size_for_secondary=min_partition_size_for_secondary,
            direction_dispersion_threshold=direction_dispersion_threshold,
        )

        print(
            f"[分区检测] 一级分区{partition_id} | cells={diagnostic['cell_count']} | "
            f"dispersion={diagnostic['direction_dispersion']:.3f} | "
            f"unit_like={diagnostic['unit_like_gradient']} | "
            f"二次分区={diagnostic['triggered_secondary_partition']} | "
            f"reasons={diagnostic['trigger_reasons']}"
        )

        if diagnostic["triggered_secondary_partition"]:
            local_mask_1d = partition_mask.reshape(-1)
            if np.count_nonzero(local_mask_1d) >= 3:
                local_cluster_matrix, local_u = _cluster_with_selected_mask(
                    secondary_features,
                    local_mask_1d,
                    rows,
                    cols,
                    U=None,
                    k_max=secondary_k_max,
                    output_dir=output_dir,
                    elbow_subdir=f"stage2_partition_{partition_id}",
                )
                diagnostic["secondary_partition_count"] = int(local_u)
                for local_label in sorted(np.unique(local_cluster_matrix).astype(int)):
                    if local_label < 0:
                        continue
                    merged_cluster_matrix[local_cluster_matrix == local_label] = next_global_id
                    next_global_id += 1
            else:
                diagnostic["secondary_partition_count"] = 1
                diagnostic["triggered_secondary_partition"] = False
                diagnostic["trigger_reasons"] = ["insufficient_local_samples"]
                merged_cluster_matrix[partition_mask] = next_global_id
                next_global_id += 1
        else:
            diagnostic["secondary_partition_count"] = 1
            merged_cluster_matrix[partition_mask] = next_global_id
            next_global_id += 1

        diagnostics.append(diagnostic)

    final_cluster_matrix, final_u = _renumber_cluster_matrix(merged_cluster_matrix)
    _save_partition_plot(
        final_cluster_matrix,
        xs,
        ys,
        output_dir,
        f"partition_final_U={final_u}.png",
        f"Two-Stage Partitioning (U={final_u})",
    )
    _save_partition_plot(
        final_cluster_matrix,
        xs,
        ys,
        output_dir,
        f"partition_U={final_u}.png",
        f"Two-Stage Partitioning (U={final_u})",
    )
    _write_secondary_partition_diagnostics(
        output_dir,
        first_stage_u,
        final_u,
        primary_feature_names,
        diagnostics,
        direction_dispersion_threshold,
        min_partition_size_for_secondary,
        secondary_k_max,
    )

    print(
        f"[分区] 一级分区数={first_stage_u} | 二次分区合并后总分区数={final_u}"
    )
    return final_cluster_matrix, final_u


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
