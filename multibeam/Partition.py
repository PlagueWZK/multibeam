import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from tool.Data import get_gx, get_gy


PRIMARY_FEATURES_BY_MODE = {
    "xy_coverage": ("X", "Y", "coverage_count"),
    "xy_coverage_depth": ("X", "Y", "coverage_count", "depth"),
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
    primary_feature_mode="coverage_only",
    depth_matrix=None,
):
    rows, cols = coverage_matrix.shape
    grid_x, grid_y = np.meshgrid(xs, ys)

    if primary_feature_mode not in PRIMARY_FEATURES_BY_MODE:
        raise ValueError(
            "不支持的一级分区特征模式："
            f"{primary_feature_mode}，可选={tuple(PRIMARY_FEATURES_BY_MODE)}"
        )

    if depth_matrix is not None:
        depth_matrix = np.asarray(depth_matrix, dtype=float)
        if depth_matrix.shape != (rows, cols):
            raise ValueError(
                "提供的深度矩阵尺寸与 coverage_matrix 不一致："
                f"coverage={coverage_matrix.shape}, depth={depth_matrix.shape}"
            )
    if (
        "depth" in PRIMARY_FEATURES_BY_MODE[primary_feature_mode]
        and depth_matrix is None
    ):
        raise ValueError(
            "一级分区特征模式 xy_coverage_depth 需要提供 depth_matrix。"
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
    elif primary_feature_mode == "xy_coverage_depth":
        feature_x = grid_x.flatten().reshape(-1, 1)
        feature_y = grid_y.flatten().reshape(-1, 1)
        feature_depth = depth_matrix.flatten().reshape(-1, 1)
        primary_features = np.hstack((feature_x, feature_y, feature_c, feature_depth))
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


def _connected_component_offsets(connectivity):
    if connectivity == 4:
        return ((-1, 0), (1, 0), (0, -1), (0, 1))
    if connectivity == 8:
        return (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )
    raise ValueError(
        f"不支持的连通性定义 connectivity={connectivity}，仅支持 4 或 8。"
    )


def _iter_connected_components(mask, connectivity=4):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"连通组件掩码必须是二维矩阵，当前 ndim={mask.ndim}")

    rows, cols = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    offsets = _connected_component_offsets(connectivity)

    for start_row, start_col in np.argwhere(mask):
        start_row = int(start_row)
        start_col = int(start_col)
        if visited[start_row, start_col]:
            continue

        stack = [(start_row, start_col)]
        visited[start_row, start_col] = True
        component_rows = []
        component_cols = []

        while stack:
            row, col = stack.pop()
            component_rows.append(row)
            component_cols.append(col)

            for d_row, d_col in offsets:
                next_row = row + d_row
                next_col = col + d_col
                if next_row < 0 or next_row >= rows or next_col < 0 or next_col >= cols:
                    continue
                if visited[next_row, next_col] or not mask[next_row, next_col]:
                    continue
                visited[next_row, next_col] = True
                stack.append((next_row, next_col))

        yield np.array(component_rows, dtype=int), np.array(component_cols, dtype=int)


def _split_disconnected_partitions(cluster_matrix, connectivity=4):
    """将同一分区 ID 下的非连通区域拆为不同分区。"""
    cluster_matrix = np.asarray(cluster_matrix, dtype=int)
    split_matrix = np.full_like(cluster_matrix, -1, dtype=int)
    diagnostics = []
    next_global_id = 0

    partition_ids = [
        int(pid) for pid in sorted(np.unique(cluster_matrix).astype(int)) if pid >= 0
    ]
    for partition_id in partition_ids:
        partition_mask = cluster_matrix == partition_id
        components = []

        for component_rows, component_cols in _iter_connected_components(
            partition_mask, connectivity=connectivity
        ):
            target_partition_id = next_global_id
            split_matrix[component_rows, component_cols] = target_partition_id
            components.append(
                {
                    "target_partition_id": int(target_partition_id),
                    "cell_count": int(len(component_rows)),
                }
            )
            next_global_id += 1

        diagnostics.append(
            {
                "source_partition_id": int(partition_id),
                "component_count": int(len(components)),
                "split": bool(len(components) > 1),
                "target_partition_ids": [
                    item["target_partition_id"] for item in components
                ],
                "component_cell_counts": [item["cell_count"] for item in components],
            }
        )

    return split_matrix, next_global_id, diagnostics


def _build_partition_area_matrix(cluster_matrix, cell_effective_area=None):
    cluster_matrix = np.asarray(cluster_matrix, dtype=int)
    if cell_effective_area is None:
        return np.ones_like(cluster_matrix, dtype=float), "cell_count"

    area_matrix = np.asarray(cell_effective_area, dtype=float)
    if area_matrix.shape != cluster_matrix.shape:
        raise ValueError(
            "cell_effective_area 尺寸必须与 cluster_matrix 一致："
            f"area={area_matrix.shape}, cluster={cluster_matrix.shape}"
        )
    if np.any(area_matrix < 0):
        raise ValueError("cell_effective_area 不能包含负面积。")
    return np.nan_to_num(area_matrix, nan=0.0, posinf=0.0, neginf=0.0), "effective_area"


def _calculate_partition_areas(cluster_matrix, area_matrix):
    areas = {}
    for partition_id in sorted(np.unique(cluster_matrix).astype(int)):
        if partition_id < 0:
            continue
        areas[int(partition_id)] = float(np.sum(area_matrix[cluster_matrix == partition_id]))
    return areas


def _count_partition_contacts(cluster_matrix, partition_id):
    cluster_matrix = np.asarray(cluster_matrix, dtype=int)
    contacts = {}

    for left_or_top, right_or_bottom in (
        (cluster_matrix[:-1, :], cluster_matrix[1:, :]),
        (cluster_matrix[:, :-1], cluster_matrix[:, 1:]),
    ):
        source_on_first = (
            (left_or_top == partition_id)
            & (right_or_bottom >= 0)
            & (right_or_bottom != partition_id)
        )
        if np.any(source_on_first):
            neighbor_ids, edge_counts = np.unique(
                right_or_bottom[source_on_first], return_counts=True
            )
            for neighbor_id, edge_count in zip(neighbor_ids, edge_counts):
                neighbor_id = int(neighbor_id)
                contacts[neighbor_id] = contacts.get(neighbor_id, 0) + int(edge_count)

        source_on_second = (
            (right_or_bottom == partition_id)
            & (left_or_top >= 0)
            & (left_or_top != partition_id)
        )
        if np.any(source_on_second):
            neighbor_ids, edge_counts = np.unique(
                left_or_top[source_on_second], return_counts=True
            )
            for neighbor_id, edge_count in zip(neighbor_ids, edge_counts):
                neighbor_id = int(neighbor_id)
                contacts[neighbor_id] = contacts.get(neighbor_id, 0) + int(edge_count)

    return contacts


def _select_small_partition_merge_target(
    cluster_matrix,
    partition_id,
    partition_areas,
):
    contacts = _count_partition_contacts(cluster_matrix, partition_id)
    if not contacts:
        return None, 0, contacts

    target_partition_id = max(
        contacts,
        key=lambda neighbor_id: (
            contacts[neighbor_id],
            partition_areas.get(neighbor_id, 0.0),
            -neighbor_id,
        ),
    )
    return target_partition_id, contacts[target_partition_id], contacts


def _merge_small_partitions(
    cluster_matrix,
    cell_effective_area=None,
    small_partition_area_ratio=0.02,
    max_iterations=None,
):
    """将面积过小的分区迭代合并到接触边最多的相邻分区。"""
    if small_partition_area_ratio < 0:
        raise ValueError(
            f"small_partition_area_ratio 必须 >= 0，当前为 {small_partition_area_ratio}"
        )

    working_matrix = np.asarray(cluster_matrix, dtype=int).copy()
    area_matrix, area_source = _build_partition_area_matrix(
        working_matrix,
        cell_effective_area=cell_effective_area,
    )
    total_area = float(np.sum(area_matrix[working_matrix >= 0]))
    threshold_area = float(total_area * small_partition_area_ratio)
    diagnostics = {
        "enabled": True,
        "area_source": area_source,
        "small_partition_area_ratio": float(small_partition_area_ratio),
        "total_area": total_area,
        "threshold_area": threshold_area,
        "merge_count": 0,
        "merges": [],
        "remaining_small_partitions": [],
    }

    partition_areas = _calculate_partition_areas(working_matrix, area_matrix)
    if max_iterations is None:
        max_iterations = max(len(partition_areas), 1)

    if total_area <= 0 or small_partition_area_ratio == 0:
        renumbered_matrix, final_u = _renumber_cluster_matrix(working_matrix)
        diagnostics["final_u"] = int(final_u)
        return renumbered_matrix, final_u, diagnostics

    iteration = 0
    while iteration < max_iterations:
        partition_areas = _calculate_partition_areas(working_matrix, area_matrix)
        small_partition_ids = [
            partition_id
            for partition_id, area in partition_areas.items()
            if area < threshold_area
        ]
        if not small_partition_ids:
            break

        merge_candidates = []
        no_neighbor_partitions = []
        for partition_id in small_partition_ids:
            target_partition_id, contact_edges, contacts = (
                _select_small_partition_merge_target(
                    working_matrix,
                    partition_id,
                    partition_areas,
                )
            )
            if target_partition_id is None:
                no_neighbor_partitions.append(
                    {
                        "partition_id": int(partition_id),
                        "area": float(partition_areas[partition_id]),
                        "area_ratio": float(partition_areas[partition_id] / total_area),
                    }
                )
                continue

            merge_candidates.append(
                {
                    "partition_id": int(partition_id),
                    "target_partition_id": int(target_partition_id),
                    "area": float(partition_areas[partition_id]),
                    "area_ratio": float(partition_areas[partition_id] / total_area),
                    "target_area_before": float(partition_areas[target_partition_id]),
                    "contact_edges": int(contact_edges),
                    "neighbor_contact_edges": {
                        str(int(neighbor_id)): int(edge_count)
                        for neighbor_id, edge_count in sorted(contacts.items())
                    },
                }
            )

        if not merge_candidates:
            diagnostics["remaining_small_partitions"] = no_neighbor_partitions
            break

        merge_action = min(
            merge_candidates,
            key=lambda item: (item["area"], item["partition_id"]),
        )
        source_partition_id = merge_action["partition_id"]
        target_partition_id = merge_action["target_partition_id"]
        working_matrix[working_matrix == source_partition_id] = target_partition_id

        merge_action["iteration"] = int(iteration + 1)
        diagnostics["merges"].append(merge_action)
        diagnostics["merge_count"] = int(len(diagnostics["merges"]))
        iteration += 1

    renumbered_matrix, final_u = _renumber_cluster_matrix(working_matrix)
    final_areas = _calculate_partition_areas(renumbered_matrix, area_matrix)
    diagnostics["final_u"] = int(final_u)
    diagnostics["remaining_small_partitions"] = [
        {
            "partition_id": int(partition_id),
            "area": float(area),
            "area_ratio": float(area / total_area),
        }
        for partition_id, area in final_areas.items()
        if area < threshold_area
    ]
    return renumbered_matrix, final_u, diagnostics


def _count_partition_boundary_grid_contacts(cluster_matrix, partition_id):
    """统计分区自身边界网格数，以及这些边界网格接触各相邻分区的数量。"""
    cluster_matrix = np.asarray(cluster_matrix, dtype=int)
    rows, cols = cluster_matrix.shape
    total_boundary_cells = 0
    neighbor_contact_cells = {}

    for row, col in np.argwhere(cluster_matrix == partition_id):
        row = int(row)
        col = int(col)
        is_boundary_cell = False
        touched_neighbor_ids = set()

        for d_row, d_col in _connected_component_offsets(4):
            next_row = row + d_row
            next_col = col + d_col
            if next_row < 0 or next_row >= rows or next_col < 0 or next_col >= cols:
                is_boundary_cell = True
                continue

            neighbor_id = int(cluster_matrix[next_row, next_col])
            if neighbor_id == partition_id:
                continue

            is_boundary_cell = True
            if neighbor_id >= 0:
                touched_neighbor_ids.add(neighbor_id)

        if not is_boundary_cell:
            continue

        total_boundary_cells += 1
        for neighbor_id in touched_neighbor_ids:
            neighbor_contact_cells[neighbor_id] = (
                neighbor_contact_cells.get(neighbor_id, 0) + 1
            )

    return total_boundary_cells, neighbor_contact_cells


def _select_boundary_dominance_merge_target(
    cluster_matrix,
    partition_id,
    partition_areas,
    boundary_contact_ratio_threshold,
):
    total_boundary_cells, neighbor_contact_cells = (
        _count_partition_boundary_grid_contacts(cluster_matrix, partition_id)
    )
    if total_boundary_cells == 0 or not neighbor_contact_cells:
        return None, total_boundary_cells, neighbor_contact_cells

    eligible_neighbors = []
    for neighbor_id, contact_cells in neighbor_contact_cells.items():
        contact_ratio = contact_cells / total_boundary_cells
        if contact_ratio > boundary_contact_ratio_threshold:
            eligible_neighbors.append((neighbor_id, contact_cells, contact_ratio))

    if not eligible_neighbors:
        return None, total_boundary_cells, neighbor_contact_cells

    target_partition_id, _, _ = max(
        eligible_neighbors,
        key=lambda item: (
            item[2],
            item[1],
            partition_areas.get(item[0], 0.0),
            -item[0],
        ),
    )
    return target_partition_id, total_boundary_cells, neighbor_contact_cells


def _merge_boundary_dominant_partitions(
    cluster_matrix,
    cell_effective_area=None,
    boundary_partition_area_ratio=0.10,
    boundary_contact_ratio_threshold=0.50,
    max_iterations=None,
):
    """将边界主要接触同一邻区的 10% 以下分区合并到该邻区。"""
    if boundary_partition_area_ratio < 0:
        raise ValueError(
            "boundary_partition_area_ratio 必须 >= 0，"
            f"当前为 {boundary_partition_area_ratio}"
        )
    if not 0 <= boundary_contact_ratio_threshold <= 1:
        raise ValueError(
            "boundary_contact_ratio_threshold 必须位于 [0, 1]，"
            f"当前为 {boundary_contact_ratio_threshold}"
        )

    working_matrix = np.asarray(cluster_matrix, dtype=int).copy()
    area_matrix, area_source = _build_partition_area_matrix(
        working_matrix,
        cell_effective_area=cell_effective_area,
    )
    total_area = float(np.sum(area_matrix[working_matrix >= 0]))
    threshold_area = float(total_area * boundary_partition_area_ratio)
    diagnostics = {
        "enabled": True,
        "area_source": area_source,
        "boundary_partition_area_ratio": float(boundary_partition_area_ratio),
        "boundary_contact_ratio_threshold": float(boundary_contact_ratio_threshold),
        "total_area": total_area,
        "threshold_area": threshold_area,
        "merge_count": 0,
        "merges": [],
        "remaining_under_threshold_partitions": [],
    }

    partition_areas = _calculate_partition_areas(working_matrix, area_matrix)
    if max_iterations is None:
        max_iterations = max(len(partition_areas), 1)

    if total_area <= 0 or boundary_partition_area_ratio == 0:
        renumbered_matrix, final_u = _renumber_cluster_matrix(working_matrix)
        diagnostics["final_u"] = int(final_u)
        return renumbered_matrix, final_u, diagnostics

    iteration = 0
    while iteration < max_iterations:
        partition_areas = _calculate_partition_areas(working_matrix, area_matrix)
        candidate_ids = [
            partition_id
            for partition_id, area in partition_areas.items()
            if area < threshold_area
        ]
        if not candidate_ids:
            break

        merge_candidates = []
        for partition_id in candidate_ids:
            target_partition_id, total_boundary_cells, neighbor_contact_cells = (
                _select_boundary_dominance_merge_target(
                    working_matrix,
                    partition_id,
                    partition_areas,
                    boundary_contact_ratio_threshold,
                )
            )
            neighbor_contact_ratios = {
                int(neighbor_id): (
                    contact_cells / total_boundary_cells
                    if total_boundary_cells > 0
                    else 0.0
                )
                for neighbor_id, contact_cells in neighbor_contact_cells.items()
            }
            if target_partition_id is None:
                continue

            contact_cells = int(neighbor_contact_cells[target_partition_id])
            contact_ratio = float(neighbor_contact_ratios[target_partition_id])
            merge_candidates.append(
                {
                    "partition_id": int(partition_id),
                    "target_partition_id": int(target_partition_id),
                    "area": float(partition_areas[partition_id]),
                    "area_ratio": float(partition_areas[partition_id] / total_area),
                    "target_area_before": float(partition_areas[target_partition_id]),
                    "total_boundary_cells": int(total_boundary_cells),
                    "target_contact_boundary_cells": contact_cells,
                    "target_contact_boundary_ratio": contact_ratio,
                    "neighbor_contact_boundary_cells": {
                        str(int(neighbor_id)): int(contact_cells)
                        for neighbor_id, contact_cells in sorted(
                            neighbor_contact_cells.items()
                        )
                    },
                    "neighbor_contact_boundary_ratios": {
                        str(int(neighbor_id)): float(contact_ratio)
                        for neighbor_id, contact_ratio in sorted(
                            neighbor_contact_ratios.items()
                        )
                    },
                }
            )

        if not merge_candidates:
            break

        merge_action = min(
            merge_candidates,
            key=lambda item: (item["area"], item["partition_id"]),
        )
        source_partition_id = merge_action["partition_id"]
        target_partition_id = merge_action["target_partition_id"]
        working_matrix[working_matrix == source_partition_id] = target_partition_id

        merge_action["iteration"] = int(iteration + 1)
        diagnostics["merges"].append(merge_action)
        diagnostics["merge_count"] = int(len(diagnostics["merges"]))
        iteration += 1

    renumbered_matrix, final_u = _renumber_cluster_matrix(working_matrix)
    final_areas = _calculate_partition_areas(renumbered_matrix, area_matrix)
    diagnostics["final_u"] = int(final_u)
    diagnostics["remaining_under_threshold_partitions"] = [
        {
            "partition_id": int(partition_id),
            "area": float(area),
            "area_ratio": float(area / total_area),
        }
        for partition_id, area in final_areas.items()
        if area < threshold_area
    ]
    return renumbered_matrix, final_u, diagnostics


def _summarize_partition_gradient_complexity(
    partition_id,
    partition_mask,
    gx_matrix,
    gy_matrix,
    unit_gx_matrix,
    unit_gy_matrix,
    direction_valid_mask,
    min_partition_size_for_secondary=12,
    direction_dispersion_threshold=0.30,
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


def _write_connectivity_partition_diagnostics(
    output_dir,
    pre_connectivity_u,
    final_u,
    connectivity,
    diagnostics,
):
    if output_dir is None:
        return

    partition_dir = Path(output_dir) / "partition"
    partition_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = partition_dir / "connectivity_partition_diagnostics.json"
    split_partition_count = int(
        sum(1 for diagnostic in diagnostics if diagnostic["split"])
    )
    payload = {
        "connectivity": int(connectivity),
        "connectivity_definition": (
            "edge-connected grid cells"
            if connectivity == 4
            else "edge-or-corner connected grid cells"
        ),
        "pre_connectivity_u": int(pre_connectivity_u),
        "final_u": int(final_u),
        "split_partition_count": split_partition_count,
        "added_partition_count": int(final_u - pre_connectivity_u),
        "partitions": diagnostics,
    }
    with diagnostics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_small_partition_merge_diagnostics(output_dir, diagnostics):
    if output_dir is None:
        return

    partition_dir = Path(output_dir) / "partition"
    partition_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = partition_dir / "small_partition_merge_diagnostics.json"
    with diagnostics_path.open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)


def _write_boundary_dominance_merge_diagnostics(output_dir, diagnostics):
    if output_dir is None:
        return

    partition_dir = Path(output_dir) / "partition"
    partition_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = partition_dir / "boundary_dominance_merge_diagnostics.json"
    with diagnostics_path.open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)


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
    primary_feature_mode="coverage_only",
    secondary_k_max=6,
    min_partition_size_for_secondary=12,
    direction_dispersion_threshold=0.30,
    connectivity=4,
    cell_effective_area=None,
    merge_small_partitions=True,
    small_partition_area_ratio=0.02,
    merge_boundary_dominance_partitions=True,
    boundary_partition_area_ratio=0.10,
    boundary_contact_ratio_threshold=0.50,
    depth_matrix=None,
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
        gx_matrix, gy_matrix: 梯度矩阵；若提供则复用 Coverage 阶段预计算结果
        primary_feature_mode: 一级分区特征模式。xy_coverage_depth 表示 X/Y/覆盖次数/深度
        depth_matrix: 深度矩阵；primary_feature_mode=xy_coverage_depth 时必须提供
        connectivity: 分区后处理的连通性定义。默认 4 表示边连通；8 表示边或角连通
        cell_effective_area: 每个网格在待测海域内的真实有效面积；提供时用于小面积分区合并
        merge_small_partitions: 是否在连通性拆分后合并小于阈值的小面积分区
        small_partition_area_ratio: 小面积分区阈值，默认 0.02 表示待测海域总面积的 2%
        merge_boundary_dominance_partitions: 是否启用 10% 以下分区边界占比合并
        boundary_partition_area_ratio: 边界占比合并候选面积阈值，默认 0.10 表示 10%
        boundary_contact_ratio_threshold: 相邻分区接触边界网格占比阈值，默认 0.50
    """
    feature_bundle = _build_feature_stacks(
        xs,
        ys,
        coverage_matrix,
        gx_matrix=gx_matrix,
        gy_matrix=gy_matrix,
        primary_feature_mode=primary_feature_mode,
        depth_matrix=depth_matrix,
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
    _save_partition_plot(
        first_stage_cluster_matrix,
        xs,
        ys,
        output_dir,
        f"partition_stage1_primary_U={first_stage_u}.png",
        f"Stage 1: Primary Partitioning (U={first_stage_u})",
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
                _save_partition_plot(
                    local_cluster_matrix,
                    xs,
                    ys,
                    output_dir,
                    f"partition_stage2_secondary_source_{partition_id}_U={local_u}.png",
                    f"Stage 2: Secondary Partitioning for Source {partition_id} (U={local_u})",
                )
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

    post_secondary_cluster_matrix, post_secondary_u = _renumber_cluster_matrix(
        merged_cluster_matrix
    )
    _save_partition_plot(
        post_secondary_cluster_matrix,
        xs,
        ys,
        output_dir,
        f"partition_stage2_secondary_merged_U={post_secondary_u}.png",
        f"Stage 2: Secondary-Merged Global Partitioning (U={post_secondary_u})",
    )
    connected_cluster_matrix, connected_u, connectivity_diagnostics = (
        _split_disconnected_partitions(
            post_secondary_cluster_matrix,
            connectivity=connectivity,
        )
    )
    _save_partition_plot(
        connected_cluster_matrix,
        xs,
        ys,
        output_dir,
        f"partition_stage3_connectivity_split_U={connected_u}.png",
        f"Stage 3: Connectivity Split Partitioning (U={connected_u})",
    )
    if merge_small_partitions:
        final_cluster_matrix, final_u, small_merge_diagnostics = _merge_small_partitions(
            connected_cluster_matrix,
            cell_effective_area=cell_effective_area,
            small_partition_area_ratio=small_partition_area_ratio,
        )
    else:
        final_cluster_matrix = connected_cluster_matrix
        final_u = connected_u
        small_merge_diagnostics = {
            "enabled": False,
            "small_partition_area_ratio": float(small_partition_area_ratio),
            "pre_merge_u": int(connected_u),
            "final_u": int(final_u),
            "merge_count": 0,
            "merges": [],
            "remaining_small_partitions": [],
        }

    _save_partition_plot(
        final_cluster_matrix,
        xs,
        ys,
        output_dir,
        f"partition_stage4_small_area_merged_U={final_u}.png",
        f"Stage 4: Small-Area-Merged Final Partitioning (U={final_u})",
    )

    small_merged_cluster_matrix = final_cluster_matrix
    small_merged_u = final_u
    if merge_boundary_dominance_partitions:
        final_cluster_matrix, final_u, boundary_dominance_diagnostics = (
            _merge_boundary_dominant_partitions(
                small_merged_cluster_matrix,
                cell_effective_area=cell_effective_area,
                boundary_partition_area_ratio=boundary_partition_area_ratio,
                boundary_contact_ratio_threshold=boundary_contact_ratio_threshold,
            )
        )
    else:
        boundary_dominance_diagnostics = {
            "enabled": False,
            "boundary_partition_area_ratio": float(boundary_partition_area_ratio),
            "boundary_contact_ratio_threshold": float(
                boundary_contact_ratio_threshold
            ),
            "pre_merge_u": int(small_merged_u),
            "final_u": int(final_u),
            "merge_count": 0,
            "merges": [],
            "remaining_under_threshold_partitions": [],
        }

    _save_partition_plot(
        final_cluster_matrix,
        xs,
        ys,
        output_dir,
        f"partition_stage5_boundary_dominance_merged_U={final_u}.png",
        f"Stage 5: Boundary-Dominance-Merged Final Partitioning (U={final_u})",
    )
    _save_partition_plot(
        final_cluster_matrix,
        xs,
        ys,
        output_dir,
        f"partition_final_U={final_u}.png",
        f"Connected and Small-Merged Final Partitioning (U={final_u})",
    )
    _save_partition_plot(
        final_cluster_matrix,
        xs,
        ys,
        output_dir,
        f"partition_U={final_u}.png",
        f"Connected and Small-Merged Final Partitioning (U={final_u})",
    )
    _write_secondary_partition_diagnostics(
        output_dir,
        first_stage_u,
        post_secondary_u,
        primary_feature_names,
        diagnostics,
        direction_dispersion_threshold,
        min_partition_size_for_secondary,
        secondary_k_max,
    )
    _write_connectivity_partition_diagnostics(
        output_dir,
        post_secondary_u,
        connected_u,
        connectivity,
        connectivity_diagnostics,
    )
    _write_small_partition_merge_diagnostics(
        output_dir,
        small_merge_diagnostics,
    )
    _write_boundary_dominance_merge_diagnostics(
        output_dir,
        boundary_dominance_diagnostics,
    )

    split_partition_count = sum(
        1 for diagnostic in connectivity_diagnostics if diagnostic["split"]
    )
    small_merge_count = int(small_merge_diagnostics.get("merge_count", 0))
    boundary_dominance_merge_count = int(
        boundary_dominance_diagnostics.get("merge_count", 0)
    )
    print(
        f"[分区] 一级分区数={first_stage_u} | "
        f"二次分区合并后总分区数={post_secondary_u} | "
        f"连通性拆分后总分区数={connected_u} | "
        f"小面积合并后总分区数={small_merged_u} | "
        f"边界占比合并后总分区数={final_u} | "
        f"发生拆分的分区数={split_partition_count} | "
        f"小面积合并次数={small_merge_count} | "
        f"边界占比合并次数={boundary_dominance_merge_count}"
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
