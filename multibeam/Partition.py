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
    "xy_depth_coverage": ("X", "Y", "depth", "coverage_count"),
    "direction_only": ("ux", "uy"),
    "xy_depth_coverage_direction": (
        "X",
        "Y",
        "depth",
        "coverage_count",
        "ux",
        "uy",
    ),
    "coverage_direction": ("coverage_count", "ux", "uy"),
}
SECONDARY_FEATURES_BY_MODE = {
    "direction_only": ("ux", "uy"),
    "depth_coverage": ("depth", "coverage_count"),
}
SECONDARY_PARTITION_POLICIES = (
    "auto_by_direction_dispersion",
    "all_partitions",
)


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
    depth_matrix=None,
    gx_matrix=None,
    gy_matrix=None,
    primary_feature_mode="coverage_only",
    secondary_feature_mode="direction_only",
):
    rows, cols = coverage_matrix.shape
    grid_x, grid_y = np.meshgrid(xs, ys)

    if primary_feature_mode not in PRIMARY_FEATURES_BY_MODE:
        raise ValueError(
            "不支持的一级分区特征模式："
            f"{primary_feature_mode}，可选={tuple(PRIMARY_FEATURES_BY_MODE)}"
        )

    if secondary_feature_mode not in SECONDARY_FEATURES_BY_MODE:
        raise ValueError(
            "不支持的二级分区特征模式："
            f"{secondary_feature_mode}，可选={tuple(SECONDARY_FEATURES_BY_MODE)}"
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
    feature_depth = None
    if depth_matrix is not None:
        depth_matrix = np.asarray(depth_matrix, dtype=float)
        if depth_matrix.shape != (rows, cols):
            raise ValueError(
                "提供的深度矩阵尺寸与 coverage_matrix 不一致："
                f"coverage={coverage_matrix.shape}, depth={depth_matrix.shape}"
            )
        feature_depth = depth_matrix.flatten().reshape(-1, 1)

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

    feature_x = grid_x.flatten().reshape(-1, 1)
    feature_y = grid_y.flatten().reshape(-1, 1)
    feature_ux = unit_gx_matrix.reshape(-1, 1)
    feature_uy = unit_gy_matrix.reshape(-1, 1)

    if primary_feature_mode == "xy_coverage":
        primary_features = np.hstack((feature_x, feature_y, feature_c))
    elif primary_feature_mode == "coverage_only":
        primary_features = feature_c
    elif primary_feature_mode == "xy_depth_coverage":
        if feature_depth is None:
            raise ValueError(
                "primary_feature_mode='xy_depth_coverage' 需要提供 depth_matrix。"
            )
        primary_features = np.hstack((feature_x, feature_y, feature_depth, feature_c))
    elif primary_feature_mode == "direction_only":
        primary_features = np.hstack((feature_ux, feature_uy))
    elif primary_feature_mode == "xy_depth_coverage_direction":
        if feature_depth is None:
            raise ValueError(
                "primary_feature_mode='xy_depth_coverage_direction' 需要提供 depth_matrix。"
            )
        primary_features = np.hstack(
            (feature_x, feature_y, feature_depth, feature_c, feature_ux, feature_uy)
        )
    elif primary_feature_mode == "coverage_direction":
        primary_features = np.hstack((feature_c, feature_ux, feature_uy))
    else:
        raise ValueError(
            "不支持的一级分区特征模式："
            f"{primary_feature_mode}，可选={tuple(PRIMARY_FEATURES_BY_MODE)}"
        )

    if secondary_feature_mode == "direction_only":
        secondary_features = np.hstack(
            (
                unit_gx_matrix.reshape(-1, 1),
                unit_gy_matrix.reshape(-1, 1),
            )
        )
    elif secondary_feature_mode == "depth_coverage":
        if feature_depth is None:
            raise ValueError(
                "secondary_feature_mode='depth_coverage' 需要提供 depth_matrix。"
            )
        secondary_features = np.hstack((feature_depth, feature_c))
    else:
        raise ValueError(
            "不支持的二级分区特征模式："
            f"{secondary_feature_mode}，可选={tuple(SECONDARY_FEATURES_BY_MODE)}"
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
        "secondary_feature_names": SECONDARY_FEATURES_BY_MODE[secondary_feature_mode],
        "secondary_features": secondary_features,
    }


def _scale_features_for_kmeans(valid_features, feature_scaling_mode="legacy_balanced"):
    weighted_features = valid_features.copy().astype(float)

    if feature_scaling_mode == "legacy_balanced":
        if valid_features.shape[1] >= 3:
            std_xy_raw = float(np.std(valid_features[:, :2]))
            weighted_features[:, 2] *= std_xy_raw / (
                float(np.std(valid_features[:, 2])) + 1e-6
            )

            if valid_features.shape[1] >= 5:
                weighted_features[:, 3:5] *= std_xy_raw / (
                    float(np.std(valid_features[:, 3:5])) + 1e-6
                )
    elif feature_scaling_mode != "equal_weight_standardized":
        raise ValueError(
            "不支持的特征缩放模式："
            f"{feature_scaling_mode}，可选=('legacy_balanced', 'equal_weight_standardized')"
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
    feature_scaling_mode="legacy_balanced",
):
    valid_features = features[selected_mask_1d]
    n_samples = len(valid_features)
    if n_samples == 0:
        raise ValueError("无有效海域网格可用于聚类，请检查有效掩码。")
    if n_samples < 3:
        raise ValueError(f"可用于聚类的数据点过少 (n={n_samples})，至少需要 3 个点。")

    scaled_features = _scale_features_for_kmeans(
        valid_features, feature_scaling_mode=feature_scaling_mode
    )

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


def _split_disconnected_partitions(cluster_matrix, valid_mask=None):
    """将同一分区标签下 4 邻接不连通的物理区域拆成不同分区。"""
    cluster_matrix = np.asarray(cluster_matrix, dtype=int)
    rows_count, cols_count = cluster_matrix.shape
    if valid_mask is None:
        valid_mask = cluster_matrix >= 0
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)

    original_ids = [
        int(pid) for pid in sorted(np.unique(cluster_matrix).astype(int)) if pid >= 0
    ]
    split_matrix = np.full_like(cluster_matrix, -1, dtype=int)
    visited = np.zeros_like(cluster_matrix, dtype=bool)
    next_id = 0
    component_stats = []

    for original_id in original_ids:
        component_sizes = []
        component_new_ids = []
        candidate_cells = np.argwhere((cluster_matrix == original_id) & valid_mask)
        for start_row, start_col in candidate_cells:
            start_row = int(start_row)
            start_col = int(start_col)
            if visited[start_row, start_col]:
                continue

            stack = [(start_row, start_col)]
            visited[start_row, start_col] = True
            cells = []
            while stack:
                row, col = stack.pop()
                cells.append((row, col))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = row + dr
                    nc = col + dc
                    if nr < 0 or nr >= rows_count or nc < 0 or nc >= cols_count:
                        continue
                    if visited[nr, nc]:
                        continue
                    if not valid_mask[nr, nc]:
                        continue
                    if int(cluster_matrix[nr, nc]) != original_id:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))

            for row, col in cells:
                split_matrix[row, col] = next_id
            component_sizes.append(int(len(cells)))
            component_new_ids.append(int(next_id))
            next_id += 1

        if component_sizes:
            component_stats.append(
                {
                    "original_partition_id": int(original_id),
                    "component_count": int(len(component_sizes)),
                    "component_partition_ids": component_new_ids,
                    "total_cells": int(sum(component_sizes)),
                    "min_component_cells": int(min(component_sizes)),
                    "max_component_cells": int(max(component_sizes)),
                    "mean_component_cells": float(np.mean(component_sizes)),
                }
            )

    split_original_ids = [
        item["original_partition_id"]
        for item in component_stats
        if item["component_count"] > 1
    ]
    summary = {
        "enabled": True,
        "applied": bool(split_original_ids),
        "connectivity": "4-neighbor_shared_edge",
        "pre_split_u": int(len(original_ids)),
        "post_split_u": int(next_id),
        "split_original_partition_ids": split_original_ids,
        "additional_partition_count": int(next_id - len(original_ids)),
        "components_by_original_partition": component_stats,
    }
    return split_matrix, summary


def _compute_total_valid_area(valid_mask, cell_effective_area=None):
    if cell_effective_area is None:
        return float(np.count_nonzero(valid_mask))
    return float(np.sum(np.asarray(cell_effective_area, dtype=float)[valid_mask]))


def _compute_partition_area_map(cluster_matrix, valid_mask, cell_effective_area=None):
    area_map = {}
    for pid in sorted(np.unique(cluster_matrix).astype(int)):
        if pid < 0:
            continue
        partition_mask = (cluster_matrix == pid) & valid_mask
        if cell_effective_area is None:
            area_map[int(pid)] = float(np.count_nonzero(partition_mask))
        else:
            area_map[int(pid)] = float(
                np.sum(np.asarray(cell_effective_area, dtype=float)[partition_mask])
            )
    return area_map


def _build_partition_contact_edge_counts(cluster_matrix):
    contact_counts = {}

    def _add_contact(a, b):
        a = int(a)
        b = int(b)
        if a < 0 or b < 0 or a == b:
            return
        contact_counts.setdefault(a, {})
        contact_counts.setdefault(b, {})
        contact_counts[a][b] = contact_counts[a].get(b, 0) + 1
        contact_counts[b][a] = contact_counts[b].get(a, 0) + 1

    upper = cluster_matrix[:-1, :]
    lower = cluster_matrix[1:, :]
    vertical_valid = (upper >= 0) & (lower >= 0) & (upper != lower)
    for a, b in zip(upper[vertical_valid], lower[vertical_valid]):
        _add_contact(a, b)

    left = cluster_matrix[:, :-1]
    right = cluster_matrix[:, 1:]
    horizontal_valid = (left >= 0) & (right >= 0) & (left != right)
    for a, b in zip(left[horizontal_valid], right[horizontal_valid]):
        _add_contact(a, b)

    return contact_counts


def _select_best_merge_target(neighbor_counts, area_map):
    if not neighbor_counts:
        return None, 0

    best_target, best_contact_count = min(
        neighbor_counts.items(),
        key=lambda item: (
            -int(item[1]),
            -float(area_map.get(int(item[0]), 0.0)),
            int(item[0]),
        ),
    )
    return int(best_target), int(best_contact_count)


def _merge_small_partitions(
    cluster_matrix,
    valid_mask,
    total_valid_area,
    *,
    area_ratio_threshold=0.10,
    cell_effective_area=None,
):
    updated = np.array(cluster_matrix, copy=True)
    merge_events = []
    skipped_events = []

    while True:
        merged_in_pass = False
        partition_ids = [
            int(pid) for pid in sorted(np.unique(updated).astype(int)) if int(pid) >= 0
        ]
        for pid in partition_ids:
            if not np.any(updated == pid):
                continue

            area_map = _compute_partition_area_map(
                updated, valid_mask, cell_effective_area=cell_effective_area
            )
            partition_area = float(area_map.get(pid, 0.0))
            area_ratio = partition_area / total_valid_area if total_valid_area > 0 else 0.0
            if area_ratio >= area_ratio_threshold:
                continue

            contact_counts = _build_partition_contact_edge_counts(updated).get(pid, {})
            target_pid, contact_edge_count = _select_best_merge_target(
                contact_counts, area_map
            )
            if target_pid is None:
                skipped_events.append(
                    {
                        "rule": "small_partition_merge",
                        "source_partition_id": int(pid),
                        "source_area": partition_area,
                        "source_area_ratio": float(area_ratio),
                        "status": "skipped_no_adjacent_partition",
                    }
                )
                continue

            target_area_before_merge = float(area_map.get(target_pid, 0.0))
            updated[updated == pid] = target_pid
            merge_events.append(
                {
                    "rule": "small_partition_merge",
                    "source_partition_id": int(pid),
                    "target_partition_id": int(target_pid),
                    "source_area": partition_area,
                    "source_area_ratio": float(area_ratio),
                    "target_area_before_merge": target_area_before_merge,
                    "contact_edge_count": int(contact_edge_count),
                }
            )
            merged_in_pass = True

        if not merged_in_pass:
            break

    return updated, merge_events, skipped_events


def _compute_partition_neighbor_boundary_summary(cluster_matrix, partition_id):
    rows, cols = np.where(cluster_matrix == partition_id)
    if len(rows) == 0:
        return {
            "eligible_boundary_cells": 0,
            "neighbor_boundary_cell_counts": {},
        }

    rows_count, cols_count = cluster_matrix.shape
    eligible_boundary_cells = 0
    neighbor_boundary_cell_counts = {}
    for row, col in zip(rows.tolist(), cols.tolist()):
        neighbor_ids = set()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr = row + dr
            nc = col + dc
            if nr < 0 or nr >= rows_count or nc < 0 or nc >= cols_count:
                continue
            neighbor_pid = int(cluster_matrix[nr, nc])
            if neighbor_pid >= 0 and neighbor_pid != partition_id:
                neighbor_ids.add(neighbor_pid)

        if not neighbor_ids:
            continue

        eligible_boundary_cells += 1
        for neighbor_pid in neighbor_ids:
            neighbor_boundary_cell_counts[neighbor_pid] = (
                neighbor_boundary_cell_counts.get(neighbor_pid, 0) + 1
            )

    return {
        "eligible_boundary_cells": int(eligible_boundary_cells),
        "neighbor_boundary_cell_counts": {
            int(pid): int(count)
            for pid, count in sorted(neighbor_boundary_cell_counts.items())
        },
    }


def _merge_enclosed_partitions(
    cluster_matrix,
    *,
    boundary_ratio_threshold=0.50,
    valid_mask=None,
    cell_effective_area=None,
    total_valid_area=None,
    source_area_ratio_min=None,
    source_area_ratio_max=None,
):
    updated = np.array(cluster_matrix, copy=True)
    merge_events = []

    while True:
        merged_in_pass = False
        partition_ids = [
            int(pid) for pid in sorted(np.unique(updated).astype(int)) if int(pid) >= 0
        ]
        for pid in partition_ids:
            if not np.any(updated == pid):
                continue

            area_map = None
            if valid_mask is not None:
                area_map = _compute_partition_area_map(
                    updated, valid_mask, cell_effective_area=cell_effective_area
                )
            else:
                area_map = {int(pid_): 0.0 for pid_ in partition_ids}

            source_area = float(area_map.get(pid, 0.0))
            source_area_ratio = (
                source_area / float(total_valid_area)
                if total_valid_area is not None and float(total_valid_area) > 0
                else 0.0
            )
            if (
                source_area_ratio_min is not None
                and source_area_ratio < float(source_area_ratio_min)
            ):
                continue
            if (
                source_area_ratio_max is not None
                and source_area_ratio > float(source_area_ratio_max)
            ):
                continue

            summary = _compute_partition_neighbor_boundary_summary(updated, pid)
            eligible_boundary_cells = int(summary["eligible_boundary_cells"])
            if eligible_boundary_cells <= 0:
                continue

            target_pid, adjacent_boundary_cell_count = _select_best_merge_target(
                summary["neighbor_boundary_cell_counts"], area_map
            )
            if target_pid is None:
                continue

            boundary_ratio = (
                adjacent_boundary_cell_count / eligible_boundary_cells
                if eligible_boundary_cells > 0
                else 0.0
            )
            if boundary_ratio <= boundary_ratio_threshold:
                continue

            updated[updated == pid] = target_pid
            merge_events.append(
                {
                    "rule": "enclosed_partition_merge",
                    "source_partition_id": int(pid),
                    "target_partition_id": int(target_pid),
                    "source_area": source_area,
                    "source_area_ratio": float(source_area_ratio),
                    "eligible_boundary_cells": int(eligible_boundary_cells),
                    "adjacent_boundary_cell_count": int(adjacent_boundary_cell_count),
                    "boundary_ratio": float(boundary_ratio),
                }
            )
            merged_in_pass = True

        if not merged_in_pass:
            break

    return updated, merge_events


def _postprocess_partition_matrix(
    cluster_matrix,
    valid_mask,
    *,
    cell_effective_area=None,
    merge_small_partitions=False,
    small_partition_area_ratio_threshold=0.10,
    merge_enclosed_partitions=False,
    enclosed_partition_boundary_ratio_threshold=0.50,
    enclosed_partition_source_area_ratio_min=None,
    enclosed_partition_source_area_ratio_max=None,
):
    total_valid_area = _compute_total_valid_area(
        valid_mask, cell_effective_area=cell_effective_area
    )
    pre_postprocess_u = int(
        np.count_nonzero(np.unique(cluster_matrix.astype(int)) >= 0)
    )
    area_unit = "effective_area_m2" if cell_effective_area is not None else "grid_cells"
    payload = {
        "enabled": bool(merge_small_partitions or merge_enclosed_partitions),
        "area_unit": area_unit,
        "total_valid_area": float(total_valid_area),
        "small_partition_merge_enabled": bool(merge_small_partitions),
        "small_partition_area_ratio_threshold": float(
            small_partition_area_ratio_threshold
        ),
        "enclosed_partition_merge_enabled": bool(merge_enclosed_partitions),
        "enclosed_partition_boundary_ratio_threshold": float(
            enclosed_partition_boundary_ratio_threshold
        ),
        "enclosed_partition_source_area_ratio_min": (
            None
            if enclosed_partition_source_area_ratio_min is None
            else float(enclosed_partition_source_area_ratio_min)
        ),
        "enclosed_partition_source_area_ratio_max": (
            None
            if enclosed_partition_source_area_ratio_max is None
            else float(enclosed_partition_source_area_ratio_max)
        ),
        "pre_postprocess_final_u": int(pre_postprocess_u),
        "postprocess_changed": False,
        "small_partition_merges": [],
        "small_partition_skipped": [],
        "enclosed_partition_merges": [],
        "final_u_after_postprocess": int(pre_postprocess_u),
    }
    if not payload["enabled"]:
        return np.array(cluster_matrix, copy=True), payload

    updated = np.array(cluster_matrix, copy=True)
    if merge_small_partitions:
        updated, merge_events, skipped_events = _merge_small_partitions(
            updated,
            valid_mask,
            total_valid_area,
            area_ratio_threshold=small_partition_area_ratio_threshold,
            cell_effective_area=cell_effective_area,
        )
        payload["small_partition_merges"] = merge_events
        payload["small_partition_skipped"] = skipped_events

    if merge_enclosed_partitions:
        updated, merge_events = _merge_enclosed_partitions(
            updated,
            boundary_ratio_threshold=enclosed_partition_boundary_ratio_threshold,
            valid_mask=valid_mask,
            cell_effective_area=cell_effective_area,
            total_valid_area=total_valid_area,
            source_area_ratio_min=enclosed_partition_source_area_ratio_min,
            source_area_ratio_max=enclosed_partition_source_area_ratio_max,
        )
        payload["enclosed_partition_merges"] = merge_events

    updated, final_u = _renumber_cluster_matrix(updated)
    payload["postprocess_changed"] = not np.array_equal(updated, cluster_matrix)
    payload["final_u_after_postprocess"] = int(final_u)
    return updated, payload


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
        if direction_dispersion > direction_dispersion_threshold:
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
    pre_postprocess_u,
    final_u,
    primary_feature_names,
    secondary_feature_names,
    primary_connected_component_split,
    diagnostics,
    direction_dispersion_threshold,
    min_partition_size_for_secondary,
    secondary_k_max,
    enable_secondary_partition,
    feature_scaling_mode,
    secondary_partition_policy,
    postprocess_summary,
):
    if output_dir is None:
        return

    partition_dir = Path(output_dir) / "partition"
    partition_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = partition_dir / "secondary_partition_diagnostics.json"
    payload = {
        "primary_features": list(primary_feature_names),
        "secondary_features": list(secondary_feature_names),
        "first_stage_u": int(first_stage_u),
        "pre_postprocess_u": int(pre_postprocess_u),
        "final_u": int(final_u),
        "secondary_partition_enabled": bool(enable_secondary_partition),
        "secondary_partition_policy": secondary_partition_policy,
        "feature_scaling_mode": feature_scaling_mode,
        "primary_connected_component_split": primary_connected_component_split,
        "secondary_detection_thresholds": {
            "direction_dispersion_threshold": float(direction_dispersion_threshold),
            "min_partition_size_for_secondary": int(min_partition_size_for_secondary),
            "secondary_k_max": int(secondary_k_max),
        },
        "postprocess": postprocess_summary,
        "partitions": diagnostics,
    }
    with diagnostics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def partition_coverage_matrix(
    xs,
    ys,
    coverage_matrix,
    depth_matrix=None,
    U=None,
    k_max=10,
    output_dir=None,
    boundary_mask=None,
    cell_effective_area=None,
    gx_matrix=None,
    gy_matrix=None,
    primary_feature_mode="coverage_only",
    feature_scaling_mode="legacy_balanced",
    enable_secondary_partition=True,
    secondary_feature_mode="direction_only",
    secondary_partition_policy="auto_by_direction_dispersion",
    secondary_k_max=6,
    min_partition_size_for_secondary=12,
    direction_dispersion_threshold=0.30,
    split_disconnected_primary_partitions=False,
    postprocess_merge_small_partitions=False,
    postprocess_small_partition_area_ratio_threshold=0.10,
    postprocess_merge_enclosed_partitions=False,
    enclosed_partition_boundary_ratio_threshold=0.50,
    enclosed_partition_source_area_ratio_min=None,
    enclosed_partition_source_area_ratio_max=None,
):
    """
    对覆盖次数矩阵进行 K-means 空间聚类并可视化边界

    参数:
        xs, ys: 坐标数组
        coverage_matrix: 覆盖次数矩阵
        depth_matrix: 深度矩阵（当一级特征包含 depth 时必需）
        U: 指定分区数量（可选）。若传入则直接使用，若为 None 则使用 Elbow 法则自动确定
        k_max: Elbow 法则搜索的最大簇数（仅在 U=None 时生效）
        output_dir: 输出目录路径。若传入则保存分区图到 output_dir/partition/，否则保存到默认路径
        boundary_mask: 有效海域掩码。若提供，则仅对有效格聚类，其余位置标记为 -1
    """
    feature_bundle = _build_feature_stacks(
        xs,
        ys,
        coverage_matrix,
        depth_matrix=depth_matrix,
        gx_matrix=gx_matrix,
        gy_matrix=gy_matrix,
        primary_feature_mode=primary_feature_mode,
        secondary_feature_mode=secondary_feature_mode,
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
    secondary_feature_names = feature_bundle["secondary_feature_names"]
    secondary_features = feature_bundle["secondary_features"]

    if secondary_partition_policy not in SECONDARY_PARTITION_POLICIES:
        raise ValueError(
            "不支持的二次分区策略："
            f"{secondary_partition_policy}，可选={SECONDARY_PARTITION_POLICIES}"
        )

    if boundary_mask is None:
        valid_mask_2d = np.ones((rows, cols), dtype=bool)
        valid_mask_1d = np.ones(rows * cols, dtype=bool)
    else:
        valid_mask_2d = np.asarray(boundary_mask, dtype=bool)
        if valid_mask_2d.shape != (rows, cols):
            raise ValueError(
                "boundary_mask 尺寸与 coverage_matrix 不一致："
                f"coverage={coverage_matrix.shape}, boundary_mask={valid_mask_2d.shape}"
            )
        valid_mask_1d = valid_mask_2d.reshape(-1)

    if cell_effective_area is not None:
        cell_effective_area = np.asarray(cell_effective_area, dtype=float)
        if cell_effective_area.shape != (rows, cols):
            raise ValueError(
                "cell_effective_area 尺寸与 coverage_matrix 不一致："
                f"coverage={coverage_matrix.shape}, cell_effective_area={cell_effective_area.shape}"
            )

    if not np.any(valid_mask_1d):
        raise ValueError("无有效海域网格可用于聚类，请检查 boundary_mask。")

    print(
        "[分区] 一级分区特征 = "
        f"{primary_feature_names} | 特征缩放 = {feature_scaling_mode} | "
        f"二级分区启用 = {enable_secondary_partition} | 二级分区策略 = {secondary_partition_policy} | "
        f"二级分区特征 = {secondary_feature_names}"
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
        feature_scaling_mode=feature_scaling_mode,
    )
    primary_split_summary = {
        "enabled": bool(split_disconnected_primary_partitions),
        "applied": False,
        "connectivity": "4-neighbor_shared_edge",
        "pre_split_u": int(first_stage_u),
        "post_split_u": int(first_stage_u),
        "split_original_partition_ids": [],
        "additional_partition_count": 0,
        "components_by_original_partition": [],
    }
    if split_disconnected_primary_partitions:
        first_stage_cluster_matrix, primary_split_summary = _split_disconnected_partitions(
            first_stage_cluster_matrix,
            valid_mask=valid_mask_2d,
        )
        first_stage_u = int(primary_split_summary["post_split_u"])
        print(
            "[一级分区连通性] "
            f"启用4邻接拆分 | 原始U={primary_split_summary['pre_split_u']} | "
            f"拆分后U={first_stage_u} | "
            f"新增分区数={primary_split_summary['additional_partition_count']}"
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

        diagnostic["auto_triggered_secondary_partition"] = bool(
            diagnostic["triggered_secondary_partition"]
        )
        diagnostic["auto_trigger_reasons"] = list(diagnostic["trigger_reasons"])
        if secondary_partition_policy == "all_partitions":
            diagnostic["triggered_secondary_partition"] = True
            diagnostic["trigger_reasons"] = ["all_partitions_policy"]

        print(
            f"[分区检测] 一级分区{partition_id} | cells={diagnostic['cell_count']} | "
            f"dispersion={diagnostic['direction_dispersion']:.3f} | "
            f"unit_like={diagnostic['unit_like_gradient']} | "
            f"自动触发={diagnostic['auto_triggered_secondary_partition']} | "
            f"实际二次分区={diagnostic['triggered_secondary_partition']} | "
            f"reasons={diagnostic['trigger_reasons']}"
        )

        if enable_secondary_partition and diagnostic["triggered_secondary_partition"]:
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
                    feature_scaling_mode="equal_weight_standardized",
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
        elif not enable_secondary_partition:
            diagnostic["would_trigger_secondary_partition"] = bool(
                diagnostic["triggered_secondary_partition"]
            )
            diagnostic["would_trigger_reasons"] = list(diagnostic["trigger_reasons"])
            diagnostic["triggered_secondary_partition"] = False
            diagnostic["trigger_reasons"] = ["disabled_by_configuration"]
            diagnostic["secondary_partition_count"] = 1
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
    final_cluster_matrix, postprocess_summary = _postprocess_partition_matrix(
        post_secondary_cluster_matrix,
        valid_mask_2d,
        cell_effective_area=cell_effective_area,
        merge_small_partitions=postprocess_merge_small_partitions,
        small_partition_area_ratio_threshold=postprocess_small_partition_area_ratio_threshold,
        merge_enclosed_partitions=postprocess_merge_enclosed_partitions,
        enclosed_partition_boundary_ratio_threshold=enclosed_partition_boundary_ratio_threshold,
        enclosed_partition_source_area_ratio_min=enclosed_partition_source_area_ratio_min,
        enclosed_partition_source_area_ratio_max=enclosed_partition_source_area_ratio_max,
    )
    final_cluster_matrix, final_u = _renumber_cluster_matrix(final_cluster_matrix)
    final_partition_title = (
        f"Two-Stage Partitioning (U={final_u})"
        if enable_secondary_partition
        else f"Single-Stage Partitioning (U={final_u})"
    )
    _save_partition_plot(
        final_cluster_matrix,
        xs,
        ys,
        output_dir,
        f"partition_final_U={final_u}.png",
        final_partition_title,
    )
    _save_partition_plot(
        final_cluster_matrix,
        xs,
        ys,
        output_dir,
        f"partition_U={final_u}.png",
        final_partition_title,
    )
    _write_secondary_partition_diagnostics(
        output_dir,
        first_stage_u,
        post_secondary_u,
        final_u,
        primary_feature_names,
        secondary_feature_names,
        primary_split_summary,
        diagnostics,
        direction_dispersion_threshold,
        min_partition_size_for_secondary,
        secondary_k_max,
        enable_secondary_partition,
        feature_scaling_mode,
        secondary_partition_policy,
        postprocess_summary,
    )

    if postprocess_summary["enabled"]:
        print(
            f"[分区后处理] 小面积合并={len(postprocess_summary['small_partition_merges'])} | "
            f"被包围合并={len(postprocess_summary['enclosed_partition_merges'])} | "
            f"后处理生效={postprocess_summary['postprocess_changed']}"
        )

    intermediate_label = (
        "二次分区后总分区数" if enable_secondary_partition else "单阶段后总分区数"
    )
    final_label = (
        "后处理后总分区数"
        if postprocess_summary["enabled"]
        else ("二次分区合并后总分区数" if enable_secondary_partition else "单阶段最终总分区数")
    )

    print(
        f"[分区] 一级分区数={first_stage_u} | "
        f"{intermediate_label}={post_secondary_u} | "
        f"{final_label}={final_u}"
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
