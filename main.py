import datetime
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from multibeam.GridCell import (
    calculate_optimal_mesh_size,
)
from multibeam.Coverage import calculate_coverage_matrix_with_ml
from multibeam.Partition import partition_coverage_matrix


GEOMETRIC_CENTER_START_TAG = "geometric-center-start"
DEEPEST_DEPTH_CELL_START_TAG = "deepest-depth-cell-center"
MEDIAN_DEPTH_CELL_START_TAG = "median-depth-cell-center"
GEOMETRIC_CENTER_GRID_CELL_START_TAG = "geometric-center-grid-cell"

COMBO1_PARTITION_KWARGS = {
    "primary_feature_mode": "xy_depth_coverage",
    "feature_scaling_mode": "equal_weight_standardized",
    "enable_secondary_partition": True,
    "secondary_feature_mode": "direction_only",
    "secondary_partition_policy": "auto_by_direction_dispersion",
}

REQUIREMENT_COMBO1_PARTITION_KWARGS = {
    **COMBO1_PARTITION_KWARGS,
    "direction_dispersion_threshold": 0.20,
}


PARTITION_EXPERIMENTS = {
    "baseline_cov_g": {
        "description": "当前分支默认基线：coverage_count 一级分区 + 方向二次分区",
        "output_tag": "cov-g",
        "partition_kwargs": {
            "primary_feature_mode": "coverage_only",
            "feature_scaling_mode": "legacy_balanced",
            "enable_secondary_partition": True,
        },
    },
    "combo1_xy_depth_coverage_stage2_direction": {
        "description": "一级分区使用 X,Y,depth,coverage；触发后二级分区使用 ux,uy",
        "output_tag": "xy-depth-coverage_stage2-direction",
        "partition_kwargs": {
            **COMBO1_PARTITION_KWARGS,
        },
    },
    "combo2_xy_depth_coverage_direction_single_stage": {
        "description": "单阶段分区使用 X,Y,depth,coverage,ux,uy，按等权标准化，不执行二次分区",
        "output_tag": "xy-depth-coverage-direction_single-stage",
        "partition_kwargs": {
            "primary_feature_mode": "xy_depth_coverage_direction",
            "feature_scaling_mode": "equal_weight_standardized",
            "enable_secondary_partition": False,
        },
    },
    "combo3_coverage_direction_single_stage": {
        "description": "单阶段分区使用 coverage,ux,uy，不执行二次分区",
        "output_tag": "coverage-direction_single-stage",
        "partition_kwargs": {
            "primary_feature_mode": "coverage_direction",
            "feature_scaling_mode": "equal_weight_standardized",
            "enable_secondary_partition": False,
        },
    },
    "requirement_combo1_xy_depth_coverage_then_direction": {
        "description": "需求组合1：一级 X,Y,depth,coverage；触发后二级 direction-only",
        "output_tag": "requirement-combo1_xy-depth-coverage_then-direction",
        "partition_kwargs": {
            **REQUIREMENT_COMBO1_PARTITION_KWARGS,
        },
    },
    "requirement_combo2_direction_then_depth_coverage": {
        "description": "需求组合2：一级 direction-only；每个一级分区二级 depth+coverage",
        "output_tag": "requirement-combo2_direction-only_then-depth-coverage",
        "partition_kwargs": {
            "primary_feature_mode": "direction_only",
            "feature_scaling_mode": "equal_weight_standardized",
            "enable_secondary_partition": True,
            "secondary_feature_mode": "depth_coverage",
            "secondary_partition_policy": "all_partitions",
            "min_partition_size_for_secondary": 0,
            "split_disconnected_primary_partitions": True,
            "postprocess_merge_small_partitions": True,
            "postprocess_small_partition_area_ratio_threshold": 0.03,
            "postprocess_merge_enclosed_partitions": True,
            "enclosed_partition_boundary_ratio_threshold": 0.50,
            "enclosed_partition_source_area_ratio_min": 0.03,
            "enclosed_partition_source_area_ratio_max": 0.05,
        },
    },
    "combo1_lineplan_deepest_depth_cell": {
        "description": "combo1 分区 + 最深点网格中心起始布线",
        "output_tag": "requirement-combo1_xy-depth-coverage_then-direction_line-planning",
        "start_point_tag": DEEPEST_DEPTH_CELL_START_TAG,
        "run_planner": True,
        "planner_kwargs": {
            "partition_start_point_strategy": "deepest_depth_cell",
        },
        "partition_kwargs": {
            **COMBO1_PARTITION_KWARGS,
        },
    },
    "combo1_lineplan_median_depth_cell": {
        "description": "combo1 分区 + 深度中位数网格中心起始布线",
        "output_tag": "requirement-combo1_xy-depth-coverage_then-direction_line-planning",
        "start_point_tag": MEDIAN_DEPTH_CELL_START_TAG,
        "run_planner": True,
        "planner_kwargs": {
            "partition_start_point_strategy": "median_depth_cell",
        },
        "partition_kwargs": {
            **COMBO1_PARTITION_KWARGS,
        },
    },
    "combo1_lineplan_geometric_center_cell": {
        "description": "combo1 分区 + 分区几何中心网格起始布线",
        "output_tag": "requirement-combo1_xy-depth-coverage_then-direction_line-planning",
        "start_point_tag": GEOMETRIC_CENTER_GRID_CELL_START_TAG,
        "run_planner": True,
        "planner_kwargs": {
            "partition_start_point_strategy": "geometric_center_cell",
        },
        "partition_kwargs": {
            **COMBO1_PARTITION_KWARGS,
        },
    },
}


def resolve_partition_experiment():
    selected_name = os.environ.get("MB_PARTITION_EXPERIMENT", "baseline_cov_g")
    if selected_name not in PARTITION_EXPERIMENTS:
        available = ", ".join(sorted(PARTITION_EXPERIMENTS))
        raise ValueError(
            f"不支持的 MB_PARTITION_EXPERIMENT={selected_name}，可选: {available}"
        )
    return selected_name, PARTITION_EXPERIMENTS[selected_name]

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data" / "data.xlsx"
    experiment_name, experiment_config = resolve_partition_experiment()
    start_point_tag = experiment_config.get("start_point_tag")
    run_planner = bool(experiment_config.get("run_planner", False))
    planner_kwargs = experiment_config.get("planner_kwargs", {})

    X_MIN, X_MAX = 0, 4 * 1852
    Y_MIN, Y_MAX = 0, 5 * 1852

    print(
        f"[主流程] 当前实验组合 = {experiment_name} | {experiment_config['description']}"
    )
    if start_point_tag is not None:
        if run_planner:
            print(f"[主流程] 起始点标签 = {start_point_tag}（本轮用于输出标识与布线起点策略）")
        else:
            print(f"[主流程] 起始点标签 = {start_point_tag}（本轮仅用于输出标识）")

    # Phase 1: 最优网格（arange 策略，d 保持原始值不变）
    d_optimal = calculate_optimal_mesh_size(DATA_DIR)

    # Phase 2: 覆盖矩阵（内部使用 arange 生成坐标 + 边界掩码过滤）
    (
        xs,
        ys,
        depth_matrix,
        coverage_matrix,
        coarse_boundary_mask,
        coarse_cell_effective_area,
        coarse_cell_area_ratio,
        gx_matrix,
        gy_matrix,
    ) = calculate_coverage_matrix_with_ml(
        x_min=X_MIN,
        x_max=X_MAX,
        y_min=Y_MIN,
        y_max=Y_MAX,
        d=d_optimal,
    )


    # 统一时间戳，确保所有输出在同一目录下
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_suffix = experiment_config["output_tag"]
    if start_point_tag is not None:
        output_suffix = f"{output_suffix}_{start_point_tag}"
    output_base = f"./multibeam/output/{current_time}_{output_suffix}"

    # Phase 3: 分区（elbow 和 partition 图写入 output/<timestamp>/partition/）
    final_cluster_matrix, U = partition_coverage_matrix(
        xs,
        ys,
        coverage_matrix,
        depth_matrix=depth_matrix,
        output_dir=output_base,
        boundary_mask=coarse_boundary_mask,
        cell_effective_area=coarse_cell_effective_area,
        gx_matrix=gx_matrix,
        gy_matrix=gy_matrix,
        **experiment_config["partition_kwargs"],
    )

    print(f"U= {U}")
    if not run_planner:
        print("[主流程] 当前实验配置只执行分区，跳过测线规划。")
    else:
        from multibeam.Planner import SurveyPlanner

        print(
            f"[主流程] 当前轮次将继续执行测线规划 | 起点策略 = {planner_kwargs.get('partition_start_point_strategy', 'median_depth_cell')}"
        )
        planner = SurveyPlanner(
            xs,
            ys,
            final_cluster_matrix,
            depth_matrix=depth_matrix,
            x_min=X_MIN,
            x_max=X_MAX,
            y_min=Y_MIN,
            y_max=Y_MAX,
            boundary_mask=coarse_boundary_mask,
            cell_effective_area=coarse_cell_effective_area,
            grid_cell_size=d_optimal,
            **planner_kwargs,
        )
        planner.plan_all(output_dir=output_base)
