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
            "primary_feature_mode": "xy_depth_coverage",
            "feature_scaling_mode": "equal_weight_standardized",
            "enable_secondary_partition": True,
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

    X_MIN, X_MAX = 0, 4 * 1852
    Y_MIN, Y_MAX = 0, 5 * 1852

    print(
        f"[主流程] 当前实验组合 = {experiment_name} | {experiment_config['description']}"
    )

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
    output_base = f"./multibeam/output/{current_time}_{experiment_config['output_tag']}"

    # Phase 3: 分区（elbow 和 partition 图写入 output/<timestamp>/partition/）
    final_cluster_matrix, U = partition_coverage_matrix(
        xs,
        ys,
        coverage_matrix,
        depth_matrix=depth_matrix,
        output_dir=output_base,
        boundary_mask=coarse_boundary_mask,
        gx_matrix=gx_matrix,
        gy_matrix=gy_matrix,
        **experiment_config["partition_kwargs"],
    )

    print(f"U= {U}")

    from multibeam.Planner import SurveyPlanner

    print("[主流程] 当前轮次将继续执行测线规划。")
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
    )
    # planner.plan_line(1000, 1000, output_dir=output_base)
    planner.plan_all(output_dir=output_base)
