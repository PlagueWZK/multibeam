import datetime

import matplotlib

matplotlib.use("Agg")

from multibeam.GridCell import (
    calculate_optimal_mesh_size,
)
from multibeam.Coverage import calculate_coverage_matrix_with_ml
from multibeam.Partition import get_partition_for_point, partition_coverage_matrix

if __name__ == "__main__":
    BASE_DIR = __import__("pathlib").Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data" / "data.xlsx"

    X_MIN, X_MAX = 0, 4 * 1852
    Y_MIN, Y_MAX = 0, 5 * 1852

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
    PRIMARY_FEATURE_MODE = "xy_coverage_depth"
    PRIMARY_FEATURE_LABEL = "xycovdepth"
    SECONDARY_FEATURE_LABEL = "grad"
    START_POINT_STRATEGY = "geometric_center"
    START_POINT_LABEL = "geocenter"
    SECONDARY_DIRECTION_DISPERSION_MIN_THRESHOLD = 0.10
    SECONDARY_MIN_AREA_RATIO_FOR_SECONDARY = 0.05
    SECONDARY_THRESHOLD_LABEL = "sec-adaptive-min0p1-area5pct"
    CHILD_LINE_PARENT_GAIN_FACTOR = 0.80
    CHILD_LINE_MIN_GAIN_THRESHOLD = 0.30
    LINE_GAIN_RULE_LABEL = "gain-adaptive-parent80pct-floor30pct"
    REFERENCE_POINT = (1000.0, 5000.0)
    PLANNING_SCOPE_LABEL = "point-1000-5000_partition-auto"
    output_base = (
        f"./multibeam/output/{current_time}_covg_"
        f"{PRIMARY_FEATURE_LABEL}_{SECONDARY_FEATURE_LABEL}_{START_POINT_LABEL}_"
        f"{SECONDARY_THRESHOLD_LABEL}_{LINE_GAIN_RULE_LABEL}_"
        f"{PLANNING_SCOPE_LABEL}"
    )
    print(
        "[主流程] 当前配置 | "
        f"一级分区特征={PRIMARY_FEATURE_MODE} | "
        f"二级分区特征=gradient-direction | "
        "二次触发规则=adaptive-maxgap | "
        f"最低有效离散阈值={SECONDARY_DIRECTION_DISPERSION_MIN_THRESHOLD:.2f} | "
        f"候选最小面积占比={SECONDARY_MIN_AREA_RATIO_FOR_SECONDARY:.0%} | "
        f"起点策略={START_POINT_STRATEGY} | "
        "测线取舍规则=adaptive-parent-gain | "
        f"父收益折减={CHILD_LINE_PARENT_GAIN_FACTOR:.0%} | "
        f"最低阈值={CHILD_LINE_MIN_GAIN_THRESHOLD:.0%}"
    )
    print(f"[主流程] 输出目录: {output_base}")

    # Phase 3: 分区（elbow 和 partition 图写入 output/<timestamp>/partition/）
    final_cluster_matrix, U = partition_coverage_matrix(
        xs,
        ys,
        coverage_matrix,
        output_dir=output_base,
        boundary_mask=coarse_boundary_mask,
        gx_matrix=gx_matrix,
        gy_matrix=gy_matrix,
        depth_matrix=depth_matrix,
        primary_feature_mode=PRIMARY_FEATURE_MODE,
        direction_dispersion_threshold=SECONDARY_DIRECTION_DISPERSION_MIN_THRESHOLD,
        min_partition_area_ratio_for_secondary=SECONDARY_MIN_AREA_RATIO_FOR_SECONDARY,
        cell_effective_area=coarse_cell_effective_area,
    )

    print(f"U= {U}")

    reference_partition_id = get_partition_for_point(
        REFERENCE_POINT[0],
        REFERENCE_POINT[1],
        xs,
        ys,
        final_cluster_matrix,
        x_min=X_MIN,
        x_max=X_MAX,
        y_min=Y_MIN,
        y_max=Y_MAX,
    )
    print(
        "[主流程] 单点所在分区规划范围 | "
        f"参考点=({REFERENCE_POINT[0]:.1f}, {REFERENCE_POINT[1]:.1f}) | "
        f"参考点所属分区={reference_partition_id}"
    )

    from multibeam.Planner import SurveyPlanner

    print("[主流程] 当前轮次将仅执行参考点所在分区测线规划。")
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
        start_point_strategy=START_POINT_STRATEGY,
        child_line_parent_gain_factor=CHILD_LINE_PARENT_GAIN_FACTOR,
        child_line_min_gain_threshold=CHILD_LINE_MIN_GAIN_THRESHOLD,
    )
    planner.plan_line(REFERENCE_POINT[0], REFERENCE_POINT[1], output_dir=output_base)
    # planner.plan_all(output_dir=output_base)
