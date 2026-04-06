import matplotlib

matplotlib.use("Agg")

from multibeam.GridCell import (
    calculate_optimal_mesh_size,
)
from multibeam.Coverage import calculate_coverage_matrix_with_ml
from multibeam.Planner import SurveyPlanner
from multibeam.Partition import partition_coverage_matrix

if __name__ == "__main__":
    BASE_DIR = __import__("pathlib").Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data" / "data.xlsx"

    X_MIN, X_MAX = 0, 5 * 1852
    Y_MIN, Y_MAX = 0, 4 * 1852

    # Phase 1: 最优网格（arange 策略，d 保持原始值不变）
    d_optimal = calculate_optimal_mesh_size(DATA_DIR)

    # Phase 2: 覆盖矩阵（内部使用 arange 生成坐标 + 边界掩码过滤）
    xs, ys, depth_matrix, coverage_matrix = calculate_coverage_matrix_with_ml(
        x_min=X_MIN,
        x_max=X_MAX,
        y_min=Y_MIN,
        y_max=Y_MAX,
        d=d_optimal,
    )
    print(f"xs: {xs}\nys: {ys}")

    # Phase 3: 分区
    final_cluster_matrix, U = partition_coverage_matrix(xs, ys, coverage_matrix)

    # Phase 4: 测线规划（封装对象，即时指标）
    planner = SurveyPlanner(
        xs,
        ys,
        final_cluster_matrix,
        x_min=X_MIN,
        x_max=X_MAX,
        y_min=Y_MIN,
        y_max=Y_MAX,
    )
    planner.plan_line(output_dir="./multibeam/output")
    planner.save_metrics_excel("./multibeam/output")

