import matplotlib

matplotlib.use("Agg")

from multibeam.GridCell import *
from multibeam.Coverage import *
from multibeam.Planner import *

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data" / "data.xlsx"
    optimal_d = calculate_optimal_mesh_size(DATA_DIR)

    X_MIN, X_MAX = 0, 5 * 1852
    Y_MIN, Y_MAX = 0, 4 * 1852

    xs, ys, depth_matrix, coverage_matrix = calculate_coverage_matrix_with_ml(
        x_min=X_MIN,
        x_max=X_MAX,
        y_min=Y_MIN,
        y_max=Y_MAX,
        d=optimal_d,
    )

    final_cluster_matrix, U = partition_coverage_matrix(xs, ys, coverage_matrix)

    # print(final_cluster_matrix)
    print(f"xs:{xs}\nys:{ys}")
    # TOTAL_AREA = (X_MAX - X_MIN) * (Y_MAX - Y_MIN)  # 5*1852 * 4*1852 = 68598080
    #
    # plan_all_line(xs, ys, final_cluster_matrix, total_area=TOTAL_AREA)

