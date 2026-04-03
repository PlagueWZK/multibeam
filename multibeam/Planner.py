import copy
import time

from multibeam.Partition import *
from tool.Data import *
import datetime


class SurveyPlanner:
    def __init__(self, xs, ys, cluster_matrix, theta=120, n=0.1, min_depth=20):
        self.xs = xs
        self.ys = ys
        self.cluster_matrix = cluster_matrix
        self.theta = theta
        self.theta_rad = np.radians(theta)
        self.n = n

        self.x_min = xs.min()
        self.x_max = xs.max()
        self.y_min = ys.min()
        self.y_max = ys.max()

        self.d_mask = self._calculate_mask_grid_size(min_depth)
        self.mask_cols = int(np.ceil((self.x_max - self.x_min) / self.d_mask)) + 1
        self.mask_rows = int(np.ceil((self.y_max - self.y_min) / self.d_mask)) + 1
        self.global_coverage_mask = np.zeros((self.mask_rows, self.mask_cols), dtype=bool)

        print(f"[掩码网格] 边长 d_mask={self.d_mask:.1f}m | 维度 {self.mask_rows}×{self.mask_cols} | 总网格 {self.mask_rows * self.mask_cols}")

    def _calculate_mask_grid_size(self, min_depth):
        w_min = 2 * min_depth * np.tan(self.theta_rad / 2)
        d_mask = w_min / 4
        print(f"[掩码网格计算] 最浅水深={min_depth}m | 最小覆盖宽度={w_min:.1f}m | 掩码网格边长={d_mask:.1f}m")
        return d_mask

    def _to_mask_idx(self, x, y):
        c_idx = int((x - self.x_min) / self.d_mask)
        r_idx = int((y - self.y_min) / self.d_mask)
        c_idx = max(0, min(c_idx, self.mask_cols - 1))
        r_idx = max(0, min(r_idx, self.mask_rows - 1))
        return r_idx, c_idx

    def _check_mask_collision(self, x, y):
        r_idx, c_idx = self._to_mask_idx(x, y)
        return self.global_coverage_mask[r_idx, c_idx]

    def _paint_point_to_mask(self, x, y):
        w_left = get_w_left(x, y, self.theta)
        w_right = get_w_right(x, y, self.theta)
        r_cover = max(w_left, w_right)
        r_cells = int(np.ceil(r_cover / self.d_mask))

        r_idx, c_idx = self._to_mask_idx(x, y)

        min_r = max(0, r_idx - r_cells)
        max_r = min(self.mask_rows - 1, r_idx + r_cells)
        min_c = max(0, c_idx - r_cells)
        max_c = min(self.mask_cols - 1, c_idx + r_cells)

        self.global_coverage_mask[min_r:max_r + 1, min_c:max_c + 1] = True

    def paint_line_to_mask(self, line):
        if isinstance(line, np.ndarray):
            for pt in line:
                self._paint_point_to_mask(pt[0], pt[1])
        else:
            for pt in line:
                self._paint_point_to_mask(pt[0], pt[1])

    def count_covered_cells(self):
        return np.sum(self.global_coverage_mask)

    def plan_line(self, start_x, start_y, step=None):
        t0 = time.time()
        save_interval = 10
        last_save_time = t0
        snap_counter = 0
        max_perp_iters = 500

        if step is None:
            step = self.d_mask

        print(f"[测线规划] 开始规划 | 起点: ({start_x}, {start_y}) | 步长: {step:.1f} | 重叠率: {self.n} | 开角: {self.theta}°")

        plt.figure(figsize=(12, 10))
        plt.xlim(self.x_min, self.x_max)
        plt.ylim(self.y_min, self.y_max)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title("Survey Line Planning")
        plt.grid(True, alpha=0.3)

        line = [[start_x, start_y]]
        target_partition_id = get_partition_for_point(
            start_x, start_y, self.xs, self.ys, self.cluster_matrix
        )
        print(f"[测线规划] 目标分区ID: {target_partition_id}")

        loop_count = 0
        curr_x, curr_y = start_x, start_y
        while True:
            loop_count += 1
            gx = get_gx(curr_x, curr_y)
            gy = get_gy(curr_x, curr_y)
            dx, dy = forward_direction(gx, gy)
            curr_x += step * dx
            curr_y += step * dy

            is_in, _ = is_point_in_partition(
                curr_x, curr_y, target_partition_id, self.xs, self.ys, self.cluster_matrix
            )
            if not is_in:
                print(f"[主测线1] 第{loop_count}步: ({curr_x:.1f}, {curr_y:.1f}) 超出边界，终止")
                break

            if self._check_mask_collision(curr_x, curr_y):
                print(f"[主测线1] 第{loop_count}步: ({curr_x:.1f}, {curr_y:.1f}) 撞上已覆盖区域，终止")
                break

            dist_to_start = np.sqrt((curr_x - line[0][0]) ** 2 + (curr_y - line[0][1]) ** 2)
            if len(line) >= 3 and dist_to_start < 1.2 * step:
                print(f"[主测线1] 第{loop_count}步: ({curr_x:.1f}, {curr_y:.1f}) 检测到环形(距起点{dist_to_start:.1f})，终止")
                break

            line.append([curr_x, curr_y])

        line = np.array(line)
        self.paint_line_to_mask(line)
        covered_cells = self.count_covered_cells()
        print(f"[主测线1] 完成 | 共{len(line)}个点 | 已覆盖{covered_cells}个掩码网格 | 起点({line[0][0]:.1f},{line[0][1]:.1f}) 终点({line[-1][0]:.1f},{line[-1][1]:.1f})")

        plt.plot(line[:, 0], line[:, 1], color="b", linewidth=1.5, label="Main line")

        perp_iter = 0
        total_points = len(line)
        skipped_by_collision = 0

        while True:
            perp_iter += 1
            if perp_iter > max_perp_iters:
                print(f"[垂直测线] 达到最大轮数限制({max_perp_iters})，强制终止")
                break

            t1 = []
            for index, i in enumerate(line):
                x = i[0]
                y = i[1]
                alpha = get_alpha(x, y)
                h = get_height(x, y)
                if alpha <= 0.005:
                    d = 2 * h * np.tan(self.theta_rad / 2) * (1 - self.n)
                    tx = d * get_gx(x, y)
                    ty = d * get_gy(x, y)
                    new_x = x - tx
                    new_y = y - ty
                else:
                    A = np.sin(np.radians(90) - self.theta_rad / 2 + np.radians(alpha))
                    B = np.sin(np.radians(90) - self.theta_rad / 2 - np.radians(alpha))
                    C = np.sin(self.theta_rad / 2) / A - 1 / np.sin(np.radians(alpha))
                    D = (
                            self.n * np.sin(self.theta_rad / 2) * (1 / A + 1 / B)
                            - np.sin(self.theta_rad / 2) / B
                            - 1 / np.sin(np.radians(alpha))
                    )
                    next_h = h * C / D
                    tx = (h - next_h) / np.tan(np.radians(alpha)) * get_gx(x, y)
                    ty = (h - next_h) / np.tan(np.radians(alpha)) * get_gy(x, y)
                    new_x = x - tx
                    new_y = y - ty

                is_in, _ = is_point_in_partition(
                    new_x, new_y, target_partition_id, self.xs, self.ys, self.cluster_matrix
                )
                if is_in:
                    t1.append([new_x, new_y])

            if len(line) > 5:
                plt.plot(line[:, 0], line[:, 1], color="lightgray", linewidth=0.6)

            elapsed = time.time() - t0
            covered_cells = self.count_covered_cells()
            print(f"[垂直测线 第{perp_iter}轮] 当前{len(line)}点 -> 新增{len(t1)}有效(跳过{skipped_by_collision}已覆盖) | 已耗时{elapsed:.1f}s | 总点数{total_points} | 已覆盖{covered_cells}网格")

            if len(t1) == 0:
                print(f"[垂直测线] 第{perp_iter}轮无新有效点，结束规划")
                break

            if time.time() - last_save_time >= save_interval:
                snap_counter += 1
                snap_path = f"./multibeam/output/lines/snap_{snap_counter:03d}_iter{perp_iter}.png"
                plt.savefig(snap_path, dpi=200, bbox_inches="tight")
                last_save_time = time.time()
                print(f"  >> 已保存快照: {snap_path}")

            self.paint_line_to_mask(t1)

            line = copy.deepcopy(t1)
            loc_x = line[-1][0]
            loc_y = line[-1][1]
            ext_step = self.d_mask
            ext_loop = 0

            while True:
                ext_loop += 1
                gx = get_gx(loc_x, loc_y)
                gy = get_gy(loc_x, loc_y)
                dx, dy = forward_direction(gx, gy)
                loc_x += ext_step * dx
                loc_y += ext_step * dy

                is_in, _ = is_point_in_partition(
                    loc_x, loc_y, target_partition_id, self.xs, self.ys, self.cluster_matrix
                )
                if not is_in:
                    print(f"  [测线延伸] 延伸{ext_loop}步后超出边界")
                    break

                if self._check_mask_collision(loc_x, loc_y):
                    print(f"  [测线延伸] 延伸{ext_loop}步后撞上已覆盖区域")
                    break

                dist_to_start = np.sqrt((loc_x - line[0][0]) ** 2 + (loc_y - line[0][1]) ** 2)
                if len(line) >= 3 and dist_to_start < 1.2 * ext_step:
                    print(f"  [测线延伸] 延伸{ext_loop}步后检测到环形(距起点{dist_to_start:.1f})")
                    break

                self._paint_point_to_mask(loc_x, loc_y)
                line.append([loc_x, loc_y])

            line = np.array(line)
            total_points += len(line)

        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = f"./multibeam/output/lines/plan_line_{current_time}.png"
        plt.legend()
        plt.savefig(final_path, dpi=300, bbox_inches="tight")
        plt.close()

        total_elapsed = time.time() - t0
        final_covered = self.count_covered_cells()
        print(f"\n[测线规划完成] 总耗时: {total_elapsed:.1f}s | 共{perp_iter}轮垂直测线 | 总点数约{total_points}")
        print(f"  最终图片: {final_path} | 已覆盖掩码网格: {final_covered}/{self.mask_rows * self.mask_cols} | 累计跳过重复点: {skipped_by_collision}")
        if snap_counter > 0:
            print(f"  中间快照: {snap_counter}张 (每{save_interval}s一张)")


def plan_line(start_x, start_y, xs, ys, cluster_matrix, n=0.1, step=50, theta=120, min_depth=20):
    planner = SurveyPlanner(xs, ys, cluster_matrix, theta=theta, n=n, min_depth=min_depth)
    planner.plan_line(start_x, start_y, step=step)
