import copy
import signal
import time
from pathlib import Path

from multibeam.Partition import *
from tool.Data import *
import datetime

_output_dir = None
_lines_dir = None
_dot_dir = None
_all_lines_ref = None


def _save_dot_csv(all_lines, dot_dir):
    dot_dir.mkdir(parents=True, exist_ok=True)
    dot_path = dot_dir / "dot.csv"
    with open(dot_path, "w", encoding="utf-8") as f:
        f.write("line_id,x,y\n")
        for line_id, line in enumerate(all_lines):
            for pt in line:
                f.write(f"{line_id},{pt[0]:.2f},{pt[1]:.2f}\n")
    return dot_path


def _signal_handler(signum, frame):
    global _lines_dir, _dot_dir, _all_lines_ref
    print("\n\n[中断] 检测到Ctrl+C，正在保存当前数据...")
    if _all_lines_ref is not None and len(_all_lines_ref) > 0:
        dot_path = _save_dot_csv(_all_lines_ref, _dot_dir)
        print(f"[中断] 已保存测线点文件: {dot_path}")
        print(f"[中断] 共保存{len(_all_lines_ref)}条测线")
    else:
        print("[中断] 没有测线数据需要保存")
    exit(0)


signal.signal(signal.SIGINT, _signal_handler)


def _point_to_segment_distance(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


def _check_line_intersection(point, line, threshold):
    if line is None or len(line) < 2:
        return False
    px, py = point[0], point[1]
    for i in range(len(line) - 1):
        x1, y1 = line[i][0], line[i][1]
        x2, y2 = line[i + 1][0], line[i + 1][1]
        dist = _point_to_segment_distance(px, py, x1, y1, x2, y2)
        if dist < threshold:
            return True
    return False


class SurveyPlanner:
    def __init__(self, xs, ys, cluster_matrix, theta=120, n=0.1):
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

        self.all_lines = []

    def _generate_perpendicular_lines(self, main_line, direction, target_partition_id, step, t0, save_interval, last_save_time, snap_counter):
        line = copy.deepcopy(main_line)
        prev_lines = [main_line.copy()]
        in_convergence_state = False
        parent_line_length = figure_length(line)
        perp_iter = 0
        total_points = 0
        intersection_terminated = False

        direction_name = "正向" if direction == 1 else "反向"
        print(f"\n[{direction_name}扩展] 开始从主测线扩展...")

        while True:
            perp_iter += 1

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
                    new_x = x - direction * tx
                    new_y = y - direction * ty
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
                    new_x = x - direction * tx
                    new_y = y - direction * ty

                is_in, _ = is_point_in_partition(
                    new_x, new_y, target_partition_id, self.xs, self.ys, self.cluster_matrix
                )
                if is_in:
                    t1.append([new_x, new_y])

            if len(line) > 5:
                plt.plot(line[:, 0], line[:, 1], color="lightgray", linewidth=0.6)

            elapsed = time.time() - t0
            print(f"[{direction_name}扩展 第{perp_iter}轮] 当前{len(line)}点 -> 新增{len(t1)}有效点 | 已耗时{elapsed:.1f}s")

            if len(t1) == 0:
                print(f"[{direction_name}扩展] 第{perp_iter}轮无新有效点，结束规划")
                break

            if len(t1) < 5:
                print(f"[{direction_name}扩展] 第{perp_iter}轮仅{len(t1)}个新点(<5)，测线退化，结束规划")
                break

            if time.time() - last_save_time >= save_interval:
                snap_counter += 1
                snap_path = _lines_dir / f"snap_{snap_counter:03d}_iter{perp_iter}.png"
                plt.savefig(snap_path, dpi=200, bbox_inches="tight")
                last_save_time = time.time()
                print(f"  >> 已保存快照: {snap_path}")

            line = copy.deepcopy(t1)
            ring_closed_saturated = False
            ring_closed_this_round = False

            if in_convergence_state:
                current_line_length = figure_length(line)
                print(f"  [收敛状态] 当前测线长度: {current_line_length:.1f}m, 父测线长度: {parent_line_length:.1f}m")

                line_arr = np.array(line)
                centroid = np.mean(line_arr, axis=0)
                dists_to_centroid = np.sqrt((line_arr[:, 0] - centroid[0])**2 + (line_arr[:, 1] - centroid[1])**2)
                nearest_idx = np.argmin(dists_to_centroid)
                nearest_point = line_arr[nearest_idx]
                min_dist_to_centroid = dists_to_centroid[nearest_idx]

                nearest_gx = get_gx(nearest_point[0], nearest_point[1])
                nearest_gy = get_gy(nearest_point[0], nearest_point[1])
                grad_dir = np.array([nearest_gx, nearest_gy])
                grad_norm = np.linalg.norm(grad_dir)
                if grad_norm > 1e-6:
                    grad_dir = grad_dir / grad_norm
                else:
                    grad_dir = np.array([0, 1])

                to_centroid = np.array([centroid[0] - nearest_point[0], centroid[1] - nearest_point[1]])
                to_centroid_norm = np.linalg.norm(to_centroid)
                if to_centroid_norm > 1e-6:
                    to_centroid = to_centroid / to_centroid_norm

                dot_product = np.dot(grad_dir, to_centroid)
                if dot_product > 0:
                    inner_width = get_w_right(nearest_point[0], nearest_point[1], self.theta)
                else:
                    inner_width = get_w_left(nearest_point[0], nearest_point[1], self.theta)

                if min_dist_to_centroid < inner_width:
                    print(f"  [测线饱和检测] 最近点距质心{min_dist_to_centroid:.1f}m < 内侧覆盖宽度{inner_width:.1f}m，测线饱和")
                    ring_closed_saturated = True
                else:
                    print(f"  [测线饱和检测] 最近点距质心{min_dist_to_centroid:.1f}m >= 内侧覆盖宽度{inner_width:.1f}m，继续生成")
            else:
                ext_step = step / 2
                ext_loop = 0
                intersection_threshold = step * 0.5

                front_x, front_y = line[-1][0], line[-1][1]
                back_x, back_y = line[0][0], line[0][1]
                front_stopped = False
                back_stopped = False
                extend_front = True

                while True:
                    if front_stopped and back_stopped:
                        break

                    if extend_front and front_stopped:
                        extend_front = False
                        continue
                    if not extend_front and back_stopped:
                        extend_front = True
                        continue

                    ext_loop += 1

                    if extend_front:
                        gx = get_gx(front_x, front_y)
                        gy = get_gy(front_x, front_y)
                        dx, dy = forward_direction(gx, gy)
                        new_x = front_x + ext_step * dx
                        new_y = front_y + ext_step * dy
                    else:
                        gx = get_gx(back_x, back_y)
                        gy = get_gy(back_x, back_y)
                        dx, dy = forward_direction(gx, gy)
                        new_x = back_x - ext_step * dx
                        new_y = back_y - ext_step * dy

                    is_in, _ = is_point_in_partition(
                        new_x, new_y, target_partition_id, self.xs, self.ys, self.cluster_matrix
                    )
                    if not is_in:
                        if extend_front:
                            print(f"  [测线延伸-前端] 延伸{ext_loop}步后超出边界")
                            front_stopped = True
                        else:
                            print(f"  [测线延伸-后端] 延伸{ext_loop}步后超出边界")
                            back_stopped = True
                        extend_front = not extend_front
                        continue

                    new_point = [new_x, new_y]
                    found_intersection = False
                    for prev_idx, prev_line in enumerate(prev_lines):
                        if _check_line_intersection(new_point, prev_line, intersection_threshold):
                            if extend_front:
                                print(f"  [测线延伸-前端] 延伸{ext_loop}步后与第{len(prev_lines)-prev_idx}条测线相交，迷路终止")
                            else:
                                print(f"  [测线延伸-后端] 延伸{ext_loop}步后与第{len(prev_lines)-prev_idx}条测线相交，迷路终止")
                            found_intersection = True
                            intersection_terminated = True
                            break

                    if found_intersection:
                        if extend_front:
                            front_stopped = True
                        else:
                            back_stopped = True
                        extend_front = not extend_front
                        continue

                    if len(line) >= 3:
                        if extend_front:
                            end_dist = np.sqrt((new_x - line[0][0])**2 + (new_y - line[0][1])**2)
                        else:
                            end_dist = np.sqrt((new_x - line[-1][0])**2 + (new_y - line[-1][1])**2)
                        if end_dist < step * 1.5:
                            if extend_front:
                                print(f"  [测线延伸-前端] 延伸{ext_loop}步后与后端点距离{end_dist:.1f}m，环形闭合")
                            else:
                                print(f"  [测线延伸-后端] 延伸{ext_loop}步后与前端点距离{end_dist:.1f}m，环形闭合")
                            if extend_front:
                                line.append(new_point)
                                front_x, front_y = new_x, new_y
                            else:
                                line.insert(0, new_point)
                                back_x, back_y = new_x, new_y

                            ring_closed_this_round = True
                            current_line_length = figure_length(line)
                            print(f"  [环形终止] 当前测线长度: {current_line_length:.1f}m, 父测线长度: {parent_line_length:.1f}m")

                            if current_line_length < parent_line_length:
                                print(f"  [环形测线收敛] 当前长度 < 父测线长度，进入收敛状态")
                                in_convergence_state = True

                                line_arr = np.array(line)
                                centroid = np.mean(line_arr, axis=0)
                                dists_to_centroid = np.sqrt((line_arr[:, 0] - centroid[0])**2 + (line_arr[:, 1] - centroid[1])**2)
                                nearest_idx = np.argmin(dists_to_centroid)
                                nearest_point = line_arr[nearest_idx]
                                min_dist_to_centroid = dists_to_centroid[nearest_idx]

                                nearest_gx = get_gx(nearest_point[0], nearest_point[1])
                                nearest_gy = get_gy(nearest_point[0], nearest_point[1])
                                grad_dir = np.array([nearest_gx, nearest_gy])
                                grad_norm = np.linalg.norm(grad_dir)
                                if grad_norm > 1e-6:
                                    grad_dir = grad_dir / grad_norm
                                else:
                                    grad_dir = np.array([0, 1])

                                to_centroid = np.array([centroid[0] - nearest_point[0], centroid[1] - nearest_point[1]])
                                to_centroid_norm = np.linalg.norm(to_centroid)
                                if to_centroid_norm > 1e-6:
                                    to_centroid = to_centroid / to_centroid_norm

                                dot_product = np.dot(grad_dir, to_centroid)
                                if dot_product > 0:
                                    inner_width = get_w_right(nearest_point[0], nearest_point[1], self.theta)
                                else:
                                    inner_width = get_w_left(nearest_point[0], nearest_point[1], self.theta)

                                if min_dist_to_centroid < inner_width:
                                    print(f"  [测线饱和检测] 最近点距质心{min_dist_to_centroid:.1f}m < 内侧覆盖宽度{inner_width:.1f}m，测线饱和")
                                    ring_closed_saturated = True
                                else:
                                    print(f"  [测线饱和检测] 最近点距质心{min_dist_to_centroid:.1f}m >= 内侧覆盖宽度{inner_width:.1f}m，继续生成")
                            else:
                                print(f"  [非收敛状态] 当前长度 >= 父测线长度，不进入收敛状态")

                            break

                    if extend_front:
                        line.append(new_point)
                        front_x, front_y = new_x, new_y
                    else:
                        line.insert(0, new_point)
                        back_x, back_y = new_x, new_y

                    extend_front = not extend_front

            line = np.array(line)
            self.all_lines.append(line.copy())
            prev_lines.append(line.copy())
            if len(prev_lines) > 3:
                prev_lines.pop(0)
            total_points += len(line)

            if ring_closed_this_round:
                plt.plot(line[:, 0], line[:, 1], color="red", linewidth=1.5)
            elif intersection_terminated:
                plt.plot(line[:, 0], line[:, 1], color="purple", linewidth=1.5)

            parent_line_length = figure_length(line)

            if ring_closed_saturated:
                print(f"[{direction_name}扩展] 测线饱和，终止该方向扩展")
                break

        print(f"[{direction_name}扩展] 完成 | 共{perp_iter}轮 | {total_points}点")
        return total_points, snap_counter, last_save_time

    def plan_line(self, start_x, start_y, step=50):
        global _output_dir, _lines_dir, _dot_dir, _all_lines_ref

        t0 = time.time()
        save_interval = 10
        last_save_time = t0
        snap_counter = 0

        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        _output_dir = Path("./multibeam/output")
        _lines_dir = _output_dir / "lines" / current_time
        _dot_dir = _output_dir / "dot" / current_time
        _lines_dir.mkdir(parents=True, exist_ok=True)
        _dot_dir.mkdir(parents=True, exist_ok=True)
        _all_lines_ref = self.all_lines

        print(f"[测线规划] 开始规划 | 起点: ({start_x}, {start_y}) | 步长: {step:.1f} | 重叠率: {self.n} | 开角: {self.theta}°")
        print(f"[测线规划] 输出目录: lines/{current_time}, dot/{current_time}")

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

            dist_to_start = np.sqrt((curr_x - line[0][0]) ** 2 + (curr_y - line[0][1]) ** 2)
            if len(line) >= 3 and dist_to_start < 1.2 * step:
                print(f"[主测线1] 第{loop_count}步: ({curr_x:.1f}, {curr_y:.1f}) 检测到环形(距起点{dist_to_start:.1f})，终止")
                break

            line.append([curr_x, curr_y])

        line = np.array(line)
        self.all_lines.append(line.copy())
        print(f"[主测线1] 完成 | 共{len(line)}个点 | 起点({line[0][0]:.1f},{line[0][1]:.1f}) 终点({line[-1][0]:.1f},{line[-1][1]:.1f})")

        plt.plot(line[:, 0], line[:, 1], color="b", linewidth=1.5, label="Main line")

        total_points1, snap_counter, last_save_time = self._generate_perpendicular_lines(
            line, direction=1, target_partition_id=target_partition_id, step=step,
            t0=t0, save_interval=save_interval, last_save_time=last_save_time, snap_counter=snap_counter
        )

        total_points2, snap_counter, last_save_time = self._generate_perpendicular_lines(
            line, direction=-1, target_partition_id=target_partition_id, step=step,
            t0=t0, save_interval=save_interval, last_save_time=last_save_time, snap_counter=snap_counter
        )

        total_points = len(line) + total_points1 + total_points2

        final_path = _lines_dir / "plan_line_final.png"
        plt.legend()
        plt.savefig(final_path, dpi=300, bbox_inches="tight")
        plt.close()

        dot_path = _save_dot_csv(self.all_lines, _dot_dir)

        total_elapsed = time.time() - t0
        print(f"\n[测线规划完成] 总耗时: {total_elapsed:.1f}s | 共生成{len(self.all_lines)}条测线 | 总点数约{total_points}")
        print(f"  最终图片: {final_path}")
        print(f"  测线点文件: {dot_path}")
        if snap_counter > 0:
            print(f"  中间快照: {snap_counter}张 (每{save_interval}s一张)")


def plan_line(start_x, start_y, xs, ys, cluster_matrix, n=0.1, step=50, theta=120):
    planner = SurveyPlanner(xs, ys, cluster_matrix, theta=theta, n=n)
    planner.plan_line(start_x, start_y, step=step)
