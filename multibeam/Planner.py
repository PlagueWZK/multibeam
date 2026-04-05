import copy
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from multibeam.Partition import (
    get_partition_for_point,
    is_point_in_partition,
)
from tool.Data import (
    figure_length,
    figure_width,
    forward_direction,
    get_alpha,
    get_gx,
    get_gy,
    get_height,
    get_w_left,
    get_w_right,
)

import datetime

# ---------------------------------------------------------------------------
# Data classes for instant metric recording
# ---------------------------------------------------------------------------


@dataclass
class LineRecord:
    """单条测线的即时指标记录"""

    line_id: int
    partition_id: int
    points: np.ndarray  # [N, 3] = [x, y, w_total]
    length: float
    coverage: float
    terminated_by: str  # "boundary" / "spiral" / "intersection" / "saturation" / "degradation" / "empty"


@dataclass
class PartitionResult:
    """单个分区的规划结果"""

    partition_id: int
    lines: list[np.ndarray]  # 原始 line 数组（用于绘图）
    records: list[LineRecord]  # 指标记录
    total_length: float
    total_coverage: float


# ---------------------------------------------------------------------------
# Helper functions (unchanged algorithm logic)
# ---------------------------------------------------------------------------


def _save_dot_csv(all_lines, dot_dir):
    dot_dir.mkdir(parents=True, exist_ok=True)
    dot_path = dot_dir / "dot.csv"
    with open(dot_path, "w", encoding="utf-8") as f:
        f.write("line_id,x,y\n")
        for line_id, line in enumerate(all_lines):
            for pt in line:
                f.write(f"{line_id},{pt[0]:.2f},{pt[1]:.2f}\n")
    return dot_path


def _compute_signed_angle(v1, v2):
    """计算从向量v1到v2的带符号夹角（弧度），逆时针为正"""
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    return np.arctan2(cross, dot)


def _point_with_width(x, y, theta):
    """返回 [x, y, 覆盖宽度(w_left + w_right)]，用于后续快速计算覆盖面积"""
    return [x, y, get_w_left(x, y, theta) + get_w_right(x, y, theta)]


def _compute_total_turning_angle(line):
    """计算整条测线的累计偏转角（所有相邻线段夹角之和）"""
    if len(line) < 3:
        return 0.0
    total = 0.0
    for i in range(len(line) - 2):
        v1 = np.array(line[i + 1]) - np.array(line[i])
        v2 = np.array(line[i + 2]) - np.array(line[i + 1])
        total += _compute_signed_angle(v1, v2)
    return total


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


# ---------------------------------------------------------------------------
# SurveyPlanner — fully encapsulated
# ---------------------------------------------------------------------------


class SurveyPlanner:
    """测线规划器：封装分区测线生成与即时指标记录"""

    def __init__(
        self, xs, ys, cluster_matrix, depth_matrix=None, step=50, theta=120, n=0.1
    ):
        self.xs = xs
        self.ys = ys
        self.cluster_matrix = cluster_matrix
        self.depth_matrix = depth_matrix
        self.step = step
        self.theta = theta
        self.theta_rad = np.radians(theta)
        self.n = n

        self.x_min = float(xs[0])
        self.x_max = float(xs[-1])
        self.y_min = float(ys[0])
        self.y_max = float(ys[-1])
        self.total_area = (self.x_max - self.x_min) * (self.y_max - self.y_min)

        # Instant metric storage
        self.all_records: list[LineRecord] = []
        self._line_counter = 0

        # For Ctrl+C emergency save
        self._emergency_lines: list[np.ndarray] = []
        self._emergency_dot_dir: Path | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_line(self, points, partition_id, terminated_by):
        """测线生成完毕后立即计算并记录指标"""
        pts = np.array(points)
        length = figure_length(pts)
        coverage = figure_width(pts)

        record = LineRecord(
            line_id=self._line_counter,
            partition_id=partition_id,
            points=pts,
            length=length,
            coverage=coverage,
            terminated_by=terminated_by,
        )
        self.all_records.append(record)
        self._line_counter += 1
        return record

    def _save_emergency_dot(self):
        """Ctrl+C 中断时保存当前测线点"""
        if self._emergency_dot_dir and self._emergency_lines:
            try:
                _save_dot_csv(self._emergency_lines, self._emergency_dot_dir)
                print(f"[中断] 已保存测线点文件: {self._emergency_dot_dir / 'dot.csv'}")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Core line generation (algorithm unchanged)
    # ------------------------------------------------------------------

    def _extend_line_bidirectional(self, start_x, start_y, target_partition_id, step):
        """
        双向延伸生成主测线: 从起点向正反两个方向交替延伸,
        使用整条测线的累计偏转角法检测环形/螺旋.
        由于是第一条测线, 不进行与其他测线相交的检查.
        """
        line = [_point_with_width(start_x, start_y, self.theta)]

        ext_step = step / 2
        ext_loop = 0
        front_x, front_y = start_x, start_y
        back_x, back_y = start_x, start_y
        front_stopped = False
        back_stopped = False
        extend_front = True
        ring_closed = False

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
                    print(f"  [主测线延伸-前端] 延伸{ext_loop}步后超出边界")
                    front_stopped = True
                else:
                    print(f"  [主测线延伸-后端] 延伸{ext_loop}步后超出边界")
                    back_stopped = True
                extend_front = not extend_front
                continue

            new_point = _point_with_width(new_x, new_y, self.theta)
            if extend_front:
                line.append(new_point)
            else:
                line.insert(0, new_point)

            total_angle = _compute_total_turning_angle(line)
            if abs(total_angle) > 2 * np.pi:
                print(
                    f"  [主测线延伸] 累计偏转角{np.degrees(total_angle):.1f}° > 360°，螺旋终止"
                )
                ring_closed = True
                break

            if extend_front:
                front_x, front_y = new_x, new_y
            else:
                back_x, back_y = new_x, new_y

            extend_front = not extend_front

        line = np.array(line)
        status = "环形闭合" if ring_closed else "边界终止"
        terminated_by = "spiral" if ring_closed else "boundary"
        print(
            f"[主测线1] 完成 | 共{len(line)}个点 | {status} | 起点({line[0][0]:.1f},{line[0][1]:.1f}) 终点({line[-1][0]:.1f},{line[-1][1]:.1f})"
        )
        return line, terminated_by

    def _generate_perpendicular_lines(
        self,
        main_line,
        direction,
        target_partition_id,
        lines_dir,
        t0,
        line_counter,
        partition_id,
    ):
        line = copy.deepcopy(main_line)
        initial_len = len(line)
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
                    new_x,
                    new_y,
                    target_partition_id,
                    self.xs,
                    self.ys,
                    self.cluster_matrix,
                )
                if is_in:
                    t1.append(_point_with_width(new_x, new_y, self.theta))

            if len(line) > 5:
                plt.plot(line[:, 0], line[:, 1], color="lightgray", linewidth=0.6)

            elapsed = time.time() - t0
            print(
                f"[{direction_name}扩展 第{perp_iter}轮] 当前{len(line)}点 -> 新增{len(t1)}有效点 | 已耗时{elapsed:.1f}s"
            )

            if len(t1) == 0:
                print(f"[{direction_name}扩展] 第{perp_iter}轮无新有效点，结束规划")
                break

            if len(t1) < 5:
                print(
                    f"[{direction_name}扩展] 第{perp_iter}轮仅{len(t1)}个新点(<5)，测线退化，结束规划"
                )
                break

            line = copy.deepcopy(t1)
            turning_angle_terminated = False
            ring_closed_saturated = False

            if in_convergence_state:
                line_arr = np.array(line)
                current_line_length = figure_length(line_arr)
                print(
                    f"  [收敛状态] 当前测线长度: {current_line_length:.1f}m, 父测线长度: {parent_line_length:.1f}m"
                )

                line_arr = np.array(line)
                centroid = np.mean(line_arr, axis=0)
                dists_to_centroid = np.sqrt(
                    (line_arr[:, 0] - centroid[0]) ** 2
                    + (line_arr[:, 1] - centroid[1]) ** 2
                )
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

                to_centroid = np.array(
                    [centroid[0] - nearest_point[0], centroid[1] - nearest_point[1]]
                )
                to_centroid_norm = np.linalg.norm(to_centroid)
                if to_centroid_norm > 1e-6:
                    to_centroid = to_centroid / to_centroid_norm

                dot_product = np.dot(grad_dir, to_centroid)
                if dot_product > 0:
                    inner_width = get_w_right(
                        nearest_point[0], nearest_point[1], self.theta
                    )
                else:
                    inner_width = get_w_left(
                        nearest_point[0], nearest_point[1], self.theta
                    )

                if min_dist_to_centroid < inner_width:
                    print(
                        f"  [测线饱和检测] 最近点距质心{min_dist_to_centroid:.1f}m < 内侧覆盖宽度{inner_width:.1f}m，测线饱和"
                    )
                    ring_closed_saturated = True
                else:
                    print(
                        f"  [测线饱和检测] 最近点距质心{min_dist_to_centroid:.1f}m >= 内侧覆盖宽度{inner_width:.1f}m，继续生成"
                    )
            else:
                ext_step = self.step / 2
                ext_loop = 0
                intersection_threshold = self.step * 0.5

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
                        new_x,
                        new_y,
                        target_partition_id,
                        self.xs,
                        self.ys,
                        self.cluster_matrix,
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

                    new_point = _point_with_width(new_x, new_y, self.theta)

                    if extend_front:
                        line.append(new_point)
                    else:
                        line.insert(0, new_point)

                    total_angle = _compute_total_turning_angle(line)
                    if abs(total_angle) > 2 * np.pi:
                        print(
                            f"  [测线延伸] 累计偏转角{np.degrees(total_angle):.1f}° > 360°，螺旋终止"
                        )
                        turning_angle_terminated = True
                        break

                    found_intersection = False
                    for prev_idx, prev_line in enumerate(prev_lines):
                        if _check_line_intersection(
                            new_point, prev_line, intersection_threshold
                        ):
                            if extend_front:
                                print(
                                    f"  [测线延伸-前端] 延伸{ext_loop}步后与第{len(prev_lines) - prev_idx}条测线相交，迷路终止"
                                )
                            else:
                                print(
                                    f"  [测线延伸-后端] 延伸{ext_loop}步后与第{len(prev_lines) - prev_idx}条测线相交，迷路终止"
                                )
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

                    if extend_front:
                        front_x, front_y = new_x, new_y
                    else:
                        back_x, back_y = new_x, new_y

                    extend_front = not extend_front

            line = np.array(line)

            # --- Instant metric recording ---
            terminated_by = (
                "saturation"
                if ring_closed_saturated
                else (
                    "spiral"
                    if turning_angle_terminated
                    else ("intersection" if intersection_terminated else "boundary")
                )
            )
            self._record_line(line, partition_id, terminated_by)
            self._emergency_lines.append(line.copy())

            prev_lines.append(line.copy())
            if len(prev_lines) > 3:
                prev_lines.pop(0)
            total_points += len(line)

            if turning_angle_terminated:
                plt.plot(line[:, 0], line[:, 1], color="red", linewidth=1.5)
                current_line_length = figure_length(line)
                print(
                    f"  [螺旋终止判定] 当前测线长度: {current_line_length:.1f}m, 父测线长度: {parent_line_length:.1f}m"
                )
                if current_line_length < parent_line_length:
                    in_convergence_state = True
                    print(
                        f"  [进入收敛状态] 长度缩短，后续测线将跳过自延伸并进行饱和检查"
                    )
                else:
                    print(f"  [保持扩展状态] 长度未缩短，继续正常扩展")
            elif intersection_terminated:
                plt.plot(line[:, 0], line[:, 1], color="purple", linewidth=1.5)

            line_counter += 1
            snap_path = lines_dir / f"line_{line_counter:04d}_iter{perp_iter}.png"
            plt.savefig(snap_path, dpi=200, bbox_inches="tight")
            print(f"  >> 已保存测线: {snap_path}")

            parent_line_length = figure_length(line)

            if ring_closed_saturated:
                print(f"[{direction_name}扩展] 测线饱和，终止该方向扩展")
                break

        print(f"[{direction_name}扩展] 完成 | 共{perp_iter}轮 | {total_points}点")
        return total_points, line_counter

    # ------------------------------------------------------------------
    # Partition-level planning
    # ------------------------------------------------------------------

    def _plan_partition(
        self,
        partition_id,
        lines_dir,
        dot_dir,
        t0,
    ):
        """
        为指定分区规划测线, 结果保存到指定目录.
        返回 PartitionResult (包含测线列表 + 即时指标).
        """
        lines_dir.mkdir(parents=True, exist_ok=True)
        dot_dir.mkdir(parents=True, exist_ok=True)

        self._emergency_dot_dir = dot_dir

        rows, cols = np.where(self.cluster_matrix == partition_id)
        if len(rows) == 0:
            print(f"[分区{partition_id}] 未找到有效网格点, 跳过")
            return PartitionResult(partition_id, [], [], 0.0, 0.0)

        start_x = float(np.mean(self.xs[cols]))
        start_y = float(np.mean(self.ys[rows]))

        print(f"\n{'=' * 60}")
        print(
            f"[分区{partition_id}] 开始规划 | 起点: ({start_x:.1f}, {start_y:.1f}) | 分区中心"
        )

        plt.figure(figsize=(12, 10))
        plt.xlim(self.x_min, self.x_max)
        plt.ylim(self.y_min, self.y_max)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title(f"Survey Line Planning - Partition {partition_id}")
        plt.grid(True, alpha=0.3)

        line_counter = 0
        partition_lines = []

        # 主测线
        print(
            f"[分区{partition_id}] 主测线开始双向延伸 | 起点: ({start_x:.1f}, {start_y:.1f})"
        )
        line, main_terminated = self._extend_line_bidirectional(
            start_x, start_y, partition_id, self.step
        )
        main_record = self._record_line(line, partition_id, main_terminated)
        self._emergency_lines.append(line.copy())
        partition_lines.append(line.copy())
        line_counter += 1
        plt.plot(line[:, 0], line[:, 1], color="b", linewidth=1.5, label="Main line")

        snap_path = lines_dir / f"line_{line_counter:04d}_main.png"
        plt.savefig(snap_path, dpi=200, bbox_inches="tight")
        print(f"  >> 已保存测线: {snap_path}")

        total_points1, line_counter = self._generate_perpendicular_lines(
            line,
            direction=1,
            target_partition_id=partition_id,
            lines_dir=lines_dir,
            t0=t0,
            line_counter=line_counter,
            partition_id=partition_id,
        )

        total_points2, line_counter = self._generate_perpendicular_lines(
            line,
            direction=-1,
            target_partition_id=partition_id,
            lines_dir=lines_dir,
            t0=t0,
            line_counter=line_counter,
            partition_id=partition_id,
        )

        total_points = len(line) + total_points1 + total_points2

        final_path = lines_dir / "plan_line_final.png"
        plt.legend()
        plt.savefig(final_path, dpi=300, bbox_inches="tight")
        plt.close()

        dot_path = _save_dot_csv(partition_lines, dot_dir)

        # 从全局记录中筛选该分区的完整记录（包含主测线 + 所有垂直测线）
        partition_records = [
            r for r in self.all_records if r.partition_id == partition_id
        ]
        total_length = sum(r.length for r in partition_records)
        total_coverage = sum(r.coverage for r in partition_records)

        # 从记录中提取完整测线列表（用于合并图和 dot.csv）
        partition_all_lines = [r.points for r in partition_records]

        print(
            f"[分区{partition_id}完成] 共生成{len(partition_records)}条测线 | 总点数约{total_points}"
        )
        print(f"  最终图片: {final_path}")
        print(f"  测线点文件: {dot_path}")
        print(f"  中间测线: {line_counter}张")

        return PartitionResult(
            partition_id=partition_id,
            lines=partition_all_lines,
            records=partition_records,
            total_length=total_length,
            total_coverage=total_coverage,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan_line(self, start_x, start_y, output_dir=None):
        """
        单分区测线规划（从指定起点开始）。
        用于向后兼容旧 API。
        """
        if output_dir is None:
            output_dir = "./multibeam/output"
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        lines_dir = Path(output_dir) / "lines" / current_time
        dot_dir = Path(output_dir) / "dot" / current_time

        target_partition_id = get_partition_for_point(
            start_x, start_y, self.xs, self.ys, self.cluster_matrix
        )
        print(f"[测线规划] 目标分区ID: {target_partition_id}")

        result = self._plan_partition(
            partition_id=target_partition_id,
            lines_dir=lines_dir,
            dot_dir=dot_dir,
            t0=time.time(),
        )
        return result

    def plan_all(self, output_dir=None):
        """
        自动遍历所有分区, 每个分区以质心为起点规划测线.

        输出结构:
            output/
              lines/<timestamp>/
                partition_0/
                  line_0001_main.png
                  line_0002_iter1.png
                  ...
                  plan_line_final.png
                partition_1/
                  ...
                all_partitions_final.png
              dot/<timestamp>/
                partition_0/dot.csv
                partition_1/dot.csv
                dot.csv
        """
        if output_dir is None:
            output_dir = "./multibeam/output"

        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir)
        lines_parent = output_path / "lines" / current_time
        dot_parent = output_path / "dot" / current_time

        partition_ids = sorted(np.unique(self.cluster_matrix).astype(int))
        print(f"[全局规划] 检测到 {len(partition_ids)} 个分区: {partition_ids}")
        print(f"[全局规划] 输出根目录: {lines_parent}")

        all_partition_results: list[PartitionResult] = []

        for pid in partition_ids:
            lines_dir = lines_parent / f"partition_{pid}"
            dot_dir = dot_parent / f"partition_{pid}"

            t0 = time.time()
            result = self._plan_partition(
                partition_id=pid,
                lines_dir=lines_dir,
                dot_dir=dot_dir,
                t0=t0,
            )
            all_partition_results.append(result)

        # 绘制所有区域合起来的最终测线图
        self._draw_merged_lines(all_partition_results, partition_ids, lines_parent)

        # 保存所有测线点
        self._save_all_dots(all_partition_results, dot_parent)

        total_lines = sum(len(r.records) for r in all_partition_results)
        print(f"[全局规划完成] 共 {len(partition_ids)} 个分区 | {total_lines} 条测线")

        return all_partition_results

    def _draw_merged_lines(self, all_results, partition_ids, lines_parent):
        """绘制所有分区合并的最终测线图（含分区底色 + 测线）"""
        print(f"\n{'=' * 60}")
        print(f"[全局规划] 正在绘制所有分区合并图...")

        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

        plt.figure(figsize=(16, 12))
        plt.xlim(self.x_min, self.x_max)
        plt.ylim(self.y_min, self.y_max)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title("All Partitions Survey Lines")
        plt.grid(True, alpha=0.3)

        # 1. 绘制分区底色（低透明度，不遮挡测线）
        grid_x, grid_y = np.meshgrid(self.xs, self.ys)
        cmap = plt.cm.get_cmap("Set3", len(partition_ids))
        plt.contourf(
            grid_x,
            grid_y,
            self.cluster_matrix,
            levels=np.arange(len(partition_ids) + 1) - 0.5,
            cmap=cmap,
            alpha=0.25,
            zorder=0,
        )

        # 2. 绘制分区边界线
        plt.contour(
            grid_x,
            grid_y,
            self.cluster_matrix,
            levels=np.arange(len(partition_ids)) + 0.5,
            colors="gray",
            linewidths=1.0,
            linestyles="dashed",
            zorder=1,
        )

        # 3. 绘制测线（在底色之上）
        for result in all_results:
            color = colors[int(result.partition_id) % len(colors)]
            for line in result.lines:
                plt.plot(
                    line[:, 0],
                    line[:, 1],
                    color=color,
                    linewidth=1.0,
                    alpha=0.8,
                    zorder=2,
                )

        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                color=colors[int(pid) % len(colors)],
                linewidth=2,
                label=f"Partition {int(pid)}",
            )
            for pid in partition_ids
        ]
        plt.legend(handles=legend_elements, loc="best")

        all_final_path = lines_parent / "all_partitions_final.png"
        plt.savefig(all_final_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  合并最终图: {all_final_path}")

    def _save_all_dots(self, all_results, dot_parent):
        """合并所有分区 dot.csv"""
        dot_parent.mkdir(parents=True, exist_ok=True)
        flat_all_lines = [line for r in all_results for line in r.lines]
        _save_dot_csv(flat_all_lines, dot_parent)
        print(f"  全部测线点: {dot_parent / 'dot.csv'}")

    # ------------------------------------------------------------------
    # Metrics output (reads from instant records, no recalculation)
    # ------------------------------------------------------------------

    def save_metrics_excel(self, output_dir=None, current_time=None):
        """
        从已记录的指标直接生成 Excel，不重新计算。
        """
        import pandas as pd

        if output_dir is None:
            output_dir = "./multibeam/output"
        if current_time is None:
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        metrics_dir = Path(output_dir) / "metrics" / current_time
        metrics_dir.mkdir(parents=True, exist_ok=True)

        partition_rows = []
        global_total_length = 0.0
        global_total_coverage = 0.0
        global_total_lines = 0

        for pid in sorted(set(r.partition_id for r in self.all_records)):
            recs = [r for r in self.all_records if r.partition_id == pid]
            total_length = sum(r.length for r in recs)
            total_coverage = sum(r.coverage for r in recs)

            partition_rows.append(
                {
                    "分区ID": int(pid),
                    "测线条数": len(recs),
                    "测线总长度(m)": round(total_length, 2),
                    "覆盖面积(m²)": round(total_coverage, 2),
                }
            )

            global_total_lines += len(recs)
            global_total_length += total_length
            global_total_coverage += total_coverage

        df_partition = pd.DataFrame(partition_rows)

        effective_coverage = global_total_coverage * (1 - self.n)
        coverage_rate = (
            effective_coverage / self.total_area if self.total_area > 0 else 0
        )
        miss_rate = 1 - coverage_rate

        global_rows = [
            {"指标": "测线条数", "值": global_total_lines},
            {"指标": "测线总长度(m)", "值": round(global_total_length, 2)},
            {"指标": "总覆盖面积(m²)", "值": round(global_total_coverage, 2)},
            {"指标": "有效覆盖面积(m²)", "值": round(effective_coverage, 2)},
            {"指标": "待测海域总面积(m²)", "值": round(self.total_area, 2)},
            {"指标": "覆盖率", "值": f"{coverage_rate:.4%}"},
            {"指标": "漏测率", "值": f"{miss_rate:.4%}"},
        ]
        df_global = pd.DataFrame(global_rows)

        metrics_path = metrics_dir / "metrics.xlsx"
        with pd.ExcelWriter(metrics_path, engine="openpyxl") as writer:
            df_partition.to_excel(writer, sheet_name="分区统计", index=False)
            df_global.to_excel(writer, sheet_name="全局统计", index=False)

        print(f"\n[指标统计完成] 已保存到: {metrics_path}")
        print(f"  测线条数: {global_total_lines}")
        print(f"  测线总长度: {global_total_length:.2f} m")
        print(f"  总覆盖面积: {global_total_coverage:.2f} m²")
        print(f"  有效覆盖面积: {effective_coverage:.2f} m²")
        print(f"  覆盖率: {coverage_rate:.4%}")
        print(f"  漏测率: {miss_rate:.4%}")

        return metrics_path


# ---------------------------------------------------------------------------
# Backward-compatible module-level functions
# ---------------------------------------------------------------------------


def plan_line(start_x, start_y, xs, ys, cluster_matrix, n=0.1, step=50, theta=120):
    """向后兼容: 单分区测线规划"""
    planner = SurveyPlanner(xs, ys, cluster_matrix, theta=theta, n=n, step=step)
    return planner.plan_line(start_x, start_y)


def plan_all_line(xs, ys, cluster_matrix, n=0.1, step=50, theta=120, total_area=None):
    """向后兼容: 全局测线规划"""
    planner = SurveyPlanner(xs, ys, cluster_matrix, theta=theta, n=n, step=step)
    results = planner.plan_all()
    planner.save_metrics_excel()
    return results
