"""
测线规划核心模块（重构版）

本模块负责测线生成的核心调度逻辑。
数据类、几何计算和可视化逻辑已拆分到独立模块。
"""

import copy
import datetime
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

# 从拆分后的模块导入
from multibeam.coverage_state_grid import PartitionCoverageStateGrid
from multibeam.models import LineRecord, PartitionResult, TerminationReason
from multibeam.Partition import get_partition_for_point, is_point_in_partition
from multibeam.planner_visualizer import SurveyVisualizer, save_dot_csv
from tool.Data import (
    figure_length,
    figure_width,
    forward_direction,
    get_alpha,
    get_gx,
    get_gy,
    get_height,
    get_total_width,
)
from tool.geometry import check_line_intersection_shapely


# ---------------------------------------------------------------------------
# SurveyPlanner — 测线规划核心类
# ---------------------------------------------------------------------------


class SurveyPlanner:
    """测线规划器：封装分区测线生成与即时指标记录"""

    def __init__(
        self,
        xs,
        ys,
        cluster_matrix,
        step=50,
        theta=120,
        n=0.1,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
    ):
        self.xs = xs
        self.ys = ys
        self.cluster_matrix = cluster_matrix
        self.default_step = float(step)
        self.step = float(step)
        self.theta = theta
        self.theta_rad = np.radians(theta)
        self.n = n

        # 双网格细网格参数（按冻结方案设置默认值）
        self.fine_grid_scale = 1.5
        self.max_cells_per_partition = 100000
        self.fine_grid_sampling_points = 9
        self.fine_grid_full_threshold = 0.9
        self.step_scale = 1.0
        self.microstep_min = 10.0
        self.microstep_max = 70.0
        self.legacy_microstep = 35.0
        self.current_microstep = self.legacy_microstep
        self.current_partition_grid = None
        self.partition_state_grids = {}
        self.partition_coverage_summaries = {}

        # 支持传入真实海域边界
        self.x_min = float(x_min) if x_min is not None else float(xs[0])
        self.x_max = float(x_max) if x_max is not None else float(xs[-1])
        self.y_min = float(y_min) if y_min is not None else float(ys[0])
        self.y_max = float(y_max) if y_max is not None else float(ys[-1])
        self.total_area = (self.x_max - self.x_min) * (self.y_max - self.y_min)

        # 即时指标存储
        self.all_records: list[LineRecord] = []
        self._line_counter = 0

        # 初始化可视化器
        self.visualizer = SurveyVisualizer(
            xs, ys, cluster_matrix, self.x_min, self.x_max, self.y_min, self.y_max
        )

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _is_in_partition(self, x, y, partition_id):
        """判断点 (x, y) 是否属于指定分区"""
        is_in, _ = is_point_in_partition(
            x,
            y,
            partition_id,
            self.xs,
            self.ys,
            self.cluster_matrix,
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=self.y_max,
        )
        return is_in

    def _is_converging(self, line: np.ndarray, parent_line: np.ndarray) -> bool:
        """
        判断测线是否在收敛（收缩）

        算法：计算当前测线质心，比较质心到自身和到父测线的平均距离。
        若前者小，说明测线在收缩，视为收敛状态。

        参数:
            line: 当前测线 [N, 3] 或 [N, 2]
            parent_line: 父测线 [M, 3] 或 [M, 2]

        返回:
            bool: True 表示收敛
        """
        centroid = np.mean(line[:, :2], axis=0)

        # 当前测线到自身质心的平均距离
        dist_current = np.mean(
            np.sqrt((line[:, 0] - centroid[0]) ** 2 + (line[:, 1] - centroid[1]) ** 2)
        )

        # 父测线到当前测线质心的平均距离
        dist_parent = np.mean(
            np.sqrt(
                (parent_line[:, 0] - centroid[0]) ** 2
                + (parent_line[:, 1] - centroid[1]) ** 2
            )
        )

        return dist_current < dist_parent

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

    def _point_with_width(self, x, y):
        """返回 [x, y, 覆盖宽度(w_total)]。"""
        return [
            x,
            y,
            get_total_width(
                x,
                y,
                theta=self.theta,
                microstep=self.current_microstep,
            ),
        ]

    def _get_alpha(self, x, y):
        """使用当前分区的差分步长计算坡度角。"""
        return get_alpha(x, y, microstep=self.current_microstep)

    def _update_current_partition_grid_with_segment(self, start_point, end_point):
        """将新线段写入当前分区细网格状态。"""
        if self.current_partition_grid is None:
            return
        self.current_partition_grid.update_segment(
            np.asarray(start_point, dtype=float), np.asarray(end_point, dtype=float)
        )

    def _update_current_partition_grid_with_polyline(self, points):
        """将整条折线写入当前分区细网格状态。"""
        if self.current_partition_grid is None:
            return
        self.current_partition_grid.update_polyline(points)

    def _evaluate_termination(
        self,
        line: np.ndarray,
        new_point: np.ndarray,
        prev_lines: list[np.ndarray],
        in_convergence_state: bool,
        parent_line_length: float,
    ) -> TerminationReason:
        """
        评估测线终止原因

        参数:
            line: 当前测线（包含新点）
            new_point: 最新添加的点
            prev_lines: 历史测线列表
            in_convergence_state: 是否处于收敛状态
            parent_line_length: 父测线长度

        返回:
            TerminationReason: 终止原因枚举
        """
        from tool.geometry import check_self_intersection

        # 1. 检查测线自交（螺旋终止）
        if check_self_intersection(line, min_points=10):
            return TerminationReason.SPIRAL

        # 2. 检查测线相交（迷路终止）
        if len(line) >= 2 and prev_lines:
            # 检查最后两个点组成的线段是否与历史测线相交
            if check_line_intersection_shapely(line[-2], line[-1], prev_lines):
                return TerminationReason.INTERSECTION

        # 3. 检查测线退化（点数过少）
        if len(line) < 5:
            return TerminationReason.DEGRADATION

        # 4. 检查测线饱和（收敛状态下）
        if in_convergence_state:
            centroid = np.mean(line, axis=0)
            dists = np.sqrt(
                (line[:, 0] - centroid[0]) ** 2 + (line[:, 1] - centroid[1]) ** 2
            )
            min_dist = np.min(dists)

            if min_dist < self.step * 2:
                return TerminationReason.SATURATION

        return TerminationReason.NONE

    # ------------------------------------------------------------------
    # 核心测线生成算法
    # ------------------------------------------------------------------

    def _extend_line_bidirectional(self, start_x, start_y, target_partition_id, step):
        """
        双向延伸生成主测线

        从起点向正反两个方向交替延伸，使用累计偏转角法检测环形/螺旋。
        """
        line = [self._point_with_width(start_x, start_y)]

        ext_step = step / 2
        ext_loop = 0
        front_x, front_y = start_x, start_y
        back_x, back_y = start_x, start_y
        front_stopped = False
        back_stopped = False
        extend_front = True
        terminated_reason = TerminationReason.NONE

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
                dx, dy = forward_direction(
                    get_gx(front_x, front_y), get_gy(front_x, front_y)
                )
                new_x = front_x + ext_step * dx
                new_y = front_y + ext_step * dy
            else:
                dx, dy = forward_direction(
                    get_gx(back_x, back_y), get_gy(back_x, back_y)
                )
                new_x = back_x - ext_step * dx
                new_y = back_y - ext_step * dy

            if not self._is_in_partition(new_x, new_y, target_partition_id):
                if extend_front:
                    print(f"  [主测线延伸-前端] 延伸{ext_loop}步后超出边界")
                    front_stopped = True
                else:
                    print(f"  [主测线延伸-后端] 延伸{ext_loop}步后超出边界")
                    back_stopped = True
                extend_front = not extend_front
                continue

            new_point = self._point_with_width(new_x, new_y)
            if extend_front:
                self._update_current_partition_grid_with_segment(line[-1], new_point)
                line.append(new_point)
            else:
                self._update_current_partition_grid_with_segment(new_point, line[0])
                line.insert(0, new_point)

            # 检查终止条件
            line_arr = np.array(line)
            reason = self._evaluate_termination(line_arr, new_point, [], False, 0.0)

            if reason == TerminationReason.SPIRAL:
                print(f"  [主测线延伸] 检测到自交，螺旋终止")
                terminated_reason = reason
                break

            if extend_front:
                front_x, front_y = new_x, new_y
            else:
                back_x, back_y = new_x, new_y

            extend_front = not extend_front

        line = np.array(line)
        status = (
            "环形闭合" if terminated_reason == TerminationReason.SPIRAL else "边界终止"
        )
        terminated_by = (
            terminated_reason.name.lower()
            if terminated_reason != TerminationReason.NONE
            else "boundary"
        )
        print(
            f"[主测线1] 完成 | 共{len(line)}个点 | {status} | "
            f"起点({line[0][0]:.1f},{line[0][1]:.1f}) 终点({line[-1][0]:.1f},{line[-1][1]:.1f})"
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
        draw_first_line=True,
    ):
        """生成垂直扩展测线"""
        line = copy.deepcopy(main_line)
        prev_lines = [main_line.copy()]
        in_convergence_state = False
        parent_line_length = figure_length(main_line)
        perp_iter = 0
        total_points = 0

        direction_name = "正向" if direction == 1 else "反向"
        print(f"\n[{direction_name}扩展] 开始从主测线扩展...")

        while True:
            perp_iter += 1
            t1 = []

            # 计算下一轮测线点
            for i in line:
                x, y = i[0], i[1]
                alpha = self._get_alpha(x, y)
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

                if self._is_in_partition(new_x, new_y, target_partition_id):
                    t1.append(self._point_with_width(new_x, new_y))

            # 绘制中间过程
            if len(line) > 5 and (draw_first_line or perp_iter > 1):
                self.visualizer.draw_light_line(line)

            elapsed = time.time() - t0
            print(
                f"[{direction_name}扩展 第{perp_iter}轮] 当前{len(line)}点 -> 新增{len(t1)}有效点 | 已耗时{elapsed:.1f}s"
            )

            # 检查空测线
            if len(t1) == 0:
                print(f"[{direction_name}扩展] 第{perp_iter}轮无新有效点，结束规划")
                break

            # 检查测线退化
            if len(t1) < 5:
                print(
                    f"[{direction_name}扩展] 第{perp_iter}轮仅{len(t1)}个新点(<5)，测线退化，结束规划"
                )
                break

            line = copy.deepcopy(t1)
            self._update_current_partition_grid_with_polyline(line)
            terminated_reason = TerminationReason.NONE

            # 主循环：自延伸和终止检测
            if not in_convergence_state:
                ext_step = self.step / 2
                ext_loop = 0

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
                        dx, dy = forward_direction(
                            get_gx(front_x, front_y), get_gy(front_x, front_y)
                        )
                        new_x = front_x + ext_step * dx
                        new_y = front_y + ext_step * dy
                    else:
                        dx, dy = forward_direction(
                            get_gx(back_x, back_y), get_gy(back_x, back_y)
                        )
                        new_x = back_x - ext_step * dx
                        new_y = back_y - ext_step * dy

                    if not self._is_in_partition(new_x, new_y, target_partition_id):
                        if extend_front:
                            print(f"  [测线延伸-前端] 延伸{ext_loop}步后超出边界")
                            front_stopped = True
                        else:
                            print(f"  [测线延伸-后端] 延伸{ext_loop}步后超出边界")
                            back_stopped = True
                        extend_front = not extend_front
                        continue

                    new_point = self._point_with_width(new_x, new_y)

                    if extend_front:
                        self._update_current_partition_grid_with_segment(
                            line[-1], new_point
                        )
                        line.append(new_point)
                    else:
                        self._update_current_partition_grid_with_segment(
                            new_point, line[0]
                        )
                        line.insert(0, new_point)

                    # 评估终止条件
                    line_arr = np.array(line)
                    reason = self._evaluate_termination(
                        line_arr,
                        new_point,
                        prev_lines,
                        in_convergence_state,
                        parent_line_length,
                    )

                    if reason == TerminationReason.SPIRAL:
                        print(f"  [测线延伸] 检测到自交，螺旋终止")
                        terminated_reason = reason
                        break

                    if reason == TerminationReason.INTERSECTION:
                        print(f"  [测线延伸] 与历史测线相交，迷路终止")
                        terminated_reason = reason
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

            # 记录指标
            terminated_by = terminated_reason.name.lower()
            self._record_line(line, partition_id, terminated_by)

            prev_lines.append(line.copy())
            if len(prev_lines) > 3:
                prev_lines.pop(0)
            total_points += len(line)

            # 绘制和保存
            if terminated_reason == TerminationReason.SPIRAL:
                self.visualizer.draw_line(line, "red", 1.5)
                # 使用质心距离比较判断收敛状态
                if self._is_converging(line, main_line):
                    in_convergence_state = True
                    print(f"  [向内螺旋] 进入收敛状态，后续将进行饱和检查")
                else:
                    print(f"  [向外螺旋] 继续正常扩展")
            elif terminated_reason == TerminationReason.INTERSECTION:
                self.visualizer.draw_line(line, "purple", 1.5)
            elif in_convergence_state:
                # 收敛状态下的测线用黄色表示
                self.visualizer.draw_line(line, "orange", 1.5)

            line_counter += 1
            snap_path = lines_dir / f"line_{line_counter:04d}_iter{perp_iter}.png"
            self.visualizer.save_snapshot(snap_path)
            print(f"  >> 已保存测线: {snap_path}")

            parent_line_length = figure_length(line)

            # 检查饱和终止
            if in_convergence_state:
                centroid = np.mean(line, axis=0)
                dists = np.sqrt(
                    (line[:, 0] - centroid[0]) ** 2 + (line[:, 1] - centroid[1]) ** 2
                )
                min_dist = np.min(dists)
                threshold = self.step * 2

                print(
                    f"  [饱和检查] min_dist={min_dist:.2f} < threshold={threshold:.2f}?"
                )

                if min_dist < threshold:
                    print(
                        f"[{direction_name}扩展] 测线饱和（最近点距质心{min_dist:.2f}m < {threshold:.2f}m），终止该方向扩展"
                    )
                    break

        print(f"[{direction_name}扩展] 完成 | 共{perp_iter}轮 | {total_points}点")
        return total_points, line_counter

    # ------------------------------------------------------------------
    # 分区级规划
    # ------------------------------------------------------------------

    def _plan_partition(self, partition_id, lines_dir, dot_dir, t0):
        """为指定分区规划测线"""
        lines_dir.mkdir(parents=True, exist_ok=True)
        dot_dir.mkdir(parents=True, exist_ok=True)

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

        state_grid = PartitionCoverageStateGrid(
            partition_id,
            self.xs,
            self.ys,
            self.cluster_matrix,
            theta=self.theta,
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=self.y_max,
            cell_size_scale=self.fine_grid_scale,
            max_cells=self.max_cells_per_partition,
            sampling_points=self.fine_grid_sampling_points,
            full_threshold=self.fine_grid_full_threshold,
            step_scale=self.step_scale,
            microstep_min=self.microstep_min,
            microstep_max=self.microstep_max,
            legacy_microstep=self.legacy_microstep,
        )
        self.partition_state_grids[partition_id] = state_grid

        previous_step = self.step
        previous_microstep = self.current_microstep
        previous_partition_grid = self.current_partition_grid
        self.step = state_grid.planning_step
        self.current_microstep = state_grid.microstep
        self.current_partition_grid = state_grid

        print(
            f"[分区{partition_id}] 细网格初始化 | l={state_grid.cell_size:.2f}m | "
            f"step={self.step:.2f}m | microstep={self.current_microstep:.2f}m | "
            f"cells={state_grid.total_cells}"
            + (" | 已触发cells上限保护" if state_grid.max_cells_guard_triggered else "")
        )

        try:
            # 设置画布
            self.visualizer.setup_figure(partition_id)

            line_counter = 0

            # 主测线
            print(f"[分区{partition_id}] 主测线开始双向延伸")
            line, main_terminated = self._extend_line_bidirectional(
                start_x, start_y, partition_id, self.step
            )
            self._record_line(line, partition_id, main_terminated)
            line_counter += 1
            self.visualizer.draw_line(line, "b", 1.5, "Main line")

            snap_path = lines_dir / f"line_{line_counter:04d}_main.png"
            self.visualizer.save_snapshot(snap_path)
            print(f"  >> 已保存测线: {snap_path}")

            # 正向扩展
            total_points1, line_counter = self._generate_perpendicular_lines(
                line, 1, partition_id, lines_dir, t0, line_counter, partition_id, False
            )

            # 反向扩展
            total_points2, line_counter = self._generate_perpendicular_lines(
                line, -1, partition_id, lines_dir, t0, line_counter, partition_id, False
            )

            total_points = len(line) + total_points1 + total_points2

            # 保存最终图
            overlay_legend = self.visualizer.draw_fine_grid_overlay(
                state_grid, show_legend=True
            )
            final_path = lines_dir / "plan_line_final.png"
            handles, labels = self.visualizer.ax.get_legend_handles_labels()
            self.visualizer.ax.legend(handles=handles + overlay_legend)
            self.visualizer.save_snapshot(final_path)
            self.visualizer.close_figure()

            # 收集结果
            partition_records = [
                r for r in self.all_records if r.partition_id == partition_id
            ]
            partition_all_lines = [r.points for r in partition_records]

            save_dot_csv(partition_all_lines, dot_dir)

            coverage_summary = state_grid.summarize()
            self.partition_coverage_summaries[partition_id] = coverage_summary

            print(
                f"[分区{partition_id}完成] 共生成{len(partition_records)}条测线 | 总点数约{total_points} | "
                f"细网格覆盖率={coverage_summary.coverage_rate:.2%}"
            )

            return PartitionResult(
                partition_id=partition_id,
                lines=partition_all_lines,
                records=partition_records,
                total_length=sum(r.length for r in partition_records),
                total_coverage=sum(r.coverage for r in partition_records),
            )
        finally:
            self.step = previous_step
            self.current_microstep = previous_microstep
            self.current_partition_grid = previous_partition_grid

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def plan_line(self, start_x, start_y, output_dir=None):
        """单分区测线规划"""
        if output_dir is None:
            output_dir = "./multibeam/output"

        output_path = Path(output_dir)
        is_timestamped = len(output_path.name) == 15 and output_path.name[8] == "_"

        if is_timestamped:
            run_dir = output_path
        else:
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = output_path / current_time

        lines_dir = run_dir / "lines"
        metrics_dir = run_dir / "metrics"
        lines_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        target_partition_id = get_partition_for_point(
            start_x,
            start_y,
            self.xs,
            self.ys,
            self.cluster_matrix,
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=self.y_max,
        )

        pid_dir = lines_dir / str(target_partition_id)
        result = self._plan_partition(
            target_partition_id, pid_dir, pid_dir, time.time()
        )

        save_dot_csv(result.lines, lines_dir)
        self.save_metrics_excel(metrics_dir=metrics_dir)

        print(f"[测线规划完成] 输出目录: {run_dir}")
        return result

    def plan_all(self, output_dir=None):
        """自动遍历所有分区规划测线"""
        if output_dir is None:
            output_dir = "./multibeam/output"

        output_path = Path(output_dir)
        is_timestamped = len(output_path.name) == 15 and output_path.name[8] == "_"

        if is_timestamped:
            run_dir = output_path
        else:
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = output_path / current_time

        lines_dir = run_dir / "lines"
        metrics_dir = run_dir / "metrics"
        lines_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        partition_ids = [
            int(pid)
            for pid in sorted(np.unique(self.cluster_matrix).astype(int))
            if pid >= 0
        ]
        print(f"[全局规划] 检测到 {len(partition_ids)} 个分区: {partition_ids}")

        all_results = []
        for pid in partition_ids:
            pid_dir = lines_dir / str(pid)
            result = self._plan_partition(pid, pid_dir, pid_dir, time.time())
            all_results.append(result)

        # 合并输出
        flat_all_lines = [line for r in all_results for line in r.lines]
        save_dot_csv(flat_all_lines, lines_dir)
        self.visualizer.draw_merged_lines(
            all_results, lines_dir, self.partition_state_grids
        )
        self.save_metrics_excel(metrics_dir=metrics_dir)

        total_lines = sum(len(r.records) for r in all_results)
        print(f"[全局规划完成] 共 {len(partition_ids)} 个分区 | {total_lines} 条测线")

        return all_results

    def save_metrics_excel(self, output_dir=None, current_time=None, metrics_dir=None):
        """保存指标到 Excel"""
        import pandas as pd

        if metrics_dir is not None:
            metrics_dir = Path(metrics_dir)
        else:
            if output_dir is None:
                output_dir = "./multibeam/output"
            if current_time is None:
                current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_dir = Path(output_dir) / "metrics" / current_time
        metrics_dir.mkdir(parents=True, exist_ok=True)

        partition_rows = []
        global_total_length = 0.0
        global_old_proxy_coverage = 0.0
        global_total_lines = 0
        global_new_partition_area = 0.0
        global_new_covered_area = 0.0

        partition_ids = sorted(
            set(r.partition_id for r in self.all_records)
            | set(self.partition_coverage_summaries.keys())
        )

        for pid in partition_ids:
            recs = [r for r in self.all_records if r.partition_id == pid]
            total_length = sum(r.length for r in recs)
            proxy_coverage = sum(r.coverage for r in recs)
            coverage_summary = self.partition_coverage_summaries.get(pid)

            partition_rows.append(
                {
                    "分区ID": int(pid),
                    "测线条数": len(recs),
                    "测线总长度(m)": round(total_length, 2),
                    "积分覆盖面积-旧口径(m²)": round(proxy_coverage, 2),
                    "细网格待测面积-新口径(m²)": round(
                        coverage_summary.total_area if coverage_summary else 0.0, 2
                    ),
                    "细网格覆盖面积-新口径(m²)": round(
                        coverage_summary.covered_area if coverage_summary else 0.0, 2
                    ),
                    "细网格覆盖率-新口径": (
                        f"{coverage_summary.coverage_rate:.4%}"
                        if coverage_summary
                        else "0.0000%"
                    ),
                    "细网格漏测率-新口径": (
                        f"{coverage_summary.miss_rate:.4%}"
                        if coverage_summary
                        else "100.0000%"
                    ),
                    "细网格边长l(m)": round(
                        coverage_summary.cell_size if coverage_summary else 0.0, 2
                    ),
                    "规划步长step(m)": round(
                        coverage_summary.planning_step if coverage_summary else 0.0, 2
                    ),
                    "坡度差分microstep(m)": round(
                        coverage_summary.microstep if coverage_summary else 0.0, 2
                    ),
                    "full_cells": coverage_summary.full_cells
                    if coverage_summary
                    else 0,
                    "partial_cells": coverage_summary.partial_cells
                    if coverage_summary
                    else 0,
                    "uncovered_cells": (
                        coverage_summary.uncovered_cells if coverage_summary else 0
                    ),
                }
            )

            global_total_lines += len(recs)
            global_total_length += total_length
            global_old_proxy_coverage += proxy_coverage
            if coverage_summary is not None:
                global_new_partition_area += coverage_summary.total_area
                global_new_covered_area += coverage_summary.covered_area

        df_partition = pd.DataFrame(partition_rows)

        old_effective_coverage = global_old_proxy_coverage * (1 - self.n)
        old_coverage_rate = (
            old_effective_coverage / self.total_area if self.total_area > 0 else 0
        )
        global_new_covered_area = min(
            max(global_new_covered_area, 0.0), self.total_area
        )
        new_coverage_rate = (
            global_new_covered_area / self.total_area if self.total_area > 0 else 0.0
        )
        new_coverage_rate = min(max(new_coverage_rate, 0.0), 1.0)
        new_miss_rate = 1.0 - new_coverage_rate

        global_rows = [
            {"指标": "测线条数", "值": global_total_lines},
            {"指标": "测线总长度(m)", "值": round(global_total_length, 2)},
            {"指标": "矩形海域总面积-旧分母(m²)", "值": round(self.total_area, 2)},
            {
                "指标": "积分总覆盖面积-旧口径(m²)",
                "值": round(global_old_proxy_coverage, 2),
            },
            {
                "指标": "积分有效覆盖面积-旧口径(m²)",
                "值": round(old_effective_coverage, 2),
            },
            {"指标": "积分覆盖率-旧口径", "值": f"{old_coverage_rate:.4%}"},
            {
                "指标": "积分漏测率-旧口径(可能为负)",
                "值": f"{1 - old_coverage_rate:.4%}",
            },
            {
                "指标": "固定矩形海域总面积-新分母(m²)",
                "值": round(self.total_area, 2),
            },
            {
                "指标": "细网格分区待测面积汇总-诊断值(m²)",
                "值": round(global_new_partition_area, 2),
            },
            {
                "指标": "细网格待测面积偏差-诊断值(m²)",
                "值": round(global_new_partition_area - self.total_area, 2),
            },
            {
                "指标": "细网格真实覆盖面积-新口径(m²)",
                "值": round(global_new_covered_area, 2),
            },
            {"指标": "细网格覆盖率-新口径", "值": f"{new_coverage_rate:.4%}"},
            {"指标": "细网格漏测率-新口径", "值": f"{new_miss_rate:.4%}"},
        ]

        for pid in partition_ids:
            coverage_summary = self.partition_coverage_summaries.get(pid)
            if coverage_summary is None:
                continue
            summary_dict = asdict(coverage_summary)
            for key in ("partition_id",):
                summary_dict.pop(key, None)
            for metric_name, metric_value in summary_dict.items():
                display_value = (
                    round(metric_value, 4)
                    if isinstance(metric_value, float)
                    else metric_value
                )
                global_rows.append(
                    {
                        "指标": f"分区{pid}-{metric_name}",
                        "值": display_value,
                    }
                )

        df_global = pd.DataFrame(global_rows)

        metrics_path = metrics_dir / "metrics.xlsx"
        with pd.ExcelWriter(metrics_path, engine="openpyxl") as writer:
            df_partition.to_excel(writer, sheet_name="分区统计", index=False)
            df_global.to_excel(writer, sheet_name="全局统计", index=False)

        print(f"\n[指标统计完成] 已保存到: {metrics_path}")
        return metrics_path


# ---------------------------------------------------------------------------
# 向后兼容接口
# ---------------------------------------------------------------------------


def plan_line(start_x, start_y, xs, ys, cluster_matrix, n=0.1, step=50, theta=120):
    """向后兼容: 单分区测线规划"""
    planner = SurveyPlanner(xs, ys, cluster_matrix, theta=theta, n=n, step=step)
    return planner.plan_line(start_x, start_y)


def plan_all_line(xs, ys, cluster_matrix, n=0.1, step=50, theta=120):
    """向后兼容: 全局测线规划"""
    planner = SurveyPlanner(xs, ys, cluster_matrix, theta=theta, n=n, step=step)
    results = planner.plan_all()
    planner.save_metrics_excel()
    return results
