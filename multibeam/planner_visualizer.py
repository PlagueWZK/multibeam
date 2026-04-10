"""
测线可视化模块

封装所有 matplotlib 绘图逻辑，从 Planner.py 中分离出来。
遵循单一职责原则，使 Planner 专注于算法调度。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


class SurveyVisualizer:
    """测线可视化器：封装所有绘图逻辑"""

    FINE_GRID_STATE_STYLES = {
        "full": {"color": "#4CAF50", "alpha": 0.18, "label": "Fine grid - full"},
        "partial": {
            "color": "#FFB300",
            "alpha": 0.22,
            "label": "Fine grid - partial",
        },
        "uncovered": {
            "color": "#E53935",
            "alpha": 0.18,
            "label": "Fine grid - uncovered",
        },
    }

    # 分区颜色方案
    PARTITION_COLORS = [
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

    def __init__(self, xs, ys, cluster_matrix, x_min, x_max, y_min, y_max):
        """
        初始化可视化器

        参数:
            xs, ys: 坐标数组
            cluster_matrix: 聚类矩阵
            x_min, x_max, y_min, y_max: 海域边界
        """
        self.xs = xs
        self.ys = ys
        self.cluster_matrix = cluster_matrix
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.fig = None
        self.ax = None

    def setup_figure(self, partition_id: int):
        """
        为单个分区设置画布

        参数:
            partition_id: 分区ID
        """
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_title(f"Survey Line Planning - Partition {partition_id}")
        self.ax.grid(True, alpha=0.3)

    def draw_line(
        self, line: np.ndarray, color: str, linewidth: float = 1.5, label: str = None
    ):
        """
        绘制单条测线

        参数:
            line: 测线点数组 [N, 3] 或 [N, 2]
            color: 颜色
            linewidth: 线宽
            label: 图例标签
        """
        self.ax.plot(
            line[:, 0], line[:, 1], color=color, linewidth=linewidth, label=label
        )

    def draw_light_line(self, line: np.ndarray):
        """绘制浅色背景线（用于中间过程显示）"""
        self.ax.plot(line[:, 0], line[:, 1], color="lightgray", linewidth=0.6)

    def save_snapshot(self, path: Path):
        """保存当前画布"""
        self.fig.savefig(path, dpi=200, bbox_inches="tight")

    def close_figure(self):
        """关闭画布"""
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

    def draw_partition_background(self):
        """绘制分区背景"""
        grid_x, grid_y = np.meshgrid(self.xs, self.ys)
        partition_ids = sorted(np.unique(self.cluster_matrix).astype(int))

        # 绘制底色
        cmap = plt.cm.get_cmap("Set3", len(partition_ids))
        self.ax.contourf(
            grid_x,
            grid_y,
            self.cluster_matrix,
            levels=np.arange(len(partition_ids) + 1) - 0.5,
            cmap=cmap,
            alpha=0.25,
            zorder=0,
        )

        # 绘制边界线
        self.ax.contour(
            grid_x,
            grid_y,
            self.cluster_matrix,
            levels=np.arange(len(partition_ids)) + 0.5,
            colors="gray",
            linewidths=1.0,
            linestyles="dashed",
            zorder=1,
        )

    def _draw_fine_grid_overlay_on_ax(self, ax, state_grid, show_legend=False):
        """在指定坐标轴上叠加细网格状态。"""
        for cell in state_grid.iter_render_cells():
            style = self.FINE_GRID_STATE_STYLES[cell["state"]]
            alpha = style["alpha"] * max(0.35, min(1.0, cell["partition_area_ratio"]))
            ax.add_patch(
                Rectangle(
                    (cell["x"], cell["y"]),
                    cell["width"],
                    cell["height"],
                    facecolor=style["color"],
                    edgecolor="none",
                    alpha=alpha,
                    zorder=0.8,
                )
            )

        if show_legend:
            return [
                Patch(
                    facecolor=cfg["color"],
                    alpha=cfg["alpha"],
                    edgecolor="none",
                    label=cfg["label"],
                )
                for cfg in self.FINE_GRID_STATE_STYLES.values()
            ]
        return []

    def draw_fine_grid_overlay(self, state_grid, show_legend=False):
        """在当前分区图中叠加细网格状态。"""
        if self.ax is None:
            return []
        return self._draw_fine_grid_overlay_on_ax(self.ax, state_grid, show_legend)

    def _setup_global_ax(self, title: str):
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111)
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        return fig, ax

    def _draw_partition_background_on_ax(self, ax, partition_ids):
        grid_x, grid_y = np.meshgrid(self.xs, self.ys)
        cmap = plt.cm.get_cmap("Set3", len(partition_ids))
        ax.contourf(
            grid_x,
            grid_y,
            self.cluster_matrix,
            levels=np.arange(len(partition_ids) + 1) - 0.5,
            cmap=cmap,
            alpha=0.25,
            zorder=0,
        )

    def _draw_partition_boundaries_on_ax(self, ax, partition_ids):
        grid_x, grid_y = np.meshgrid(self.xs, self.ys)
        ax.contour(
            grid_x,
            grid_y,
            self.cluster_matrix,
            levels=np.arange(len(partition_ids)) + 0.5,
            colors="gray",
            linewidths=1.0,
            linestyles="dashed",
            zorder=1,
        )

    def _draw_lines_on_ax(self, ax, all_results, colors):
        for result in all_results:
            color = colors[int(result.partition_id) % len(colors)]
            for line in result.lines:
                ax.plot(
                    line[:, 0],
                    line[:, 1],
                    color=color,
                    linewidth=1.0,
                    alpha=0.8,
                    zorder=2,
                )

    def _build_partition_line_legend(self, partition_ids, colors):
        return [
            Line2D(
                [0],
                [0],
                color=colors[int(pid) % len(colors)],
                linewidth=2,
                label=f"Partition {int(pid)}",
            )
            for pid in partition_ids
        ]

    def draw_merged_lines(
        self, all_results, lines_parent: Path, partition_state_grids=None
    ):
        """
        绘制所有分区合并的最终测线图

        参数:
            all_results: 所有分区的规划结果列表
            lines_parent: 输出目录
        """
        print(f"\n{'=' * 60}")
        print(f"[全局规划] 正在绘制所有分区合并图...")

        partition_ids = [
            int(pid)
            for pid in sorted(np.unique(self.cluster_matrix).astype(int))
            if int(pid) >= 0
        ]
        colors = self.PARTITION_COLORS

        # 图1：分区视图（分区底色 + 分区边界 + 测线）
        fig_partition, ax_partition = self._setup_global_ax(
            "All Partitions Survey Lines - Partition View"
        )
        self._draw_partition_background_on_ax(ax_partition, partition_ids)
        self._draw_partition_boundaries_on_ax(ax_partition, partition_ids)
        self._draw_lines_on_ax(ax_partition, all_results, colors)
        ax_partition.legend(
            handles=self._build_partition_line_legend(partition_ids, colors), loc="best"
        )
        partition_view_path = lines_parent / "all_partitions_final_partition_view.png"
        fig_partition.savefig(partition_view_path, dpi=300, bbox_inches="tight")
        plt.close(fig_partition)
        print(f"  全局分区视图: {partition_view_path}")

        # 图2：细网格状态视图（细网格状态 + 分区边界 + 测线）
        fig_fine, ax_fine = self._setup_global_ax(
            "All Partitions Survey Lines - Fine Grid View"
        )
        self._draw_partition_boundaries_on_ax(ax_fine, partition_ids)
        state_legend_elements = []
        if partition_state_grids:
            for pid in partition_ids:
                state_grid = partition_state_grids.get(int(pid))
                if state_grid is not None:
                    state_legend_elements = self._draw_fine_grid_overlay_on_ax(
                        ax_fine, state_grid, show_legend=True
                    )
        self._draw_lines_on_ax(ax_fine, all_results, colors)
        ax_fine.legend(
            handles=self._build_partition_line_legend(partition_ids, colors)
            + state_legend_elements,
            loc="best",
        )
        fine_grid_view_path = lines_parent / "all_partitions_final_fine_grid_view.png"
        fig_fine.savefig(fine_grid_view_path, dpi=300, bbox_inches="tight")
        plt.close(fig_fine)
        print(f"  全局细网格视图: {fine_grid_view_path}")


def save_dot_csv(all_lines: list[np.ndarray], dot_dir: Path):
    """
    保存测线点数据到 CSV 文件

    参数:
        all_lines: 所有测线列表
        dot_dir: 输出目录
    """
    dot_dir.mkdir(parents=True, exist_ok=True)
    dot_path = dot_dir / "dot.csv"
    with open(dot_path, "w", encoding="utf-8") as f:
        f.write("line_id,x,y\n")
        for line_id, line in enumerate(all_lines):
            for pt in line:
                f.write(f"{line_id},{pt[0]:.2f},{pt[1]:.2f}\n")
    return dot_path
