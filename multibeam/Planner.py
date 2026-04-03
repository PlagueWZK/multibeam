import copy

from multibeam.Partition import *
from tool.Data import *
import datetime


def plan_line(start_x, start_y, xs, ys, cluster_matrix, n=0.1, step=50, theta=120):
    line = [[start_x, start_y]]
    target_partition_id = get_partition_for_point(
        start_x, start_y, xs, ys, cluster_matrix
    )
    # 生成第一条主测线路径
    while True:  # 无限循环直到满足退出条件
        # 获取当前位置的x方向梯度
        gx = get_gx(start_x, start_y)  # 调用函数获取x方向梯度
        # 获取当前位置的y方向梯度
        gy = get_gy(start_x, start_y)
        dx, dy = forward_direction(gx, gy)  # 获取垂直于梯度的前进方向
        # 更新当前位置x坐标
        start_x += step * dx  # 新x坐标 = 原x坐标 + 步长 * x方向分量
        # 更新当前位置y坐标
        start_y += step * dy  # 新y坐标 = 原y坐标 + 步长 * y方向分量
        # 检查是否超出测量区域边界
        is_in, _ = is_point_in_partition(
            start_x, start_y, target_partition_id, xs, ys, cluster_matrix
        )
        if not is_in:
            break  # 超出边界则退出循环
        dist_to_start = np.sqrt((start_x - line[0][0]) ** 2 + (start_y - line[0][1]) ** 2)
        if dist_to_start < 1.2 * step:
            break  # 检测到环形路径则退出循环
        # 将当前位置添加到测线路径
        line.append([start_x, start_y])  # 记录路径点坐标
    # 将路径列表转换为numpy数组
    line = np.array(line)  # 转换为numpy数组便于后续处理
    # 检查路径是否为空

    # 绘制第一条主测线
    plt.plot(line[:, 0], line[:, 1], color="b")  # 用蓝色绘制主测线路径
    # 为第一条主测线生成垂直测线
    while True:  # 无限循环直到所有垂直测线生成完毕
        # 初始化下一轮测线点列表
        t1 = []  # 存储下一轮垂直测线的点坐标
        # 遍历当前测线上的每个点
        for index, i in enumerate(line):  # 对测线上每个点进行处理
            # 获取当前点的x坐标
            x = i[0]  # 提取x坐标
            # 获取当前点的y坐标
            y = i[1]  # 提取y坐标
            # 计算当前位置的坡度角
            alpha = get_alpha(x, y)  # 获取海底坡度角
            # 获取当前位置的深度
            h = get_height(x, y)  # 获取海底深度
            # 根据坡度角大小选择不同的计算方法
            if alpha <= 0.005:  # 如果坡度角很小（近似平坦海底）
                # 使用简化公式计算测线间距
                d = 2 * h * tan(theta / 2) * (1 - n)  # 平坦海底的测线间距公式
                # 计算x方向的偏移量
                tx = d * get_gx(x, y)  # x偏移 = 间距 * x方向梯度
                # 计算y方向的偏移量
                ty = d * get_gy(x, y)  # y偏移 = 间距 * y方向梯度
                # 更新x坐标
                x = x - tx  # 新x坐标 = 原x坐标 + x偏移
                # 更新y坐标
                y = y - ty  # 新y坐标 = 原y坐标 + y偏移
            else:  # 如果坡度角较大（倾斜海底）
                # 计算几何参数A
                A = sin(90 - theta / 2 + alpha)  # 几何参数A
                # 计算几何参数B
                B = sin(90 - theta / 2 - alpha)  # 几何参数B
                # 计算迭代参数C
                C = sin(theta / 2) / A - 1 / sin(alpha)  # 迭代公式中的参数C
                # 计算迭代参数D（注意换行符的处理）
                D = (
                        n * sin(theta / 2) * (1 / A + 1 / B)
                        - sin(theta / 2) / B
                        - 1 / sin(alpha)
                )  # 迭代公式中的参数D

                # 计算下一个测线位置的深度
                next_h = h * C / D  # 根据迭代公式计算新深度
                # 计算x方向的偏移量
                tx = (h - next_h) / tan(alpha) * get_gx(x, y)
                # 计算y方向的偏移量
                ty = (h - next_h) / tan(alpha) * get_gy(x, y)
                # 更新x坐标
                x = x - tx  # 新x坐标 = 原x坐标 + x偏移
                # 更新y坐标
                y = y - ty  # 新y坐标 = 原y坐标 + y偏移
            # 检查新位置是否在有效测量范围内
            is_in, _ = is_point_in_partition(
                x, y, target_partition_id, xs, ys, cluster_matrix
            )
            if is_in:  # 如果在有效范围内
                # 将新位置添加到下一轮测线点列表
                t1.append([x, y])  # 记录有效的测线点
        # 如果当前测线长度足够，进行统计和绘制
        if len(line) > 5:  # 测线点数大于5时才进行统计
            # 绘制当前垂直测线
            plt.plot(line[:, 0], line[:, 1], color="lightgray")  # 用浅灰色绘制垂直测线
            # 记录测线长度统计
            # length[0].append(figure_length(line))  # 将长度添加到第一条测线的统计中
            # # 记录测线覆盖面积统计
            # width[0].append(figure_width(line))  # 将覆盖面积添加到第一条测线的统计中
        # 将当前测线点合并到总测量点集合
        # dot = np.concatenate((dot, line), axis=0)  # 合并所有测量点

        if len(t1) == 0:  # 检查是否还有测线点
            break  # 退出垂直测线生成循环
        # 深拷贝下一轮测线点
        line = copy.deepcopy(t1)  # 深拷贝避免引用问题
        # 从最后一个测线点继续生成主测线
        loc_x = line[-1][0]  # 获取最后一个点的x坐标作为新起点
        # 获取最后一个测线点的y坐标
        loc_y = line[-1][1]  # 获取最后一个点的y坐标作为新起点
        # 调整步长为更小值
        step = 25  # 减小步长以提高精度
        # 继续生成主测线路径
        while True:  # 继续主测线生成循环
            # 获取当前位置的x方向梯度
            gx = get_gx(loc_x, loc_y)  # 计算x方向梯度
            # 获取当前位置的y方向梯度
            gy = get_gy(loc_x, loc_y)  # 计算y方向梯度
            dx, dy = forward_direction(gx, gy)  # 获取前进方向
            # 更新当前位置x坐标
            loc_x += step * dx  # 新x坐标 = 原x坐标 + 步长 * x方向分量
            # 更新当前位置y坐标
            loc_y += step * dy  # 新y坐标 = 原y坐标 + 步长 * y方向分量
            # 检查是否超出测量区域边界
            is_in, _ = is_point_in_partition(
                loc_x, loc_y, target_partition_id, xs, ys, cluster_matrix
            )
            if not is_in:
                break  # 超出边界则退出主测线生成
            dist_to_start = np.sqrt((loc_x - line[0][0]) ** 2 + (loc_y - line[0][1]) ** 2)
            if dist_to_start < 1.2 * step:
                break  # 检测到环形路径则退出循环
            line.append([loc_x, loc_y])  # 记录新的路径点
        # 将路径列表转换为numpy数组
        line = np.array(line)  # 转换为numpy数组
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"./multibeam/output/lines/plan_line_{current_time}.png", dpi=300, bbox_inches="tight")
    plt.close()
