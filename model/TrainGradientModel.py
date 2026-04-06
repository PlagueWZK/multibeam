"""
脚本用途：第四问(2)验证。
主要包含数据处理、计算与可视化步骤。
"""

from pathlib import Path

import numpy as np

from tool.Tool import read_grid, get_model

# 获取当前文件的上级目录作为基础路径
BASE_DIR = Path(__file__).resolve().parent  # 获取项目根目录路径

# 设置输入数据文件路径
path = BASE_DIR.parent / "data" / "data.xlsx"  # Excel数据文件路径
# 调用函数读取网格数据（只需要深度数据Z）
_, _, Z = read_grid(path)  # 获取深度数据


def compute_normalized_gradient_vectorized(array):
    """
    向量化实现：计算归一化梯度

    使用 np.gradient 进行中心差分计算，自动处理边界条件
    相比原始的双重 for 循环实现，性能提升 10x+

    参数:
        array: 二维数组（深度矩阵）

    返回:
        normalized_gradient_x: 归一化后的 x 方向梯度（列方向）
        normalized_gradient_y: 归一化后的 y 方向梯度（行方向）
    """
    # np.gradient 返回 (y方向梯度, x方向梯度)
    # 注意：对于二维数组，gradient 返回顺序是 (row_gradient, col_gradient)
    gradient_y, gradient_x = np.gradient(array)

    # 计算梯度幅值（模长）
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # 避免除零：创建掩码，只对非零幅值进行归一化
    nonzero_mask = magnitude > 0

    # 初始化归一化梯度数组
    normalized_gradient_x = np.zeros_like(gradient_x)
    normalized_gradient_y = np.zeros_like(gradient_y)

    # 仅对非零幅值点进行归一化
    normalized_gradient_x[nonzero_mask] = (
        gradient_x[nonzero_mask] / magnitude[nonzero_mask]
    )
    normalized_gradient_y[nonzero_mask] = (
        gradient_y[nonzero_mask] / magnitude[nonzero_mask]
    )

    return normalized_gradient_x, normalized_gradient_y


# 保留原函数名以保持向后兼容（使用向量化实现）
def compute_normalized_gradient(array):
    """向后兼容接口，内部调用向量化实现"""
    return compute_normalized_gradient_vectorized(array)


# 使用深度数据作为测试数组
data_array = Z  # 将深度数据赋值给array变量
# 计算归一化梯度
normalized_gx, normalized_gy = compute_normalized_gradient(
    data_array
)  # 调用函数计算归一化梯度
# 打印x方向梯度数组的形状
print("Gradient in x direction:\n", normalized_gx.shape)
# 打印y方向梯度数组
print("Gradient in y direction:\n", normalized_gy)

# 构造x坐标数组（0到4海里，201个点）
x = np.linspace(0, 4 * 1852, 201)  # 生成x坐标数组，单位为米
# 构造y坐标数组（0到5海里，251个点）
y = np.linspace(0, 5 * 1852, 251)  # 生成y坐标数组，单位为米
# 打印x坐标数组
print(x)
# 打印x坐标数组的形状
print(x.shape)
# 打印y坐标数组的形状
print(y.shape)
# 打印深度数组的形状
print(Z.shape)
# 初始化x方向梯度数据列表
gx = []  # 存储x方向梯度的坐标和值
# 遍历所有y坐标
for j in range(len(y)):  # 对每个y坐标进行迭代
    # 遍历所有x坐标
    for i in range(len(x)):  # 对每个x坐标进行迭代
        # 创建包含x、y坐标和x方向梯度的数据点
        t = [x[i], y[j], normalized_gx[j][i]]  # [x坐标, y坐标, x方向梯度值]
        # 将数据点添加到列表中
        gx.append(t)  # 添加到x方向梯度数据列表
# 将x方向梯度数据列表转换为numpy数组
gx = np.array(gx)  # 转换为numpy数组便于处理
# 初始化y方向梯度数据列表
gy = []  # 存储y方向梯度的坐标和值
# 遍历所有y坐标
for j in range(len(y)):  # 对每个y坐标进行迭代
    # 遍历所有x坐标
    for i in range(len(x)):  # 对每个x坐标进行迭代
        # 创建包含x、y坐标和y方向梯度的数据点
        t = [x[i], y[j], normalized_gy[j][i]]  # [x坐标, y坐标, y方向梯度值]
        # 将数据点添加到列表中
        gy.append(t)  # 添加到y方向梯度数据列表
# 将y方向梯度数据列表转换为numpy数组
gy = np.array(gy)  # 转换为numpy数组便于处理

from joblib import dump

# 使用x方向梯度数据训练模型
rf_gx = get_model(gx)  # 训练x方向梯度预测模型,预测x梯度归一化的模型
# 测试x方向梯度模型预测功能
print(rf_gx.predict([[30, 30]]))  # 预测坐标(30,30)处的x方向梯度
# 将训练好的x方向梯度模型保存到文件
dump(
    rf_gx, str(BASE_DIR.parent / "data" / "gx_random_forest_model.pkl")
)  # 保存模型为pkl文件
# 使用y方向梯度数据训练模型
rf_gy = get_model(gy)  # 训练y方向梯度预测模型,预测y梯度归一化的模型
# 测试y方向梯度模型预测功能
print(rf_gy.predict([[30, 30]]))  # 预测坐标(30,30)处的y方向梯度
# 将训练好的y方向梯度模型保存到文件
dump(
    rf_gy, str(BASE_DIR.parent / "data" / "gy_random_forest_model.pkl")
)  # 保存模型为pkl文件
