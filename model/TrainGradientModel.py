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


# 定义计算归一化梯度的函数
def compute_normalized_gradient(array):  # 输入二维数组，返回归一化的x和y方向梯度
    # 获取数组的行数和列数
    m, n = array.shape  # m为行数，n为列数

    # 初始化x方向梯度数组
    gradient_x = np.zeros((m, n))  # 创建与输入数组同样大小的零数组
    # 初始化y方向梯度数组
    gradient_y = np.zeros((m, n))  # 创建与输入数组同样大小的零数组

    # 初始化归一化后的x方向梯度数组
    normalized_gradient_x = np.zeros((m, n))  # 创建与输入数组同样大小的零数组
    # 初始化归一化后的y方向梯度数组
    normalized_gradient_y = np.zeros((m, n))  # 创建与输入数组同样大小的零数组

    # 遍历数组的每一行
    for i in range(m):  # 对每一行进行迭代
        # 遍历数组的每一列
        for j in range(n):  # 对每一列进行迭代
            # 计算x方向的梯度（列方向）
            if j == 0:  # 如果是第一列
                # 使用前向差分
                gradient_x[i, j] = array[i, j + 1] - array[i, j]  # 前向差分公式
            elif j == n - 1:  # 如果是最后一列
                # 使用后向差分
                gradient_x[i, j] = array[i, j] - array[i, j - 1]  # 后向差分公式
            else:  # 如果是中间列
                # 使用中心差分
                gradient_x[i, j] = (array[i, j + 1] - array[i, j - 1]) / 2.0  # 中心差分公式

            # 计算y方向的梯度（行方向）
            if i == 0:  # 如果是第一行
                # 使用前向差分
                gradient_y[i, j] = array[i + 1, j] - array[i, j]  # 前向差分公式
            elif i == m - 1:  # 如果是最后一行
                # 使用后向差分
                gradient_y[i, j] = array[i, j] - array[i - 1, j]  # 后向差分公式
            else:  # 如果是中间行
                # 使用中心差分
                gradient_y[i, j] = (array[i + 1, j] - array[i - 1, j]) / 2.0  # 中心差分公式

            # 计算梯度的幅值（强度）模长
            magnitude = np.sqrt(gradient_x[i, j] ** 2 + gradient_y[i, j] ** 2)  # 梯度幅值公式

            # 进行归一化处理，避免除零错误
            if magnitude != 0:  # 如果梯度幅值不为零
                # 对x方向梯度进行归一化
                normalized_gradient_x[i, j] = gradient_x[i, j] / magnitude  # 归一化x方向梯度
                # 对y方向梯度进行归一化
                normalized_gradient_y[i, j] = gradient_y[i, j] / magnitude  # 归一化y方向梯度
            else:  # 如果梯度幅值为零
                # 设置归一化梯度为零
                normalized_gradient_x[i, j] = 0  # x方向归一化梯度为0
                # 设置归一化梯度为零
                normalized_gradient_y[i, j] = 0  # y方向归一化梯度为0

    # 返回归一化后的x和y方向梯度
    return normalized_gradient_x, normalized_gradient_y


# 使用深度数据作为测试数组
data_array = Z  # 将深度数据赋值给array变量
# 计算归一化梯度
normalized_gx, normalized_gy = compute_normalized_gradient(data_array)  # 调用函数计算归一化梯度
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
dump(rf_gx, str(BASE_DIR.parent / "data" / "gx_random_forest_model.pkl"))  # 保存模型为pkl文件
# 使用y方向梯度数据训练模型
rf_gy = get_model(gy)  # 训练y方向梯度预测模型,预测y梯度归一化的模型
# 测试y方向梯度模型预测功能
print(rf_gy.predict([[30, 30]]))  # 预测坐标(30,30)处的y方向梯度
# 将训练好的y方向梯度模型保存到文件
dump(rf_gy, str(BASE_DIR.parent / "data" / "gy_random_forest_model.pkl"))  # 保存模型为pkl文件
