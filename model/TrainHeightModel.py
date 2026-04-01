"""
脚本用途：第四问(1)综合路径。
训练随机森林模型,数据拟合效果图
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from joblib import dump

from tool.Tool import read_grid, get_model

# 设置matplotlib支持中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文

# 获取当前文件的上级目录作为基础路径
BASE_DIR = Path(__file__).resolve().parent  # 获取项目根目录路径

# 设置输入数据文件路径
path = BASE_DIR.parent / "data" / "data.xlsx"  # Excel数据文件路径
# 调用函数读取网格数据
x, y, Z = read_grid(path)  # 获取坐标和深度数据

# 将x坐标从海里转换为米
x = x * 1852  # 1海里 = 1852米
# 将y坐标从海里转换为米
y = y * 1852  # 1海里 = 1852米

# 初始化数据点列表
data = []  # 存储所有数据点的坐标和深度
# 遍历所有y坐标
for j in range(len(y)):  # 对每个y坐标进行迭代
    # 遍历所有x坐标
    for i in range(len(x)):  # 对每个x坐标进行迭代
        # 创建包含x、y坐标和深度的数据点
        t = [x[i], y[j], Z[j][i]]  # [x坐标, y坐标, 深度值]
        # 将数据点添加到列表中
        data.append(t)  # 添加到数据列表
# 将数据列表转换为numpy数组
data = np.array(data)  # 转换为numpy数组便于处理
# 打印数据数组的形状
print(data.shape)

# 定义创建和训练随机森林模型的函数


# 使用数据训练深度预测模型
rf_height = get_model(data)  # 训练随机森林模型
# 测试模型预测功能
print(rf_height.predict([[30, 30]]))  # 预测坐标(30,30)处的深度
# 将训练好的模型保存到文件
dump(rf_height, str(BASE_DIR.parent / "data" / "height_random_forest_model.pkl"))  # 保存模型为pkl文件
