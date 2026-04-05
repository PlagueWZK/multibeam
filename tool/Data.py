import numpy as np

from tool.Tool import load_model

height_rf, gx_rf, gy_rf = load_model()


def get_height(x, y):  # 输入x、y坐标，返回该位置的深度
    return float(height_rf.predict([[x, y]])[0])  # 使用深度预测模型预测深度


# 定义获取指定位置x方向梯度的函数
def get_gx(x, y):  # 输入x、y坐标，返回该位置的x方向梯度
    return float(gx_rf.predict([[x, y]])[0])  # 使用x方向梯度预测模型预测梯度


# 定义获取指定位置y方向梯度的函数
def get_gy(x, y):  # 输入x、y坐标，返回该位置的y方向梯度
    return float(gy_rf.predict([[x, y]])[0])  # 使用y方向梯度预测模型预测梯度


def get_alpha(x, y, microstep=35):  # 输入x、y坐标，返回该位置的坡度角
    # 使用与原始网格分辨率一致的步长，避免随机森林在极小位移下输出不变
    # step = 10 # 约等于 1 海里 / 50，对应原始数据网格间距（米）
    # 计算沿梯度方向前进一步后的坐标
    tx1 = x + microstep * get_gx(x, y)  # 沿x方向梯度前进
    ty1 = y + microstep * get_gy(x, y)  # 沿y方向梯度前进
    # 获取前进后位置的深度
    h1 = get_height(tx1, ty1)  # 前进后的深度
    # 计算沿梯度方向后退一步后的坐标
    tx2 = x - microstep * get_gx(x, y)  # 沿x方向梯度后退
    ty2 = y - microstep * get_gy(x, y)  # 沿y方向梯度后退
    # 获取后退后位置的深度
    h2 = get_height(tx2, ty2)  # 后退后的深度
    # 计算坡度角（使用反正切函数）
    return float(
        np.arctan((abs(h1 - h2)) / (2 * microstep)) * 180 / np.pi
    )  # 坡度角计算（弧度转角度）


def sin(a):  # 输入角度，返回正弦值
    return np.sin(np.radians(a))  # 将角度转换为弧度后计算正弦值


# 定义余弦函数（角度转弧度）
def cos(a):  # 输入角度，返回余弦值
    return np.cos(np.radians(a))  # 将角度转换为弧度后计算余弦值


# 定义正切函数（角度转弧度）
def tan(a):  # 输入角度，返回正切值
    return np.tan(np.radians(a))  # 将角度转换为弧度后计算正切值


def get_w_left(x, y, theta=120):  # 输入x、y坐标，返回该位置的左侧覆盖宽度
    # 获取该位置的深度
    D = get_height(x, y)  # 获取深度
    # 获取该位置的坡度角
    alpha = get_alpha(x, y)  # 获取坡度角
    # 计算左侧覆盖宽度
    return (D * sin(theta / 2) / sin(90 - theta / 2 - alpha)) * cos(
        alpha
    )  # 左侧覆盖宽度计算公式(转化为水平标量)


# 定义计算指定位置右侧覆盖宽度的函数
def get_w_right(x, y, theta=120):  # 输入x、y坐标，返回该位置的右侧覆盖宽度
    # 获取该位置的深度
    D = get_height(x, y)  # 获取深度
    # 获取该位置的坡度角
    alpha = get_alpha(x, y)  # 获取坡度角
    # 计算右侧覆盖宽度
    return (D * sin(theta / 2) / sin(90 - theta / 2 + alpha)) * cos(
        alpha
    )  # 右侧覆盖宽度计算公式(转化为水平标量)


def forward_direction(gx, gy):  # 输入梯度分量，返回垂直于梯度的前进方向
    return -gy, gx  # 返回垂直于梯度方向的向量


def figure_length(line):
    """输入测线路径点 [x, y, ...]，返回总长度（矢量化实现）"""
    return float(np.sum(np.sqrt(np.diff(line[:, 0]) ** 2 + np.diff(line[:, 1]) ** 2)))


def figure_width(line):
    """
    输入测线路径点 [x, y, w_total]，返回覆盖总面积。
    要求 line 第 3 列已预计算覆盖宽度 (w_left + w_right)，
    避免逐点调用模型造成性能退化。
    """
    dists = np.sqrt(np.diff(line[:, 0]) ** 2 + np.diff(line[:, 1]) ** 2)
    return float(np.sum(dists * line[:-1, 2]))

