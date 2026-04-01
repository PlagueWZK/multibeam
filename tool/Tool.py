from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent


def load_model():
    height_rf = load(str(BASE_DIR.parent / "data" / "height_random_forest_model.pkl"))
    gx_rf = load(str(BASE_DIR.parent / "data" / "gx_random_forest_model.pkl"))
    gy_rf = load(str(BASE_DIR.parent / "data" / "gy_random_forest_model.pkl"))
    return height_rf, gx_rf, gy_rf


def read_grid(data_path):  # 输入Excel文件路径，返回x、y坐标和深度数据Z
    # 读取Excel文件，不使用表头
    df = pd.read_excel(data_path, header=None)  # 读取Excel文件
    # 提取x坐标数据（第2行，从第3列开始）
    x = pd.to_numeric(df.iloc[1, 2:], errors="coerce").to_numpy()  # 转换为数值类型
    # 提取y坐标数据（从第3行开始，第2列）
    y = pd.to_numeric(df.iloc[2:, 1], errors="coerce").to_numpy()  # 转换为数值类型
    # 提取深度数据Z（从第3行第3列开始的矩形区域）
    Z = df.iloc[2:, 2:].apply(pd.to_numeric, errors="coerce").to_numpy()  # 转换为数值类型

    # 创建x坐标的有效数据掩码
    x_mask = ~np.isnan(x)  # 非NaN值为True
    # 创建y坐标的有效数据掩码
    y_mask = ~np.isnan(y)  # 非NaN值为True
    # 过滤掉NaN值的x坐标
    x = x[x_mask]  # 保留有效的x坐标
    # 过滤掉NaN值的y坐标
    y = y[y_mask]  # 保留有效的y坐标
    # 根据掩码过滤深度数据Z
    Z = Z[np.ix_(y_mask, x_mask)]  # 保留有效坐标对应的深度数据
    # 返回处理后的坐标和深度数据
    return x, y, Z


def get_model(data):  # 输入数据，返回训练好的随机森林模型
    # 提取特征数据（x、y坐标）
    X = data[:, 0:2]  # 前两列作为特征
    # 提取目标数据（深度值）
    y = data[:, 2]  # 第三列作为目标值

    # 将数据分割为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80%训练，20%测试

    # 创建随机森林回归模型
    rf = RandomForestRegressor(n_estimators=100, random_state=42)  # 100棵决策树

    # 使用训练数据训练模型
    rf.fit(X_train, y_train.ravel())  # 训练随机森林模型

    # 使用测试数据进行预测
    y_pred = rf.predict(X_test)  # 预测测试集结果

    # 计算模型评估分数
    score = rf.score(X_test, y_test)  # 计算R²分数
    # 打印模型评估结果
    print(f"R^2 Score: {score}")

    # 创建3D图形对象
    fig = plt.figure()  # 创建图形
    # 添加3D子图
    ax = fig.add_subplot(111, projection='3d')  # 创建3D坐标轴
    # 绘制真实值散点图
    ax.scatter(X_test[:, 0], X_test[:, 1], y_test, label='True', marker='^')  # 真实值用三角形标记
    # 绘制预测值散点图
    ax.scatter(X_test[:, 0], X_test[:, 1], y_pred, label='Predicted', marker='*')  # 预测值用星形标记
    # 设置x轴标签
    ax.set_xlabel('Feature 1')  # x轴标签
    # 设置y轴标签
    ax.set_ylabel('Feature 2')  # y轴标签
    # 设置z轴标签
    ax.set_zlabel('Target')  # z轴标签
    # 显示图例
    ax.legend()  # 显示图例
    # 显示图表
    plt.show()  # 显示3D图表
    return rf  # 返回训练好的模型
