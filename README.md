# Kernel Ridge Regression

一个高效实现的核岭回归算法库，支持多种核函数，适用于处理非线性回归问题。

## 简介

核岭回归(Kernel Ridge Regression)是一种结合了核方法和岭回归的机器学习算法。它通过核函数将输入空间映射到高维特征空间，从而能够处理非线性关系，同时通过L2正则化来防止过拟合。

## 安装
# 克隆仓库
git clone https://github.com/yourusername/kernel-ridge-regression.git
cd kernel-ridge-regression

# 安装依赖
pip install -r requirements.txt

# 安装库
pip install .
## 支持的核函数

- 线性核 (Linear Kernel)
- 多项式核 (Polynomial Kernel)
- 径向基函数核 (RBF Kernel)
-  sigmoid核 (Sigmoid Kernel)
- 自定义核函数

## 使用示例
from kernel_ridge import KernelRidge
import numpy as np
import matplotlib.pyplot as plt

# 生成示例数据
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X).ravel() + 0.5 * np.random.randn(100)

# 初始化模型
kr = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.5)

# 拟合模型
kr.fit(X, y)

# 预测
X_test = np.linspace(0, 10, 200).reshape(-1, 1)
y_pred = kr.predict(X_test)

# 可视化结果
plt.scatter(X, y, label='训练数据')
plt.plot(X_test, y_pred, 'r-', label='预测曲线')
plt.plot(X_test, np.sin(X_test), 'g--', label='真实曲线')
plt.legend()
plt.show()
## API 文档

### KernelRidge 类

#### 初始化参数

- `kernel`: 核函数类型，可选值为 'linear', 'poly', 'rbf', 'sigmoid' 或自定义函数，默认为 'linear'
- `alpha`: 正则化参数，必须为正数，默认为 1.0
- `degree`: 多项式核的阶数，默认为 3
- `gamma`: rbf, poly 和 sigmoid 核的核系数，默认为 None (此时将使用 1/n_features)
- `coef0`: 多项式核和 sigmoid 核的独立项，默认为 0.0

#### 方法

- `fit(X, y)`: 训练模型
  - `X`: 形状为 (n_samples, n_features) 的训练样本
  - `y`: 形状为 (n_samples,) 的目标值
  
- `predict(X)`: 预测新样本
  - `X`: 形状为 (n_samples, n_features) 的测试样本
  - 返回: 形状为 (n_samples,) 的预测值

## 测试
pytest tests/
## 许可证

本项目基于MIT许可证开源 - 详见LICENSE文件
