import numpy as np


def linear_kernel(X, Y=None, **kwargs):
    """
    线性核函数
    K(X, Y) = X @ Y.T
    
    参数:
        X: 形状为 (n_samples_X, n_features) 的数组
        Y: 形状为 (n_samples_Y, n_features) 的数组，默认为 None
            如果为 None，则 Y = X
        **kwargs: 其他参数（不使用）
    
    返回:
        K: 形状为 (n_samples_X, n_samples_Y) 的核矩阵
    """
    if Y is None:
        Y = X
    return np.dot(X, Y.T)


def polynomial_kernel(X, Y=None, degree=3, coef0=0.0, gamma=None,** kwargs):
    """
    多项式核函数
    K(X, Y) = (gamma * X @ Y.T + coef0) ^ degree
    
    参数:
        X: 形状为 (n_samples_X, n_features) 的数组
        Y: 形状为 (n_samples_Y, n_features) 的数组，默认为 None
            如果为 None，则 Y = X
        degree: 多项式的阶数
        coef0: 独立项
        gamma: 核系数，默认为 None (此时 gamma = 1/n_features)
        **kwargs: 其他参数（不使用）
    
    返回:
        K: 形状为 (n_samples_X, n_samples_Y) 的核矩阵
    """
    if Y is None:
        Y = X
    
    n_features = X.shape[1]
    if gamma is None:
        gamma = 1.0 / n_features
    
    K = np.dot(X, Y.T)
    K *= gamma
    K += coef0
    K **= degree
    return K


def rbf_kernel(X, Y=None, gamma=None,** kwargs):
    """
    径向基函数核（RBF核）
    K(X, Y) = exp(-gamma * ||X - Y||^2)
    
    参数:
        X: 形状为 (n_samples_X, n_features) 的数组
        Y: 形状为 (n_samples_Y, n_features) 的数组，默认为 None
            如果为 None，则 Y = X
        gamma: 核系数，默认为 None (此时 gamma = 1/n_features)
        **kwargs: 其他参数（不使用）
    
    返回:
        K: 形状为 (n_samples_X, n_samples_Y) 的核矩阵
    """
    if Y is None:
        Y = X
    
    n_samples_X, n_features = X.shape
    
    if gamma is None:
        gamma = 1.0 / n_features
    
    # # 计算||X - Y||^2
    # # 使用广播机制优化计算
    # K = np.sum(X**2, axis=1)[:, np.newaxis]
    # K += np.sum(Y**2, axis=1)[:, np.newaxis]
    # K -= 2 * np.dot(X, Y.T)
    
    # # 应用RBF公式
    # K *= -gamma
    # np.exp(K, out=K)  # 原地计算指数，节省内存
    K = np.exp(-gamma * (np.sum(X**2, axis=1)[:, np.newaxis] + np.sum(Y**2, axis=1) - 2 * np.dot(X, Y.T)))
    
    return K


def sigmoid_kernel(X, Y=None, coef0=0.0, gamma=None, **kwargs):
    """
    Sigmoid核函数
    K(X, Y) = tanh(gamma * X @ Y.T + coef0)
    
    参数:
        X: 形状为 (n_samples_X, n_features) 的数组
        Y: 形状为 (n_samples_Y, n_features) 的数组，默认为 None
            如果为 None，则 Y = X
        coef0: 独立项
        gamma: 核系数，默认为 None (此时 gamma = 1/n_features)
        **kwargs: 其他参数（不使用）
    
    返回:
        K: 形状为 (n_samples_X, n_samples_Y) 的核矩阵
    """
    if Y is None:
        Y = X
    
    n_features = X.shape[1]
    if gamma is None:
        gamma = 1.0 / n_features
    
    K = np.dot(X, Y.T)
    K *= gamma
    K += coef0
    np.tanh(K, out=K)  # 原地计算双曲正切，节省内存
    
    return K


# 核函数映射字典
KERNEL_FUNCTIONS = {
    'linear': linear_kernel,
    'poly': polynomial_kernel,
    'rbf': rbf_kernel,
    'sigmoid': sigmoid_kernel
}
