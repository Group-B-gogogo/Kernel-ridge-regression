import numpy as np
from .kernels import KERNEL_FUNCTIONS


class KernelRidge:
    """
    核岭回归模型
    
    核岭回归是一种结合了核方法和岭回归的非线性回归技术。
    它通过核函数将输入空间映射到高维特征空间，从而能够处理非线性关系，
    同时通过L2正则化来防止过拟合。
    """
    
    def __init__(self, kernel='linear', alpha=1.0, degree=3, gamma=None, coef0=0.0):
        """
        初始化核岭回归模型
        
        参数:
            kernel: 核函数类型，可选值为 'linear', 'poly', 'rbf', 'sigmoid' 或自定义函数
            alpha: 正则化参数，必须为正数，控制正则化强度，值越大正则化越强
            degree: 多项式核的阶数，仅在 kernel='poly' 时有效
            gamma: rbf, poly 和 sigmoid 核的核系数，默认为 None (此时将使用 1/n_features)
            coef0: 多项式核和 sigmoid 核的独立项，仅在 kernel='poly' 或 'sigmoid' 时有效
        """
        self.kernel = kernel
        self.alpha = alpha
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        
        # 检查 alpha 是否为正数
        if alpha <= 0:
            raise ValueError("正则化参数 alpha 必须为正数")
        
        # 获取核函数
        if callable(kernel):
            self.kernel_func = kernel
        else:
            if kernel not in KERNEL_FUNCTIONS:
                raise ValueError(f"不支持的核函数: {kernel}")
            self.kernel_func = KERNEL_FUNCTIONS[kernel]
        
        # 模型参数
        self.X_fit_ = None  # 训练样本
        self.dual_coef_ = None  # 对偶系数
    
    def fit(self, X, y):
        """
        训练核岭回归模型
        
        参数:
            X: 形状为 (n_samples, n_features) 的训练样本
            y: 形状为 (n_samples,) 的目标值
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()  # 确保 y 是一维数组
        
        if X.ndim != 2:
            raise ValueError("X 必须是二维数组 (n_samples, n_features)")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("X 和 y 的样本数量必须一致")
        
        # 存储训练样本
        self.X_fit_ = X
        
        # 计算核矩阵 K
        kernel_params = {
            'gamma': self.gamma,
            'degree': self.degree,
            'coef0': self.coef0
        }
        K = self.kernel_func(X, X, **kernel_params)
        
        # 求解对偶问题: (K + n_samples * alpha * I) * alpha = y
        n_samples = X.shape[0]
        K_reg = K + n_samples * self.alpha * np.eye(n_samples)
        
        # 求解线性方程组得到对偶系数
        self.dual_coef_ = np.linalg.solve(K_reg, y)
        
        return self
    
    def predict(self, X):
        """
        预测新样本
        
        参数:
            X: 形状为 (n_samples, n_features) 的测试样本
        
        返回:
            y_pred: 形状为 (n_samples,) 的预测值
        """
        if self.X_fit_ is None or self.dual_coef_ is None:
            raise RuntimeError("模型尚未训练，请先调用 fit 方法")
        
        X = np.asarray(X)
        
        if X.ndim != 2:
            raise ValueError("X 必须是二维数组 (n_samples, n_features)")
        
        if X.shape[1] != self.X_fit_.shape[1]:
            raise ValueError(f"X 的特征数量必须为 {self.X_fit_.shape[1]}")
        
        # 计算测试样本与训练样本之间的核矩阵
        kernel_params = {
            'gamma': self.gamma,
            'degree': self.degree,
            'coef0': self.coef0
        }
        K = self.kernel_func(X, self.X_fit_,** kernel_params)
        
        # 预测: y_pred = K @ dual_coef_
        y_pred = np.dot(K, self.dual_coef_)
        
        return y_pred
    
    def __repr__(self):
        """返回模型的字符串表示"""
        params = [
            f"kernel='{self.kernel}'",
            f"alpha={self.alpha}",
        ]
        
        if self.kernel == 'poly':
            params.append(f"degree={self.degree}")
            params.append(f"coef0={self.coef0}")
        
        if self.kernel in ['poly', 'rbf', 'sigmoid']:
            params.append(f"gamma={self.gamma}")
        
        return f"KernelRidge({', '.join(params)})"
