import numpy as np
from kernel_ridge import KernelRidge, linear_kernel, rbf_kernel
from sklearn.metrics import mean_squared_error

def test_linear_kernel():
    """测试线性核函数"""
    X = np.array([[1, 2], [3, 4]])
    K = linear_kernel(X)
    expected = np.array([[5, 11], [11, 25]])  # 1*1 + 2*2 = 5, 1*3 + 2*4 = 11, 等
    np.testing.assert_allclose(K, expected, rtol=1e-6)

def test_rbf_kernel():
    """测试RBF核函数"""
    X = np.array([[0, 0], [1, 1]])
    K = rbf_kernel(X, gamma=1.0)
    # 两个相同样本的距离为0，核值为1
    # 两个不同样本的距离平方为2，核值为exp(-2)
    expected = np.array([[1.0, np.exp(-2)], [np.exp(-2), 1.0]])
    np.testing.assert_allclose(K, expected, rtol=1e-6)

def test_linear_regression():
    """测试核岭回归在线性数据上的表现"""
    # 生成线性数据 y = 2x + 3 + 噪声
    np.random.seed(42)
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = 2 * X.ravel() + 3 + 0.5 * np.random.randn(100)
    
    # 使用线性核的核岭回归（等价于普通岭回归）
    kr = KernelRidge(kernel='linear', alpha=0.1)
    kr.fit(X, y)
    y_pred = kr.predict(X)
    
    # 确保MSE足够小
    mse = mean_squared_error(y, y_pred)
    assert mse < 0.5
    
    # 检查模型参数是否接近真实值
    # 对于线性核，我们可以从对偶系数中恢复出权重
    # 这只是一个近似检查，因为有正则化
    K = linear_kernel(X)
    weights = np.dot(K, kr.dual_coef_)
    slope = np.mean((weights[1:] - weights[:-1]) / (X[1:] - X[:-1]))
    assert np.abs(slope - 2) < 0.2
    assert np.abs(np.mean(weights - 2 * X.ravel()) - 3) < 0.5

def test_nonlinear_regression():
    """测试核岭回归在非线性数据上的表现"""
    # 生成非线性数据 y = sin(x) + 噪声
    np.random.seed(42)
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = np.sin(X).ravel() + 0.1 * np.random.randn(100)
    
    # 使用RBF核的核岭回归
    kr = KernelRidge(kernel='rbf', alpha=0.01, gamma=0.5)
    kr.fit(X, y)
    y_pred = kr.predict(X)
    
    # 确保MSE足够小
    mse = mean_squared_error(y, y_pred)
    assert mse < 0.02

def test_different_kernels():
    """测试不同核函数的输出是否不同"""
    # 生成非线性数据
    np.random.seed(42)
    X = np.linspace(0, 10, 50).reshape(-1, 1)
    y = np.sin(X).ravel() + 0.1 * np.random.randn(50)
    X_test = np.array([[5.0]])  # 测试点
    
    # 使用不同核函数
    kr_linear = KernelRidge(kernel='linear', alpha=0.1)
    kr_rbf = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.5)
    kr_poly = KernelRidge(kernel='poly', alpha=0.1, degree=3, gamma=0.1)
    
    kr_linear.fit(X, y)
    kr_rbf.fit(X, y)
    kr_poly.fit(X, y)
    
    # 不同核函数的预测结果应该不同
    pred_linear = kr_linear.predict(X_test)
    pred_rbf = kr_rbf.predict(X_test)
    pred_poly = kr_poly.predict(X_test)
    
    assert not np.isclose(pred_linear, pred_rbf, rtol=1e-3)
    assert not np.isclose(pred_linear, pred_poly, rtol=1e-3)

def test_invalid_parameters():
    """测试无效参数是否会抛出错误"""
    # 测试负的alpha值
    try:
        KernelRidge(alpha=-0.1)
        assert False, "应该拒绝负的alpha值"
    except ValueError:
        pass
    
    # 测试不存在的核函数
    try:
        KernelRidge(kernel='invalid')
        assert False, "应该拒绝不存在的核函数"
    except ValueError:
        pass
    
    # 测试训练前预测
    kr = KernelRidge()
    try:
        kr.predict(np.array([[1, 2]]))
        assert False, "训练前预测应该失败"
    except RuntimeError:
        pass
    
    # 测试特征数量不匹配
    kr = KernelRidge()
    kr.fit(np.array([[1, 2]]), np.array([3]))
    try:
        kr.predict(np.array([[1]]))  # 特征数量不匹配
        assert False, "特征数量不匹配应该失败"
    except ValueError:
        pass
