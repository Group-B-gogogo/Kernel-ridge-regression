import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("D:\\german\\Kernel-ridge-regression")
from kernel_ridge import KernelRidge

# 设置随机种子，确保结果可复现
np.random.seed(1002)

num_samples = 1000
gamma_test = 0.1
alpha_test = 1e-4
def generate_nonlinear_data():
    """生成非线性数据用于演示"""

    # 生成输入特征
    X = np.linspace(0, 20, num_samples).reshape(-1, 1)
    
    # 生成带有噪声的非线性目标值
    y = np.sin(X).ravel() + 0.3 * np.random.randn(num_samples)
    
    return X, y

def plot_results(X, y, X_test, y_pred, y_true):
    """绘制训练数据、预测结果和真实曲线"""
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, c='blue', alpha=0.45, s=15, label='Training data')
    plt.plot(X_test, y_pred, 'r-', linewidth=2, label=f'Nuclear ridge regression prediction,alpha={alpha_test}, gamma={gamma_test}')
    plt.plot(X_test, y_true, 'g--', linewidth=2, label='True value curve')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Example of Nonlinear Fitting with Kernel Ridge Regression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def compare_kernels(X, y, X_test, y_true):
    """比较不同核函数的效果"""
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    plt.figure(figsize=(12, 10))
    
    for i, kernel in enumerate(kernels, 1):
        # 初始化并训练模型
        if kernel == 'poly':
            kr = KernelRidge(kernel=kernel, alpha=alpha_test, degree=3, gamma=gamma_test)
        else:
            kr = KernelRidge(kernel=kernel, alpha=alpha_test, gamma=gamma_test)
        
        kr.fit(X, y)
        y_pred = kr.predict(X_test)
        
        # 绘制结果
        plt.subplot(2, 2, i)
        plt.scatter(X, y, c='blue', alpha=0.5, s=15, label='training data')
        plt.plot(X_test, y_pred, 'r-', linewidth=2, label='Nuclear ridge regression prediction')
        plt.plot(X_test, y_true, 'g--', linewidth=2, label='True value curve')
        plt.title(f'{kernel} kernel')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    # 生成数据
    X, y = generate_nonlinear_data()
    
    # 创建测试数据
    X_test = np.linspace(0, 20, num_samples).reshape(-1, 1)
    y_true = np.sin(X_test).ravel()  # 真实函数
    
    # 使用RBF核的核岭回归
    kr = KernelRidge(kernel='rbf', alpha=alpha_test, gamma=gamma_test)
    kr.fit(X, y)
    y_pred = kr.predict(X_test)
    
    # 打印模型信息
    print("The trained model:", kr)
    
    # 绘制结果
    plot_results(X, y, X_test, y_pred, y_true)
    
    # 比较不同核函数
    #compare_kernels(X, y, X_test, y_true)

if __name__ == "__main__":
    main()
