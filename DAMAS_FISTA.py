import numpy as np
from scipy import linalg

def process_damas_fista(S, e, max_iterations=1000, tol=1e-6):
    """
    实现DAMAS-FISTA算法
    
    参数:
    S: ndarray - 波束形成结果
    e: ndarray - 导向矢量
    max_iterations: int - 最大迭代次数
    tol: float - 收敛阈值
    
    返回:
    ndarray - DAMAS-FISTA处理后的结果
    """
    # 取实部作为输入
    Y = np.real(S)
    M, N, P = e.shape
    
    # 重塑导向矢量
    ee = e.reshape(M*N, P)
    
    # 构建字典矩阵A (PSF)
    A = np.abs(ee @ ee.conj().T)**2 / P**2
    
    # 初始化变量
    b = Y.flatten()
    x = np.zeros_like(b)  # x0
    x_old = x.copy()
    y = x.copy()
    t = 1
    
    # 预计算 A^T @ A 和 A^T @ b
    ATA = A.T @ A
    ATb = A.T @ b
    
    # 计算Lipschitz常数
    L = np.real(linalg.eigvals(ATA)).max()
    
    # 计算初始梯度
    grad = ATA @ y - ATb
    
    # FISTA迭代
    for iteration in range(max_iterations):
        # 更新x
        x = np.maximum(0, y - (1/L) * grad)
        
        # 检查收敛性
        if iteration > 0:
            x_diff = np.linalg.norm(x - x_old) / np.linalg.norm(x_old)
            if x_diff < tol:
                print(f'DAMAS-FISTA在{iteration}次迭代后收敛')
                break
        
        # 更新步长
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        
        # 更新y
        y = x + ((t - 1) / t_new) * (x - x_old)
        
        # 重新计算梯度
        grad = ATA @ y - ATb
        x_old = x.copy()
        t = t_new
    
    if iteration == max_iterations - 1:
        print(f'DAMAS-FISTA达到最大迭代次数 ({max_iterations})')
    
    # 将结果重塑为原始维度
    return x.reshape(M, N) 