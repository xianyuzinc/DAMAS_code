import numpy as np
from scipy.fft import fft2, ifft2, fftshift
from scipy import sparse

def dfista_gradient(fft_psf, das_map, y, D, lambda_param):
    """
    计算FFT-DFISTA的梯度
    
    参数:
    fft_psf: ndarray - PSF的FFT
    das_map: ndarray - 波束形成图
    y: ndarray - 当前解
    D: sparse matrix - 差分矩阵
    lambda_param: float - 正则化参数
    
    返回:
    ndarray - 梯度
    """
    # 数据项梯度
    Ay = fftshift(ifft2(fft2(y) * fft_psf))
    grad_data = fftshift(ifft2(fft2(Ay - das_map) * np.conj(fft_psf)))
    
    # 正则化项梯度 (TV梯度)
    Dy = D @ y.flatten()
    DTDy = (D.T @ Dy).reshape(y.shape)
    
    return np.real(grad_data) + lambda_param * DTDy

def dfista_lipschitz(PSF, fft_psf, D, lambda_param, n_iter=10):
    """
    估计Lipschitz常数
    
    参数:
    PSF: ndarray - 点扩散函数
    fft_psf: ndarray - PSF的FFT
    D: sparse matrix - 差分矩阵
    lambda_param: float - 正则化参数
    n_iter: int - 幂迭代次数
    
    返回:
    float - Lipschitz常数
    """
    x = np.random.rand(*PSF.shape)
    for _ in range(n_iter):
        # 数据项
        x_data = fftshift(ifft2(fft2(x) * fft_psf))
        x_data = x_data / np.linalg.norm(x_data, 'fro')
        
        # 正则化项
        x_reg = (D.T @ (D @ x.flatten())).reshape(x.shape)
        x_reg = x_reg / np.linalg.norm(x_reg, 'fro')
        
        x = x_data + lambda_param * x_reg
        x = x / np.linalg.norm(x, 'fro')
    
    return np.linalg.norm(x, 'fro') ** 2

def process_fft_dfista(S, e, lambda_param=0.1, max_iterations=1000, tol=1e-6):
    """
    实现FFT-DFISTA算法
    
    参数:
    S: ndarray - 波束形成结果
    e: ndarray - 导向矢量
    lambda_param: float - 正则化参数
    max_iterations: int - 最大迭代次数
    tol: float - 收敛阈值
    
    返回:
    ndarray - FFT-DFISTA处理后的结果
    """
    # 取实部作为输入
    Y = np.real(S)
    M, N, P = e.shape
    
    # 计算中心点的导向矢量
    center_x = M // 2
    center_y = N // 2
    g_center = e[center_x, center_y, :]
    
    # 计算shift-invariant PSF
    ee = e.reshape(M*N, P)
    PSF = np.abs(ee @ g_center.conj())**2 / P**2
    PSF = PSF.reshape(M, N)
    
    # 构建差分矩阵 - 修改这部分
    pad_size = 2 * M
    n_pixels = pad_size * pad_size  # 总像素数
    
    # 水平差分
    rows_h = []
    cols_h = []
    data_h = []
    for i in range(pad_size):
        for j in range(pad_size-1):
            idx = i * pad_size + j
            rows_h.append(i * (pad_size-1) + j)
            cols_h.append(idx)
            data_h.append(1)
            
            rows_h.append(i * (pad_size-1) + j)
            cols_h.append(idx + 1)
            data_h.append(-1)
    
    # 垂直差分
    rows_v = []
    cols_v = []
    data_v = []
    for i in range(pad_size-1):
        for j in range(pad_size):
            idx = i * pad_size + j
            rows_v.append(i * pad_size + j)
            cols_v.append(idx)
            data_v.append(1)
            
            rows_v.append(i * pad_size + j)
            cols_v.append(idx + pad_size)
            data_v.append(-1)
    
    # 组合水平和垂直差分
    rows = rows_h + rows_v
    cols = cols_h + cols_v
    data = data_h + data_v
    
    # 创建稀疏差分矩阵
    n_diffs = (pad_size * (pad_size-1) + pad_size * (pad_size-1))
    D = sparse.csr_matrix((data, (rows, cols)), shape=(n_diffs, n_pixels))
    
    # 零填充
    Y_pad = np.zeros((pad_size, pad_size))
    Y_pad[M//2:M//2+M, N//2:N//2+N] = Y
    
    PSF_pad = np.zeros((pad_size, pad_size))
    PSF_pad[M//2:M//2+M, N//2:N//2+N] = PSF
    
    # 预计算PSF的FFT
    fft_psf = fft2(PSF_pad)
    
    # 初始化变量
    x = np.zeros_like(Y_pad)
    x_old = x.copy()
    y = x.copy()
    t = 1
    
    # 计算Lipschitz常数
    L = dfista_lipschitz(PSF_pad, fft_psf, D, lambda_param)
    
    # 计算初始梯度
    grad = dfista_gradient(fft_psf, Y_pad, y, D, lambda_param)
    
    # FFT-DFISTA迭代
    for iteration in range(max_iterations):
        # 更新x
        x = np.maximum(0, y - (1/L) * grad)
        
        # 检查收敛性
        if iteration > 0:
            x_diff = np.linalg.norm(x - x_old) / np.linalg.norm(x_old)
            if x_diff < tol:
                print(f'FFT-DFISTA在{iteration}次迭代后收敛')
                break
        
        # 更新t和y
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        y = x + ((t - 1) / t_new) * (x - x_old)
        
        # 更新梯度
        grad = dfista_gradient(fft_psf, Y_pad, y, D, lambda_param)
        x_old = x.copy()
        t = t_new
    
    if iteration == max_iterations - 1:
        print(f'FFT-DFISTA达到最大迭代次数 ({max_iterations})')
    
    # 移除零填充
    result = x[M//2:M//2+M, N//2:N//2+N]
    
    return result 