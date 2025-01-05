import numpy as np
from scipy.fft import fft2, ifft2, fftshift

def process_damas2(S, e, max_iterations=1000, tol=1e-6):
    """
    实现DAMAS2算法
    
    参数:
    S: ndarray - 波束形成结果
    e: ndarray - 导向矢量
    max_iterations: int - 最大迭代次数
    tol: float - 收敛阈值
    
    返回:
    ndarray - DAMAS2处理后的结果
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
    
    # 零填充
    pad_M = 2 * M
    pad_N = 2 * N
    
    Y_pad = np.zeros((pad_M, pad_N))
    Y_pad[M//2:M//2+M, N//2:N//2+N] = Y
    
    PSF_pad = np.zeros((pad_M, pad_N))
    PSF_pad[M//2:M//2+M, N//2:N//2+N] = PSF
    
    # 计算离散积分常数
    a = np.sum(PSF)
    
    # 设计高斯正则化滤波器
    k = 2 * np.pi * 10000  # 使用默认频率10kHz
    k_c = 1.2 * k
    psi = np.exp(-k**2 / (2 * k_c**2))
    
    # 预计算PSF的FFT
    fft_psf = fft2(PSF_pad)
    
    # 初始化结果
    x = np.zeros((pad_M, pad_N))
    
    # DAMAS2迭代
    for iteration in range(max_iterations):
        x_old = x.copy()
        
        # 计算残差向量
        r = fftshift(np.real(ifft2(fft2(x) * fft_psf * psi)))
        
        # 更新解
        x = np.maximum(0, x_old + (Y_pad - r) / a)
        
        # 检查收敛
        if iteration > 0:
            x_diff = np.linalg.norm(x - x_old) / np.linalg.norm(x_old)
            if x_diff < tol:
                print(f'DAMAS2在{iteration}次迭代后收敛')
                break
    
    if iteration == max_iterations - 1:
        print(f'DAMAS2达到最大迭代次数 ({max_iterations})')
    
    # 移除零填充
    result = x[M//2:M//2+M, N//2:N//2+N]
    
    return result 