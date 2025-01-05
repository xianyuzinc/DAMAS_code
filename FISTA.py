import numpy as np
from scipy.fft import fft2, ifft2, fftshift
from time import time
from dataclasses import dataclass

@dataclass
class FISTAInfo:
    """存储FISTA算法的信息"""
    obj: list  # 目标函数值
    time: float  # 总运行时间

def objective_function(PSF, b, x, Fps, FpsT):
    """
    计算目标函数值和梯度
    
    参数:
    PSF: ndarray - 点扩散函数
    b: ndarray - 波束形成图
    x: ndarray - 当前解
    Fps: ndarray - PSF的FFT
    FpsT: ndarray - PSF转置的FFT
    
    返回:
    tuple - (目标函数值, 梯度)
    """
    # 计算 Ax
    Ax = fftshift(ifft2(fft2(x) * Fps))
    
    # 计算目标函数值: 1/2 * ||Ax - b||^2
    f = 0.5 * np.sum((Ax - b) ** 2)
    
    # 计算梯度: A^T(Ax - b)
    grad = fftshift(ifft2(fft2(Ax - b) * FpsT))
    
    return f, np.real(grad)

def estimate_lipschitz(PSF, Fps):
    """
    使用幂迭代法估计Lipschitz常数
    
    参数:
    PSF: ndarray - 点扩散函数
    Fps: ndarray - PSF的FFT
    
    返回:
    float - Lipschitz常数
    """
    x = np.random.rand(*PSF.shape)
    for _ in range(10):
        x = fftshift(ifft2(fft2(x) * Fps))
        x = x / np.linalg.norm(x, 'fro')
    
    return np.linalg.norm(x, 'fro') ** 2

def process_fista(PSF, b, x0, max_iterations=1000):
    """
    FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)
    
    参数:
    PSF: ndarray - 点扩散函数
    b: ndarray - 波束形成图
    x0: ndarray - 初始解
    max_iterations: int - 最大迭代次数
    
    返回:
    tuple - (最终解, 算法信息)
    """
    start_time = time()
    
    # 初始化变量
    x = x0.copy()
    x_old = x.copy()
    y = x.copy()
    t = 1
    
    # 预计算PSF的FFT
    Fps = fft2(PSF)
    FpsT = fft2(np.rot90(PSF, 2))
    
    # 计算Lipschitz常数
    L = estimate_lipschitz(PSF, Fps)
    
    # 创建函数句柄
    fgx = lambda x: objective_function(PSF, b, x, Fps, FpsT)
    
    # 计算初始目标函数值
    fy, grad_y = fgx(y)
    obj_values = [fy]
    
    # FISTA迭代
    for iteration in range(max_iterations):
        # 更新x
        x = np.maximum(0, y - (1/L) * grad_y)
        
        # 更新t和y
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        y = x + ((t - 1) / t_new) * (x - x_old)
        
        # 计算新的目标函数值和梯度
        fy, grad_y = fgx(y)
        obj_values.append(fy)
        
        # 更新变量
        x_old = x.copy()
        t = t_new
    
    # 创建信息结构
    info = FISTAInfo(
        obj=obj_values,
        time=time() - start_time
    )
    
    return x, info 