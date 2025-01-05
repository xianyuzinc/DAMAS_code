# steering_vector.py

import numpy as np

def compute_steering_vector(x_pos, y_pos, z_pos, freq, c, scan_x, scan_y, scan_z):
    """
    计算阵列的导向矢量
    
    参数:
    x_pos: ndarray - 麦克风x坐标 [1×P]
    y_pos: ndarray - 麦克风y坐标 [1×P]
    z_pos: ndarray - 麦克风z坐标 [1×P]
    freq: float - 波频率 [Hz]
    c: float - 声速 [m/s]
    scan_x: ndarray - 扫描平面x坐标 [M×N]
    scan_y: ndarray - 扫描平面y坐标 [M×N]
    scan_z: ndarray - 扫描平面z坐标 [M×N]
    
    返回:
    ndarray - 导向矢量 [M×N×P]
    """
    # 输入检查
    if not isinstance(x_pos, np.ndarray):
        raise ValueError("X坐标必须是numpy数组")
    if not isinstance(y_pos, np.ndarray):
        raise ValueError("Y坐标必须是numpy数组")
    if not isinstance(z_pos, np.ndarray):
        raise ValueError("Z坐标必须是numpy数组")

    # 计算波数
    k = 2 * np.pi * freq / c

    # 获取麦克风数量
    P = len(x_pos)

    # 处理扫描网格
    if np.isscalar(scan_x) and np.isscalar(scan_y) and np.isscalar(scan_z):
        # 如果是单个点，转换为数组
        scan_x = np.array([[scan_x]])
        scan_y = np.array([[scan_y]])
        scan_z = np.array([[scan_z]])
    elif scan_x.ndim == 1:
        # 如果是向量，转换为网格
        N = len(scan_x)
        M = len(scan_y)
        scan_x, scan_y = np.meshgrid(scan_x, scan_y)
        scan_z = np.ones_like(scan_x) * scan_z

    # 获取扫描网格维度
    M, N = scan_x.shape

    # 初始化导向矢量
    e = np.zeros((M, N, P), dtype=complex)

    # 计算每个扫描点到每个麦克风的距离
    for i in range(M):
        for j in range(N):
            # 计算距离差
            dx = scan_x[i,j] - x_pos
            dy = scan_y[i,j] - y_pos
            dz = scan_z[i,j] - z_pos
            
            # 计算总距离
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            
            # 计算导向矢量
            e[i,j,:] = np.exp(-1j * k * r) / (4 * np.pi * r)

    return e

def compute_single_steering_vector(x_pos, y_pos, z_pos, freq, c, source_x, source_y, source_z):
    """
    计算单个声源的导向矢量
    
    参数:
    x_pos: ndarray - 麦克风x坐标
    y_pos: ndarray - 麦克风y坐标
    z_pos: ndarray - 麦克风z坐标
    freq: float - 频率 [Hz]
    c: float - 声速 [m/s]
    source_x: float - 声源x坐标
    source_y: float - 声源y坐标
    source_z: float - 声源z坐标
    
    返回:
    ndarray - 导向矢量
    """
    # 计算波数
    k = 2 * np.pi * freq / c
    
    # 计算距离
    dx = x_pos - source_x
    dy = y_pos - source_y
    dz = z_pos - source_z
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    
    # 计算导向矢量
    e = np.exp(-1j * k * r) / (4 * np.pi * r)
    
    return e