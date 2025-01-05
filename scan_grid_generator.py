import numpy as np

def create_scan_grid(x_length, y_length, z_distance, n_elements):
    """
    生成扫描网格
    
    参数:
    x_length: float - 扫描平面x方向长度 [m]
    y_length: float - 扫描平面y方向长度 [m]
    z_distance: float - 扫描平面到阵列平面的距离 [m]
    n_elements: int - 网格点数
    
    返回:
    ndarray - 扫描点坐标 [M×N×3]
    """
    x_scan_pos = np.linspace(-x_length/2, x_length/2, n_elements)
    y_scan_pos = np.linspace(-y_length/2, y_length/2, n_elements)
    
    # 生成网格
    x_grid, y_grid = np.meshgrid(x_scan_pos, y_scan_pos)
    z_grid = np.ones_like(x_grid) * z_distance
    
    # 组合坐标
    return np.stack([x_grid, y_grid, z_grid], axis=2)