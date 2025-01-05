import numpy as np

def create_rect_array(n_elements, array_size):
    """
    创建矩形麦克风阵列
    
    参数:
    n_elements: int - 每边麦克风数量
    array_size: float - 阵列尺寸
    
    返回:
    ndarray - 阵列元素的坐标 [3×N]
    """
    n_mic = n_elements ** 2
    
    # 生成X坐标
    x_pos = np.tile(np.linspace(-array_size/2, array_size/2, n_elements), n_elements)
    
    # 生成Y坐标
    y_pos = np.repeat(np.linspace(-array_size/2, array_size/2, n_elements), n_elements)
    
    # Z坐标全为0
    z_pos = np.zeros(n_mic)
    
    return np.array([x_pos, y_pos, z_pos])