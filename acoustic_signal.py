import numpy as np
from steering_vector import compute_steering_vector

def generate_signal(x_pos, y_pos, z_pos, freq, c, fs, source_x, source_y, source_z, 
                   amplitude, n_samples, coherence=False):
    """
    生成声源信号
    
    参数:
    x_pos, y_pos, z_pos: ndarray - 麦克风位置坐标
    freq: float - 波频率 [Hz]
    c: float - 声速 [m/s]
    fs: float - 采样频率 [Hz]
    source_x, source_y, source_z: float - 声源位置
    amplitude: float - 声源强度
    n_samples: int - 采样点数
    coherence: bool - 是否相干
    
    返回:
    ndarray - 声压信号，形状为 (n_samples, n_mics)
    """
    # 生成时间序列
    t = np.arange(n_samples) / fs
    
    # 计算导向矢量
    doa = compute_steering_vector(x_pos, y_pos, z_pos, freq, c, 
                                source_x, source_y, source_z).squeeze()
    
    # 将doa扩展为与时间序列相同的维度
    doa = doa[np.newaxis, :]  # 变成 (1, n_mics)
    t = t[:, np.newaxis]      # 变成 (n_samples, 1)
    
    # 生成信号
    if coherence:
        signal = 10**(amplitude/20) * doa * np.exp(1j * 2 * np.pi * freq * t)
    else:
        phase = np.random.randn(n_samples)[:, np.newaxis]  # 变成 (n_samples, 1)
        signal = 10**(amplitude/20) * doa * np.exp(1j * 2 * np.pi * (freq * t + phase))
    
    return signal  # 返回形状为 (n_samples, n_mics) 的数组