import numpy as np

def compute_csm(input_signal, freq, fs, n_fft, n_snapshots):
    """
    计算互谱矩阵
    
    参数:
    input_signal: ndarray - 输入信号
    freq: float - 分析频率
    fs: float - 采样频率
    n_fft: int - FFT点数
    n_snapshots: int - 快照数量
    
    返回:
    ndarray - 互谱矩阵
    """
    M, N = input_signal.shape  # M: n_samples, N: n_mics

    if n_fft * n_snapshots > M:
        raise ValueError("FFT长度乘以快照数必须小于输入信号的采样数")

    # 重组信号为 (n_mics, n_snapshots, n_fft)
    y = input_signal[:n_fft * n_snapshots, :].reshape(n_snapshots, n_fft, N)
    y = np.transpose(y, (2, 0, 1))  # 转换为 (n_mics, n_snapshots, n_fft)

    # 计算FFT
    X = np.fft.fft(y, axis=2) / np.sqrt(n_fft)  # 形状: (n_mics, n_snapshots, n_fft)

    # 找到目标频率的索引
    freq_vec = np.fft.fftfreq(n_fft, 1/fs)
    freq_idx = np.argmin(np.abs(freq - freq_vec))

    # 提取目标频率的数据
    Xf = X[:, :, freq_idx]  # 形状: (n_mics, n_snapshots)

    # 计算互谱矩阵
    R = Xf @ Xf.conj().T / n_snapshots  # 形状: (n_mics, n_mics)

    return R