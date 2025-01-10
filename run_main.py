# acoustic_imaging_monopole.py

import numpy as np
import matplotlib.pyplot as plt
import os
from mic_array_design import create_rect_array
from acoustic_signal import generate_signal
from cross_spectral_matrix import compute_csm
from scan_grid_generator import create_scan_grid
from steering_vector import compute_steering_vector
from beamforming import delay_and_sum
from DAMAS import process_damas
from DAMAS2 import process_damas2
from visualization import plot_result_2d, plot_spl_vs_frequency, plot_spl_map, plot_source_spl_vs_frequency
from DAMAS_FISTA import process_damas_fista
from FISTA import process_fista
from tqdm import tqdm  # 添加在文件开头的导入部分

def process_frequency(freq, params, source_positions, mic_pos, scan_grid, x_pos, y_pos, z_pos):
    """
    处理单个频率的声源定位
    
    参数:
    freq: float - 处理频率
    params: dict - 基本参数
    source_positions: dict - 声源位置
    mic_pos: tuple - 麦克风位置
    scan_grid: ndarray - 扫描网格
    x_pos, y_pos, z_pos: ndarray - 麦克风坐标
    
    返回:
    dict - 包含各种处理结果和声压级
    """
    # 生成信号
    signal_total = np.zeros((params['n_samples'], len(x_pos)), dtype=complex)
    amplitude = 100  # 给定声源声压级

    for source in source_positions.values():
        signal = generate_signal(x_pos, y_pos, z_pos, freq, params['c'],
                               params['fs'], source['x'], source['y'], source['z'],
                               amplitude, params['n_samples'], params['coherence'])
        signal_total += signal

    # 计算CSM
    R = compute_csm(signal_total, freq, params['fs'], params['n_fft'], params['k'])

    # 计算导向矢量
    e = compute_steering_vector(x_pos, y_pos, z_pos, freq, params['c'],
                              scan_grid[..., 0], scan_grid[..., 1], scan_grid[..., 2])

    # 各种处理方法
    beamforming_result = delay_and_sum(R, e)
    damas_result = process_damas(beamforming_result, e, max_iterations=5000)
    damas2_result = process_damas2(beamforming_result, e, max_iterations=50000)
    damas_fista_result = process_damas_fista(beamforming_result, e, max_iterations=20000)
    
    # 准备PSF
    M, N, P = e.shape
    g_center = e[M//2, N//2, :]
    ee = e.reshape(M*N, P)
    PSF = np.abs(ee @ g_center.conj())**2 / P**2
    PSF = PSF.reshape(M, N)
    
    # FISTA和FFT-DFISTA处理
    fista_result, _ = process_fista(PSF, np.real(beamforming_result), 
                                  np.zeros_like(beamforming_result), max_iterations=20000)

    # 计算各方法的最大声压级
    max_spl = {
        'DAS': 20 * np.log10(np.max(np.abs(beamforming_result))),
        'DAMAS': 20 * np.log10(np.max(np.abs(damas_result))),
        'DAMAS2': 20 * np.log10(np.max(np.abs(damas2_result))),
        'DAMAS-FISTA': 20 * np.log10(np.max(np.abs(damas_fista_result))),
        'FISTA': 20 * np.log10(np.max(np.abs(fista_result)))
    }

    # 计算每个点的声压级
    spl_results = {
        'DAS': 20 * np.log10(np.abs(beamforming_result) + 1e-10),  # 加小常数避免log(0)
        'DAMAS': 20 * np.log10(np.abs(damas_result) + 1e-10),
        'DAMAS2': 20 * np.log10(np.abs(damas2_result) + 1e-10),
        'DAMAS-FISTA': 20 * np.log10(np.abs(damas_fista_result) + 1e-10),
        'FISTA': 20 * np.log10(np.abs(fista_result) + 1e-10)
    }

    # 获取声源点在网格中的索引
    M, N = scan_grid[..., 0].shape
    source_indices = {}
    
    # 获取网格坐标
    x_grid = scan_grid[..., 0].flatten()  # 展平为1D数组
    y_grid = scan_grid[..., 1].flatten()
    
    for source_id, source in source_positions.items():
        # 找到最近的网格点
        distances = np.sqrt((x_grid - source['x'])**2 + (y_grid - source['y'])**2)
        nearest_idx = np.argmin(distances)
        
        # 将1D索引转换为2D索引
        i = nearest_idx // N
        j = nearest_idx % N
        
        source_indices[source_id] = (i, j)

    # 计算每个声源点的声压级
    source_spl = {
        'DAS': {src_id: 20 * np.log10(np.abs(beamforming_result[i, j]) + 1e-10) 
                for src_id, (i, j) in source_indices.items()},
        'DAMAS': {src_id: 20 * np.log10(np.abs(damas_result[i, j]) + 1e-10) 
                  for src_id, (i, j) in source_indices.items()},
        'DAMAS2': {src_id: 20 * np.log10(np.abs(damas2_result[i, j]) + 1e-10) 
                   for src_id, (i, j) in source_indices.items()},
        'DAMAS-FISTA': {src_id: 20 * np.log10(np.abs(damas_fista_result[i, j]) + 1e-10) 
                        for src_id, (i, j) in source_indices.items()},
        'FISTA': {src_id: 20 * np.log10(np.abs(fista_result[i, j]) + 1e-10) 
                  for src_id, (i, j) in source_indices.items()}
    }

    return {
        'DAS': beamforming_result,
        'DAMAS': damas_result,
        'DAMAS2': damas2_result,
        'DAMAS-FISTA': damas_fista_result,
        'FISTA': fista_result,
        'max_spl': max_spl,
        'spl_results': spl_results,
        'source_spl': source_spl
    }

def main():
    # 添加当前目录
    current_folder = os.getcwd()
    
    # 基本参数设置
    params = {
        'c': 343,           # 声速
        'fs': 48000,        # 采样频率
        'n_samples': 262144, # 增加采样点数，确保大于 n_fft * k
        'coherence': 0,     # 不相干
        'n_fft': 256,       # 减小FFT点数
        'k': 400           # 调整快照数
    }
    
    # 频率范围设置
    frequencies = np.arange(8000, 12000, 2000)
    
    # 麦克风阵列设置
    n_elements, array_length = 8, 0.8  # 8×8阵列，阵列尺寸为0.8m
    mic_pos = create_rect_array(n_elements, array_length)
    x_pos, y_pos, z_pos = mic_pos[0], mic_pos[1], mic_pos[2]

    # 定义声源位置（A字母形状）
    # 左边斜线（3个点）
    A_X1 = [-0.08, 0, 0.08]
    A_Y1 = [-0.06, -0.03, 0]
    A_Z1 = [0.5, 0.5, 0.5]

    # 右边斜线（3个点）
    A_X2 = [-0.08, 0, 0.08]
    A_Y2 = [0.06, 0.03, 0]
    A_Z2 = [0.5, 0.5, 0.5]

    # 中间横线（1个点）
    A_X3 = [0]
    A_Y3 = [0]
    A_Z3 = [0.5]

    # 定义声源位置
    source_positions = {}

    # 添加左边斜线的声源
    for i in range(3):
        source_id = f'source_left_{i+1}'
        source_positions[source_id] = {
            'x': A_X1[i],
            'y': A_Y1[i],
            'z': A_Z1[i]
        }

    # 添加右边斜线的声源
    for i in range(3):
        source_id = f'source_right_{i+1}'
        source_positions[source_id] = {
            'x': A_X2[i],
            'y': A_Y2[i],
            'z': A_Z2[i]
        }

    # 添加中间横线的声源
    source_positions['source_middle'] = {
        'x': A_X3[0],
        'y': A_Y3[0],
        'z': A_Z3[0]
    }

    # 扫描网格设置
    x_length, y_length, z_distance = 0.6, 0.6, 0.5
    n_scan_points = 121  # 增加网格点数以获得更细的分辨率
    scan_grid = create_scan_grid(x_length, y_length, z_distance, n_scan_points)
    scan_x, scan_y = scan_grid[..., 0], scan_grid[..., 1]

    # 存储所有频率的结果，添加进度条
    print("正在处理各个频率...")
    all_results = {}
    for freq in tqdm(frequencies, desc="频率处理进度"):
        all_results[freq] = process_frequency(
            freq, params, source_positions, mic_pos, 
            scan_grid, x_pos, y_pos, z_pos
        )

    # 绘制10kHz的声源定位结果
    results_10k = all_results[10000]
    titles = {
        'DAS': 'A波束形成结果 (10kHz)',
        'DAMAS': 'A-DAMAS结果 (10kHz)',
        'DAMAS2': 'A-DAMAS2结果 (10kHz)',
        'DAMAS-FISTA': 'A-DAMAS-FISTA结果 (10kHz)',
        'FISTA': 'A-FISTA结果 (10kHz)'
    }
    
    results_path = os.path.join(current_folder, 'results')
    os.makedirs(results_path, exist_ok=True)

    print("\n正在生成声源定位结果图像...")
    for method in tqdm(titles.keys(), desc="声源定位图像生成进度"):
        if method in results_10k:  # 检查方法是否存在
            plot_result_2d(results_10k[method], scan_x, scan_y, 
                          titles[method], results_path, list(titles.keys()).index(method)+1)

    print("\n正在生成声压级图像...")
    for freq in tqdm(frequencies, desc="声压级图像生成进度"):
        results_freq = all_results[freq]
        for method, spl_map in results_freq['spl_results'].items():
            title = f"{method} 声压级图 (频率: {freq/1000} kHz)"
            plot_spl_map(spl_map, scan_x, scan_y, title, results_path)
    
    print("\n正在生成声压级-频率曲线...")
    plot_spl_vs_frequency(frequencies, all_results, results_path)

    print("\n正在生成声源点声压级频谱图...")
    plot_source_spl_vs_frequency(frequencies, all_results, results_path)

    print("\n所有处理完成！")

if __name__ == "__main__":
    main()