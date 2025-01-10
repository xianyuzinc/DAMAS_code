# 声源定位与声压级分析系统

## 文件架构

```
.
├── README.md                       # 项目说明文档
├── run_main.py                    # 主程序入口
├── acoustic_signal.py             # 声学信号生成模块
├── mic_array_design.py           # 麦克风阵列设计模块
├── cross_spectral_matrix.py      # 互谱矩阵计算模块
├── scan_grid_generator.py        # 扫描网格生成模块
├── steering_vector.py            # 导向矢量计算模块
├── beamforming.py               # 波束形成算法模块
├── DAMAS.py                     # DAMAS算法实现
├── DAMAS2.py                    # DAMAS2算法实现
├── DAMAS_FISTA.py              # DAMAS-FISTA算法实现
├── FISTA.py                    # FISTA算法实现
├── visualization.py            # 可视化模块
│
└── results/                    # 结果输出目录
    ├── A波束形成结果.jpg
    ├── A波束形成结果 (10kHz).jpg
    ├── A-DAMAS2结果.jpg
    ├── A-DAMAS2结果 (10kHz).jpg
    ├── A-DAMAS-FISTA结果.jpg
    ├── A-DAMAS-FISTA结果 (10kHz).jpg
    ├── A-FISTA结果.jpg
    └── SPL_vs_Frequency.png
```

### beamforming.py - 波束形成算法
实现延迟和求和(DAS)波束形成方法：
```python
def delay_and_sum(R, e, w=None):
    """
    延迟和求和波束形成
    
    参数:
    R: ndarray - 互谱矩阵 [P×P]
    e: ndarray - 导向矢量 [M×N×P]
    w: ndarray - 可选的权重向量 [P]
    
    返回:
    ndarray - 波束形成结果 [M×N]
    """
    M, N, P = e.shape
    
    # 使用均匀权重或自定义权重
    if w is None:
        w = np.ones(P)
    w = np.asarray(w).reshape(-1)
    
    # 初始化结果矩阵
    S = np.zeros((M, N), dtype=complex)
    
    # 计算延迟和求和结果
    for y in range(M):
        for x in range(N):
            ee = e[y, x, :].reshape(P, 1)
            S[y, x] = ((w * ee).conj().T @ R @ (ee * w)).flatten()[0]
    
    return S
```

## 模块依赖关系

1. run_main.py
   - 依赖所有其他模块
   - 协调整个系统的运行

2. acoustic_signal.py
   - 依赖 steering_vector.py
   - 生成声源信号

3. beamforming.py
   - 依赖 numpy
   - 实现DAS波束形成算法

4. DAMAS.py, DAMAS2.py, DAMAS_FISTA.py, FISTA.py
   - 依赖 numpy, scipy
   - 实现各种声源定位算法

5. visualization.py
   - 依赖 matplotlib, numpy
   - 负责结果可视化

6. 其他辅助模块
   - mic_array_design.py
   - cross_spectral_matrix.py
   - scan_grid_generator.py
   - steering_vector.py

## 数据流向

1. 信号生成流程：
   mic_array_design.py → acoustic_signal.py → cross_spectral_matrix.py

2. 声源定位流程：
   steering_vector.py → beamforming.py → DAMAS系列算法

3. 结果处理流程：
   算法结果 → visualization.py → results/

## 项目概述
本项目实现了一个完整的声源定位与声压级分析系统，包括波束形成、DAMAS系列算法以及FISTA算法等多种声源定位方法。系统能够处理多频率声源信号，生成声源定位图像，并分析不同方法下的声压级特性。

## 系统架构

### 1. 核心模块

#### mic_array_design.py - 麦克风阵列设计
创建矩形麦克风阵列的布局设计。
```python
def create_rect_array(n_elements, array_size):
    """创建矩形麦克风阵列"""
    n_mic = n_elements ** 2
    x_pos = np.tile(np.linspace(-array_size/2, array_size/2, n_elements), n_elements)
    y_pos = np.repeat(np.linspace(-array_size/2, array_size/2, n_elements), n_elements)
    z_pos = np.zeros(n_mic)
    return np.array([x_pos, y_pos, z_pos])
```

#### acoustic_signal.py - 声学信号生成
生成声源信号，支持相干和不相干信号。
```python
def generate_signal(x_pos, y_pos, z_pos, freq, c, fs, source_x, source_y, source_z, 
                   amplitude, n_samples, coherence=False):
    """生成声源信号"""
    t = np.arange(n_samples) / fs
    doa = compute_steering_vector(x_pos, y_pos, z_pos, freq, c, 
                                source_x, source_y, source_z).squeeze()
    if coherence:
        signal = 10**(amplitude/20) * doa * np.exp(1j * 2 * np.pi * freq * t)
    else:
        phase = np.random.randn(n_samples)[:, np.newaxis]
        signal = 10**(amplitude/20) * doa * np.exp(1j * 2 * np.pi * (freq * t + phase))
    return signal
```

#### cross_spectral_matrix.py - 互谱矩阵计算
计算麦克风阵列信号的互谱矩阵。
```python
def compute_csm(input_signal, freq, fs, n_fft, n_snapshots):
    """计算互谱矩阵"""
    y = input_signal[:n_fft * n_snapshots, :].reshape(n_snapshots, n_fft, -1)
    y = np.transpose(y, (2, 0, 1))
    X = np.fft.fft(y, axis=2) / np.sqrt(n_fft)
    freq_vec = np.fft.fftfreq(n_fft, 1/fs)
    freq_idx = np.argmin(np.abs(freq - freq_vec))
    Xf = X[:, :, freq_idx]
    R = Xf @ Xf.conj().T / n_snapshots
    return R
```

#### steering_vector.py - 导向矢量计算
计算声源定位所需的导向矢量。
```python
def compute_steering_vector(x_pos, y_pos, z_pos, freq, c, scan_x, scan_y, scan_z):
    """计算导向矢量"""
    k = 2 * np.pi * freq / c
    dx = scan_x[i,j] - x_pos
    dy = scan_y[i,j] - y_pos
    dz = scan_z[i,j] - z_pos
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    e[i,j,:] = np.exp(-1j * k * r) / (4 * np.pi * r)
    return e
```

### 2. 声源定位算法

#### DAMAS.py - DAMAS算法实现
使用Gauss-Seidel迭代求解声源分布。
```python
def process_damas(S, e, max_iterations=1000):
    """DAMAS算法处理"""
    Y = np.real(S)
    ee = e.reshape(M*N, P)
    A = np.abs(ee @ ee.conj().T)**2 / P**2
    Q = np.zeros(M*N)
    
    for iteration in range(max_iterations):
        Q_prev = Q.copy()
        for n in range(M*N):
            sum_before = A[n, :n] @ Q[:n]
            sum_after = A[n, n+1:] @ Q0[n+1:]
            Q[n] = max(0, (Y.flatten()[n] - sum_before - sum_after) / A[n,n])
    return Q.reshape(M, N)
```

#### DAMAS2.py - DAMAS2算法实现
使用FFT加速的DAMAS算法。
```python
def process_damas2(S, e, max_iterations=1000, tol=1e-6):
    """DAMAS2算法处理"""
    Y = np.real(S)
    g_center = e[center_x, center_y, :]
    PSF = np.abs(ee @ g_center.conj())**2 / P**2
    
    for iteration in range(max_iterations):
        r = fftshift(np.real(ifft2(fft2(x) * fft_psf * psi)))
        x = np.maximum(0, x_old + (Y_pad - r) / a)
    return result
```

#### DAMAS_FISTA.py - DAMAS-FISTA算法实现
结合FISTA的快速DAMAS算法。
```python
def process_damas_fista(S, e, max_iterations=1000, tol=1e-6):
    """DAMAS-FISTA算法处理"""
    Y = np.real(S)
    A = np.abs(ee @ ee.conj().T)**2 / P**2
    L = np.real(linalg.eigvals(ATA)).max()
    
    for iteration in range(max_iterations):
        x = np.maximum(0, y - (1/L) * grad)
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        y = x + ((t - 1) / t_new) * (x - x_old)
    return x.reshape(M, N)
```

#### FISTA.py - FISTA算法实现
快速迭代收缩阈值算法。
```python
def process_fista(PSF, b, x0, max_iterations=1000):
    """FISTA算法处理"""
    L = estimate_lipschitz(PSF, Fps)
    
    for iteration in range(max_iterations):
        x = np.maximum(0, y - (1/L) * grad_y)
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        y = x + ((t - 1) / t_new) * (x - x_old)
    return x, info
```

### 3. 可视化模块

#### visualization.py - 结果可视化
提供多种可视化功能：
```python
def plot_result_2d(S, scan_x, scan_y, title_name, save_path, num_fig):
    """绘制2D声源定位结果"""

def plot_spl_vs_frequency(frequencies, all_results, save_path):
    """绘制声压级-频率曲线"""

def plot_spl_map(spl_map, scan_x, scan_y, title_name, save_path):
    """绘制声压级分布图"""

def plot_source_spl_vs_frequency(frequencies, all_results, save_path):
    """绘制声源点声压级频谱"""
```

### 4. 主程序

#### run_main.py - 系统主程序
协调各模块工作，主要功能：
```python
def process_frequency(freq, params, source_positions, mic_pos, scan_grid, x_pos, y_pos, z_pos):
    """处理单个频率的声源定位"""
    # 生成信号
    signal_total = generate_signal(...)
    
    # 计算CSM
    R = compute_csm(...)
    
    # 计算导向矢量
    e = compute_steering_vector(...)
    
    # 各种处理方法
    beamforming_result = delay_and_sum(...)
    damas_result = process_damas(...)
    damas2_result = process_damas2(...)
    damas_fista_result = process_damas_fista(...)
    fista_result = process_fista(...)
    
    return results
```

## 使用方法与参数设置

### 1. 环境要求
```python
numpy
matplotlib
scipy
tqdm
```

### 2. 运行方式
```bash
python run_main.py
```

### 3. 参数设置
- 声速：343 m/s
- 采样频率：48000 Hz
- 采样点数：262144
- FFT点数：256
- 快照数：400

#### 麦克风阵列设置
- 阵列类型：8×8矩形阵列
- 阵列尺寸：0.8m × 0.8m
- 阵列平面：XY平面（Z=0）
- 麦克风数量：64个
- 麦克风间距：11.4cm（均匀分布）
- 工作频率范围：600Hz - 15kHz
- 最高无混叠频率：1.5kHz

#### 扫描网格设置
- 网格点数：121×121点
- 扫描范围：0.6m × 0.6m
- 扫描平面距离：0.5m（Z方向）
- 网格分辨率：5mm（Δx/B ≈ 0.24 @10kHz）
- 空间采样定理：满足奈奎斯特采样准则

## 输出结果
所有结果保存在`results`文件夹中：
1. `A波束形成结果 (10kHz).jpg`等声源定位图像
2. 各方法在不同频率下的声压级图像
3. `SPL_vs_Frequency.png`声压级-频率曲线
4. `SPL_vs_Frequency_[方法名].png`各方法的声源点声压级频谱

## 性能说明
- 处理时间随频率点数和迭代次数增加
- DAMAS系列算法迭代次数：5000
- FISTA算法迭代次数：20000
- 进度条显示处理进度

