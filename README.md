# 声源定位与声压级分析系统

## 项目概述
本项目实现了一个完整的声源定位与声压级分析系统，包括波束形成、DAMAS系列算法以及FISTA算法等多种声源定位方法。系统能够处理多频率声源信号，生成声源定位图像，并分析不同方法下的声压级特性，代码均为python，相较于matlab的运行速度更快。

本项目声源定位算法DAMAS、DAMAS2、DAMAS_FISTA、FISTA代码均来源于[HauLiang/Acoustic-Beamforming-Advanced: Scan-frequency Version for Acoustic Imaging, including the following methods: DAS, MUSIC, DAMAS, DAMAS2, DAMAS-FISTA, CLEAN-PSF, CLEAN-SC, FFT-NNLS, and FFT-DFISTA...](https://github.com/HauLiang/Acoustic-Beamforming-Advanced)

> [Liang, Hao and Zhou, Guanxing and Tu, Xiaotong and Jakobsson, Andreas and Ding, Xinghao and Huang, Yue, "Learning an Interpretable End-to-End Network for Real-Time Acoustic Beamforming", *Journal of Sound and Vibration*, 2024.](https://doi.org/10.1016/j.jsv.2024.118620)

## 系统架构

### 1. 核心模块
- `mic_array_design.py`: 麦克风阵列设计
- `acoustic_signal.py`: 声学信号生成
- `cross_spectral_matrix.py`: 互谱矩阵计算
- `steering_vector.py`: 导向矢量计算
- `scan_grid_generator.py`: 扫描网格生成

### 2. 声源定位算法
- `beamforming.py`: 延迟和求和(DAS)波束形成

> [Van Veen, Barry D and Buckley, Kevin M, "Beamforming: A versatile approach to spatial filtering", *IEEE assp magazine*, 1988.](https://ieeexplore.ieee.org/abstract/document/665/)

- `DAMAS.py`: DAMAS算法实现

> [Brooks, Thomas F and Humphreys, William M, "A deconvolution approach for the mapping of acoustic sources (DAMAS) determined from phased microphone arrays", *Journal of sound and vibration*, 2006.](https://www.sciencedirect.com/science/article/pii/S0022460X06000289)

- `DAMAS2.py`: DAMAS2算法实现

> [Dougherty, Robert, "Extensions of DAMAS and benefits and limitations of deconvolution in beamforming", *11th AIAA/CEAS aeroacoustics conference*, 2005.](https://doi.org/10.2514/6.2005-2961)

- `DAMAS_FISTA.py`: DAMAS-FISTA算法实现

> [Liang, Hao and Zhou, Guanxing and Tu, Xiaotong and Jakobsson, Andreas and Ding, Xinghao and Huang, Yue, "Learning an Interpretable End-to-End Network for Real-Time Acoustic Beamforming", *The Journal of Sound and Vibration*, 2024.](https://doi.org/10.1016/j.jsv.2024.118620)

- `FISTA.py`: FISTA算法实现

> [Lylloff, Oliver and Fernández-Grande, Efrén and Agerkvist, Finn and Hald, Jørgen and Tiana Roig, Elisabet and Andersen, Martin S. "Improving the efficiency of deconvolution algorithms for sound source localization". *The journal of the acoustical society of America*, 2015](http://dx.doi.org/10.1121/1.4922516)

### 3. 可视化模块
- `visualization.py`: 结果可视化，包括：
  - 声源定位结果图像
  - 声压级分布图
  - 声压级-频率曲线
  - 声源点声压级频谱图

### 4. 主程序
- `run_main.py`: 系统主程序，协调各模块工作

## 功能特点

### 1. 声源模拟
- 支持多个声源点的A字形排布
- 可设置声源的声压级和相干性
- 频率范围：6kHz - 20kHz，步长2kHz

### 2. 声源定位
- 实现多种定位算法：
  - DAS波束形成
  - DAMAS算法
  - DAMAS2算法
  - DAMAS-FISTA算法
  - FISTA算法
- 支持10kHz频率下的声源定位结果可视化

### 3. 声压级分析
- 计算并显示每个方法的声压级分布
- 生成声压级随频率变化的曲线
- 分析每个声源点的声压级频谱特性

### 4. 可视化输出
- 声源定位结果图像（10kHz）
- 各频率下的声压级分布图
- 声压级-频率变化曲线
- 各方法下声源点的声压级频谱

## 使用方法

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
- 采样频率：12800 Hz
- 采样点数：65536

#### 麦克风阵列设置
- 阵列类型：8×8矩形阵列
- 阵列尺寸：1m × 1m
- 阵列平面：XY平面（Z=0）
- 麦克风数量：64个
- 麦克风间距：均匀分布

#### 声源位置设置（A字形排布）
- 左斜线（3个点）：
  - 点1：(-0.08m, -0.06m, 0.5m)
  - 点2：(0m, -0.03m, 0.5m)
  - 点3：(0.08m, 0m, 0.5m)
- 右斜线（3个点）：
  - 点1：(-0.08m, 0.06m, 0.5m)
  - 点2：(0m, 0.03m, 0.5m)
  - 点3：(0.08m, 0m, 0.5m)
- 中间横线（1个点）：
  - 点：(0m, 0m, 0.5m)

#### 扫描网格设置
- 网格点数：41×41点
- 扫描范围：0.6m × 0.6m
- 扫描平面距离：0.5m（Z方向）
- 网格分辨率：约0.015m

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

