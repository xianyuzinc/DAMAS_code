import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm

def plot_result_2d(S, scan_x, scan_y, title_name, save_path, num_fig):
    """
    绘制并保存2D结果图
    
    参数:
    S: ndarray - 波束形成结果
    scan_x: ndarray - X坐标网格
    scan_y: ndarray - Y坐标网格
    title_name: str - 图像标题
    save_path: str - 保存路径
    num_fig: int - 图像编号
    """
    # 设置中文字体
    font_path = 'C:/Windows/Fonts/simhei.ttf'  # 确保路径正确
    if os.path.exists(font_path):
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
    else:
        print(f"字体文件未找到: {font_path}. 使用默认字体。")
        font_prop = None  # 默认字体
    
    # 解决负号 '-' 显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False

    result = np.abs(S)
    
    plt.figure(num_fig)
    plt.pcolormesh(scan_y, scan_x, result, shading='auto')
    plt.title(title_name, fontproperties=font_prop)
    plt.xlabel('x', fontproperties=font_prop)
    plt.ylabel('y', fontproperties=font_prop)
    plt.colorbar()
    
    # 保存图像
    file_name = f"{title_name}.jpg"
    save_path = os.path.join(save_path, file_name)
    plt.savefig(save_path)
    
    # 关闭图像
    plt.close()

def plot_spl_vs_frequency(frequencies, all_results, save_path):
    """
    绘制声压级随频率变化的曲线
    """
    # 设置中文字体
    font_path = 'C:/Windows/Fonts/simhei.ttf'  # 确保路径正确
    if os.path.exists(font_path):
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
    else:
        print(f"字体文件未找到: {font_path}. 使用默认字体。")
        font_prop = None  # 默认字体
    
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False
    
    # 设置绘图样式
    plt.style.use('default')
    
    plt.figure(figsize=(12, 8))
    
    # 设置颜色和标记
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    
    # 绘制每种方法的曲线
    methods = ['DAS', 'DAMAS', 'DAMAS2', 'DAMAS-FISTA', 'FISTA']
    for method, color, marker in zip(methods, colors, markers):
        spl_values = [all_results[f]['max_spl'][method] for f in frequencies]
        plt.plot(frequencies/1000, spl_values, 
                label=method, color=color, marker=marker, 
                linewidth=2, markersize=8)
    
    # 添加给定声源声压级的参考线
    plt.axhline(y=100, color='k', linestyle='--', label='参考声压级')
    
    # 设置图表属性
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('频率 (kHz)', fontsize=12, fontproperties=font_prop)
    plt.ylabel('声压级 (dB)', fontsize=12, fontproperties=font_prop)
    plt.title('各方法声压级随频率变化曲线', fontsize=14, fontproperties=font_prop)
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left', prop=font_prop)
    
    # 美化图表
    plt.minorticks_on()  # 显示次要刻度
    plt.grid(True, which='minor', linestyle=':', alpha=0.3)  # 添加次要网格
    plt.grid(True, which='major', linestyle='--', alpha=0.7)  # 添加主要网格
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(os.path.join(save_path, 'SPL_vs_Frequency.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_spl_map(spl_map, scan_x, scan_y, title_name, save_path):
    """
    绘制声压级图像并保存
    
    参数:
    spl_map: ndarray - 声压级结果
    scan_x: ndarray - X坐标网格
    scan_y: ndarray - Y坐标网格
    title_name: str - 图像标题
    save_path: str - 保存路径
    """
    plt.figure(figsize=(12, 8))
    plt.pcolormesh(scan_y, scan_x, spl_map, shading='auto')
    plt.colorbar(label='声压级 (dB)')
    plt.title(title_name)
    plt.xlabel('X坐标 (m)')
    plt.ylabel('Y坐标 (m)')
    
    # 保存图像
    file_name = f"{title_name}.jpg"
    save_path = os.path.join(save_path, file_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_source_spl_vs_frequency(frequencies, all_results, save_path):
    """
    为每个方法绘制声源点声压级随频率变化的曲线
    """
    # 设置中文字体
    font_path = 'C:/Windows/Fonts/simhei.ttf'
    if os.path.exists(font_path):
        font_prop = fm.FontProperties(fname=font_path)
    else:
        font_prop = None

    # 获取所有声源点ID
    source_ids = list(all_results[frequencies[0]]['source_spl']['DAS'].keys())
    
    # 为每个方法创建单独的图
    methods = ['DAS', 'DAMAS', 'DAMAS2', 'DAMAS-FISTA', 'FISTA']
    colors = plt.cm.tab20(np.linspace(0, 1, len(source_ids)))  # 为每个声源点使用不同的颜色
    
    for method in methods:
        plt.figure(figsize=(12, 8))
        
        # 绘制每个声源点的声压级曲线
        for idx, source_id in enumerate(source_ids):
            spl_values = [all_results[f]['source_spl'][method][source_id] 
                         for f in frequencies]
            plt.plot(frequencies/1000, spl_values, 
                    label=f'声源 {source_id}', 
                    color=colors[idx], 
                    marker='o',
                    linewidth=2, 
                    markersize=6)
        
        # 添加给定声源声压级的参考线
        plt.axhline(y=100, color='k', linestyle='--', label='参考声压级')
        
        # 设置图表属性
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('频率 (kHz)', fontsize=12, fontproperties=font_prop)
        plt.ylabel('声压级 (dB)', fontsize=12, fontproperties=font_prop)
        plt.title(f'{method}方法各声源点声压级频谱', fontsize=14, fontproperties=font_prop)
        plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left', prop=font_prop)
        
        # 美化图表
        plt.minorticks_on()
        plt.grid(True, which='minor', linestyle=':', alpha=0.3)
        plt.grid(True, which='major', linestyle='--', alpha=0.7)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(os.path.join(save_path, f'SPL_vs_Frequency_{method}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()