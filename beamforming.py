import numpy as np

def delay_and_sum(R, e, w=None):
    """
    延迟和求和波束形成
    DAS
    参数:
    R: ndarray - 互谱矩阵
    e: ndarray - 导向矢量
    w: ndarray, optional - 权重向量
    
    返回:
    ndarray - 波束形成结果
    """
    M, N, P = e.shape
    
    # 如果没有提供权重，使用均匀权重
    if w is None:
        w = np.ones(P)
    w = np.asarray(w).reshape(-1)
    
    # 初始化结果矩阵
    S = np.zeros((M, N), dtype=complex)
    
    # 计算延迟和求和结果
    for y in range(M):
        for x in range(N):
            ee = e[y, x, :].reshape(P, 1)
            # 将结果展平并提取第一个元素作为标量
            S[y, x] = ((w * ee).conj().T @ R @ (ee * w)).flatten()[0]
    
    return S