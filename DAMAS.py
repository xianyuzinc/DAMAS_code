# deconvolution.py

import numpy as np

def process_damas(S, e, max_iterations=1000):

    # 取实部作为输入
    Y = np.real(S)
    deps = 1e-10

    # 获取维度
    M, N, P = e.shape

    # 重塑导向矢量和初始化
    ee = e.reshape(M*N, P)
    
    # 计算A矩阵 (PSF)
    A = np.abs(ee @ ee.conj().T)**2 / P**2

    # 初始化源强度
    Q = np.zeros(M*N)
    Q0 = Y.flatten()

    # Gauss-Seidel迭代
    for iteration in range(max_iterations):
        Q_prev = Q.copy()
        
        # 对每个网格点进行迭代
        for n in range(M*N):
            # 计算当前点的强度
            sum_before = A[n, :n] @ Q[:n]
            sum_after = A[n, n+1:] @ Q0[n+1:]
            Q[n] = max(0, (Y.flatten()[n] - sum_before - sum_after) / A[n,n])

        # 计算收敛条件
        if iteration > 0:
            diff = np.abs(Q - Q_prev)
            if np.mean(Q_prev) > 0:
                rel_change = np.max(diff) / np.mean(Q_prev)
                if rel_change < deps:
                    print(f'DAMAS在{iteration}次迭代后收敛')
                    break

        Q0 = Q.copy()

    if iteration == max_iterations - 1:
        print(f'DAMAS达到最大迭代次数 ({max_iterations})')
    else:
        print(f'DAMAS在{iteration+1}次迭代后收敛')

    # 将结果重塑为原始维度
    return Q.reshape(M, N)