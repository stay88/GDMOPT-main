import numpy as np

import numpy as np

def generate_rayleigh_channel(N: int, M: int, Omega: float=1) -> np.ndarray:
    """
    生成瑞利衰落信道矩阵（幅度服从瑞利分布，相位均匀分布）
    输入参数：
        N      : 矩阵行数（天线数/用户数）
        M      : 矩阵列数（时间样本/子载波数）
        Omega  : 平均功率（E[|h|²] = Omega）
    输出：
        H_rayleigh : N x M 的复信道矩阵（复数形式包含幅度和相位）
    """
    # 生成独立的高斯实部和虚部（均值为0，方差为Omega/2）
    real_part = np.random.normal(0, np.sqrt(Omega/2), (N, M))
    imag_part = np.random.normal(0, np.sqrt(Omega/2), (N, M))
    
    # 合成复信道矩阵
    return real_part + 1j * imag_part



def generate_nakagami_channel(N: int, M: int, m: float, Omega: float) -> np.ndarray:
    """
    生成Nakagami-m衰落信道矩阵（幅度服从Nakagami-m分布，相位均匀分布）
    输入参数：
        N      : 矩阵行数（天线数/用户数）
        M      : 矩阵列数（时间样本/子载波数）
        m      : Nakagami形状参数（m >= 0.5）
        Omega  : 平均功率（E[|h|²] = Omega）
    输出：
        H_nakagami : N x M 的复信道矩阵（复数形式包含幅度和相位）
    """
    if m < 0.5:
        raise ValueError("Nakagami形状参数m必须 >= 0.5")
    
    # 生成伽马分布随机变量（形状参数=m，尺度参数=Omega/m）
    gamma_samples = np.random.gamma(shape=m, scale=Omega/m, size=(N, M))
    
    # 幅度 = sqrt(伽马变量)
    amplitude = np.sqrt(gamma_samples)
    
    # 生成均匀相位（0到2π）
    phase = np.random.uniform(0, 2*np.pi, (N, M))
    
    # 合成复信道矩阵
    return amplitude * np.exp(1j * phase)


def generate_RIS_random(N) -> np.ndarray:
    """
    生成 NxN 随机的 RIS 对角矩阵
    :param N: 矩阵维度
    :return: NxN 的对角矩阵，对角线元素为幅度为1的复数
    """
    # 生成均匀分布的相位
    phase = 2 * np.pi * np.random.rand(N, 1)
    
    # 生成复数，幅度为1
    h = 1*np.exp(1j * phase)
    
    # 创建对角矩阵
    ris = np.diag(np.squeeze(h))
    return ris


def precoder_normalization(H):
    """
    计算预编码向量和矩阵
    :param H: Alice-User 的信道增益矩阵 [N_t, K] (天线数 x 用户数)
    :return: p_c (共形预编码向量), p_k (归一化的零迫预编码矩阵)
    """
    # 共形预编码计算
    precoding_c = np.sum(H, axis=1)  # 按列求和
    p_c = precoding_c / np.linalg.norm(precoding_c)  # 归一化
    
    # 计算零迫预编码矩阵
    procoding_uk = np.linalg.pinv(H.T @ H) @ H.T  # H/(H'H) 的等价实现
    
    # 归一化每一列
    for k in range(procoding_uk.shape[1]):
        norm_k = np.linalg.norm(procoding_uk[:, k])
        if norm_k != 0:
            procoding_uk[:, k] /= norm_k
    
    p_k = procoding_uk
    return p_c, p_k

def rate_uk(
    power_common_cof: float,  # 公共消息功率分配系数
    power_private_cof: np.ndarray,  # 私有消息功率分配系数 [K,1]
    power_alice: float,  # Alice的功率
    power_jammer: float,  # Jammer的功率
    channal_ruk: np.ndarray,  # RIS-Uk的信道增益 [N,K]
    channal_ar: np.ndarray,  # Alice-RIS的信道增益 [N,M]
    channal_jr: np.ndarray,  # Jammer-RIS的信道增益 [N,1]
    ris: np.ndarray,  # RIS反射单元 [N,N]
    distance_loss_cof_aru: np.ndarray,  # Alice-RIS-Uk的路径损失 [K,1]
    distance_loss_cof_jru: np.ndarray,  # Jammer-RIS-Uk的路径损失 [K,1]
    self_interference_cof: float,  # 自干扰系数
    noise_variance: float,  # 噪声方差
    user_number: int,  # 用户数量 K
    procoder_common: np.ndarray,  # 公共消息预编码 [M,1]
    procoder_private: np.ndarray  # 私有消息预编码 [M,K]
) -> tuple[np.ndarray, np.ndarray]:
    """
    计算公共消息速率和私有消息速率。
    """
    # 运行时检查，确保矩阵参数是 np.ndarray 类型
    for param in [channal_ruk, channal_ar, channal_jr, ris, 
                  distance_loss_cof_aru, distance_loss_cof_jru, 
                  procoder_common, procoder_private]:
        assert isinstance(param, np.ndarray), f"{param} must be a numpy ndarray"
    
    rate_common_temp = np.zeros(user_number)
    
    for k in range(user_number):
        rate_common_up = power_common_cof * power_alice * \
            np.abs(channal_ruk[:, k].conj().T @ ris @ channal_ar @ procoder_common) ** 2 * distance_loss_cof_aru[k]
        temp = 0
        for j in range(user_number):
            temp += power_private_cof[j] * power_alice * distance_loss_cof_aru[k] * \
                   np.abs(channal_ruk[:, k].conj().T @ ris @ channal_ar @ procoder_private[:, j]) ** 2
        
        rate_common_down = temp + self_interference_cof * power_jammer * distance_loss_cof_jru[k] * \
                           np.abs(channal_ruk[:, k].conj().T @ ris @ channal_jr) ** 2 + noise_variance
        
        rate_common_temp[k] = rate_common_up / rate_common_down
    
    rate_private_temp = np.zeros(user_number)
    
    for k in range(user_number):
        rate_private_up = power_private_cof[k] * power_alice * \
            np.abs(channal_ruk[:, k].conj().T @ ris @ channal_ar @ procoder_private[:, k]) * distance_loss_cof_aru[k]
        temp = 0
        for j in range(user_number):
            if j == k:
                continue
            temp += power_private_cof[j] * power_alice * distance_loss_cof_aru[k] * \
                   np.abs(channal_ruk[:, k].conj().T @ ris @ channal_ar @ procoder_private[:, j]) ** 2
        
        rate_private_down = temp + self_interference_cof * power_jammer * distance_loss_cof_jru[k] * \
                            np.abs(channal_ruk[:, k].conj().T @ ris @ channal_jr) ** 2 + noise_variance
        
        rate_private_temp[k] = rate_private_up / rate_private_down
    
    return rate_common_temp, rate_private_temp


