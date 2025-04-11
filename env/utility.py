import numpy as np
import torch
from scipy.stats import nakagami
from scipy.special import gammainc
import math
from scipy.io import savemat
import os
from env import tools
os.environ['KMP_DUPLICATE_LIB_OK']='True'



# Function to compute utility (reward) for the given state and action
def CompUtility(State, 
                Action, 
                user_number,
                antenna_number,
                ris_number,
                power_total,
                channel_gain_UK,
                channel_gain_AR,
                channel_gain_JR,
                channel_gain_RW,
                L1,
                L2,
                L3,
                L4,
                self_interference_cof,
                noise_variance,
                AMDEP_cof
                ):
    '''
    计算奖励函数
    :param State: 信道增益
    :param Action: 动作
    :param user_number: 用户数量
    :return: 奖励
        例如[8,12,12]
        [RIS相位, 预编码器相位,预编码器幅度]
    '''
    actions = torch.from_numpy(np.array(Action)).float()
    actions = torch.abs(actions)
    Action = actions.numpy()
    cof1 = 1/3
    cof2 = 0.9
    power_total = 10**( power_total /10) # 功率分配 dBm 转为 mw
    # print(f"power_allocation: {power_allocation}")
    power_Jammer = power_total * (1-cof1) # 转化为mw
    power_Alice = power_total * cof1  # Alice功率
    power_common = power_Alice * cof2# 转化为mw
    power_private = power_Alice * (1-cof2) # 转化为mw
    power_common_cof =  cof2 # 公共消息功率分配系数
    power_private_cof = np.full((user_number, 1), (1-power_common_cof)/user_number)  # 私有消息功率分配系数
    # print(f"power_Jammer: {power_Jammer}w, power_common: {power_common}w, power_private: {power_private}w, power_Alice: {power_Alice}w")
    # 生成RIS
    # 生成RIS
    RIS_phase = 2 * math.pi * Action[ : ris_number]
    RIS = tools.generate_RIS_from_phase(RIS_phase)


    # 生成预编码器
    procoder_amplitude = Action[ ris_number : ris_number+antenna_number*(user_number+1)]
    procoder_phase = 2 * math.pi * Action[ris_number+antenna_number*(user_number+1): ]
    common_procoder, private_procoder = tools.procoder(
        antenna_number=antenna_number,
        user_number=user_number,
        amplitude=procoder_amplitude,
        phase=procoder_phase
    )

    # 计算AMDEP
    AMDEP = tools.AMDEP(
        user_number=user_number,
        power_Jammer_max=power_Jammer,
        power_Alice=power_Alice,
        power_common_cof = power_common_cof,
        L3=L3,
        L4=L4
    )

    # 计算速率
    rate_common_temp, rate_private_temp = tools.rate_uk(
        power_common_cof=power_common_cof,
        power_private_cof=power_private_cof,
        power_alice=power_Alice,
        power_jammer=power_Jammer,
        channal_ruk=channel_gain_UK,
        channal_ar=channel_gain_AR,
        channal_jr=channel_gain_JR,
        ris=RIS,
        distance_loss_cof_aru=L1,
        distance_loss_cof_jru=L2,
        self_interference_cof=self_interference_cof,
        noise_variance=noise_variance,
        user_number=user_number,
        procoder_common=common_procoder,
        procoder_private=private_procoder
        )
    reward = np.min(rate_common_temp) + np.sum(rate_private_temp)
    # print(f"AMDEP: {AMDEP}")
    # print(f"公共消息速率: {np.log2(1+np.min(rate_common_temp))}, 私有消息速率: {np.sum(np.log2(1+rate_private_temp))}")
    
    expert_action = None
    subopt_expert_action = None
    return reward, expert_action, subopt_expert_action, Action