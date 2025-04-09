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
                Action_dim_distinguish,
                user_number,
                antenna_number,
                ris_number,
                power_total,
                channel_gain_UK,
                channel_gain_AR,
                channel_gain_JR,
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
    :param Action_dim: 动作维度的区分 例如[8,12,12]
    :param user_number: 用户数量
    :return: 奖励
        例如[8,12,12]
        [RIS相位, 预编码器相位,预编码器幅度]
    '''
    actions = torch.from_numpy(np.array(Action)).float()
    actions = torch.abs(actions)
    Action = actions.numpy()
    # print(f"action: {Action}")
    normalized_weights = Action/ np.sum(Action) # 归一化映射功率分配系数
    power_allocation = normalized_weights * 10**( power_total /10) # 功率分配 dBm
    # print(f"power_allocation: {power_allocation}")
    power_Jammer = power_allocation[0] # 转化为mw
    power_common = power_allocation[1] # 转化为mw
    power_private = power_allocation[2] # 转化为mw
    power_Alice = power_common + power_private  # Alice功率
    power_common_cof =  power_common/power_Alice # 公共消息功率分配系数
    power_private_cof = np.full((user_number, 1), (1-power_common_cof)/user_number)  # 私有消息功率分配系数
    # print(f"power_Jammer: {power_Jammer}w, power_common: {power_common}w, power_private: {power_private}w, power_Alice: {power_Alice}w")
    # 生成RIS
    RIS = tools.generate_RIS_random(ris_number, seed=1)
    # print(f"RIS_phase: {RIS_phase}")

    # 生成预编码器
    common_procoder, private_procoder = tools.precoder_normalization(channel_gain_UK.conj().T @ RIS @ channel_gain_AR)
    # print(f"common_procoder: {common_procoder}, private_procoder: {private_procoder}")

    # 计算AMDEP
    AMDEP = tools.AMDEP(
        user_number=user_number,
        power_Jammer_max=power_Jammer,
        power_Alice=power_Alice,
        power_common_cof = power_common_cof,
        L3=L3,
        L4=L4
    )
    # print(f"AMDEP: {AMDEP}")

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
    # reward = AMDEP
    # print(f"AMDEP: {AMDEP}")
    # print(f"公共消息速率: {np.log2(1+np.min(rate_common_temp))}, 私有消息速率: {np.sum(np.log2(1+rate_private_temp))}")
    
    if AMDEP < AMDEP_cof:
        reward =  AMDEP-AMDEP_cof
    else:
        reward = np.log2(1+np.min(rate_common_temp)) + np.sum(np.log2(1+rate_private_temp))
        # print(f"公共消息速率: {np.log2(1+np.min(rate_common_temp))}, 私有消息速率: {np.sum(np.log2(1+rate_private_temp))}")
    expert_action = None
    subopt_expert_action = None
    return reward, expert_action, subopt_expert_action, Action