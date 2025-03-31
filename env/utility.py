import numpy as np
import torch
from scipy.stats import nakagami
from scipy.special import gammainc
import math
from scipy.io import savemat
import os
import tools
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def rayleigh_channel_gain(ex, sta):
    num_samples = 1
    gain = np.random.normal(ex, sta, num_samples)
    # Square the absolute value to get Rayleigh-distributed gains
    gain = np.abs(gain) ** 2
    return gain


# Function to compute utility (reward) for the given state and action
def CompUtility(State, 
                Action, 
                Action_dim_distinguish,
                user_number,
                power_total,
                channel_gain_UK,
                channel_gain_AR,
                channel_gain_JR,
                RIS,
                L1,
                L2,
                L3,
                L4,
                self_interference_cof,
                noise_variance,):
    '''
    计算奖励函数
    :param State: 信道增益
    :param Action: 动作
    :param Action_dim: 动作维度的区分 例如[3,8,12,12]
    :return: 奖励
        动作的前3个元素是功率分配，中间是RIS的相位，最后是预编码器的幅度和相位
        例如[3,8,12,12]
        [Jammer功率分配, 公共消息功率分配, 私有消息功率分配, RIS相位, 预编码器幅度, 预编码器相位]
    '''
    actions = torch.from_numpy(np.array(Action)).float()
    actions = torch.abs(actions)
    Action = actions.numpy()

    normalized_weights = Action[:3] / np.sum(Action[:3]) # 归一化映射功率分配系数
    power_common_cof = normalized_weights[1]  # 公共消息功率分配系数
    power_private_cof = np.full((user_number, 1), normalized_weights[2]/user_number)  # 私有消息功率分配系数
    power_allocation = normalized_weights * power_total  # 功率分配
    power_Jammer = power_allocation[0]  # Jammer功率
    power_common = power_allocation[1]  # 公共消息功率
    power_private = power_allocation[2]  # 私有消息总功率
    power_Alice = power_common + power_private  # Alice功率

    # 生成RIS
    RIS_phase = 2 * math.pi * Action[ Action_dim_distinguish[0] : Action_dim_distinguish[0]+Action_dim_distinguish[1]]
    RIS = tools.generate_RIS_from_phase(RIS_phase)

    # 生成预编码器
    procoder_amplitude = Action[Action_dim_distinguish[0]+Action_dim_distinguish[1] : Action_dim_distinguish[0]+Action_dim_distinguish[1]+Action_dim_distinguish[2]]
    procoder_phase = 2 * math.pi * Action[Action_dim_distinguish[0]+Action_dim_distinguish[1]+Action_dim_distinguish[2]:]
    

    AMDEP = tools.AMDEP(user_number=user_number,
                        power_Jammer_max=power_Jammer,
                        power_Alice=power_Alice,
                        power_common_cof = power_common_cof,
                        L3=L3,
                        L4=L4)
    
    rate_common_temp, rate_private_temp = tools.rate_uk(power_common_cof=power_common_cof,
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
                                                        procoder_common=)
    
    
    return reward, expert_action, subopt_expert_action, Aution