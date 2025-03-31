import numpy as np
import torch
from scipy.stats import nakagami
from scipy.special import gammainc
import math
from scipy.io import savemat
import os
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
                Action_dim_distinguish):
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
    total_power = 20 # 总功率
    normalized_weights = Action[:3] / np.sum(Action)
    power_allocation = normalized_weights * total_power
    power_Jammer = power_allocation[0]  # Jammer功率
    power_common = power_allocation[1]  # 公共消息功率
    power_private = power_allocation[2]  # 私有消息功率
    
    
    return reward, expert_action, subopt_expert_action, Aution