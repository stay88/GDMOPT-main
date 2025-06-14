import gym
from gym.spaces import Box, Discrete
from tianshou.env import DummyVectorEnv
from .utility import CompUtility
import numpy as np
from env import tools

class AIGCEnv(gym.Env):

    def __init__(self):
        
        self._flag = 0
        # 定义环境参数
        self._AMDEP_cof = 0.8 # AMDEP系数下限
        self._power_total = 20 # 总功率dBm
        self._antanna_number = 3 # 天线数量
        self._user_number = 3 # 用户数量
        self._ris_number = 8 # RIS数量
        self._power_J_cof = 1 # Jammer功率分配系数
        self._power_c_cof = 1 # 公共消息功率分配系数
        self._power_p_cof = 1 # 私有消息功率分配系数
        self._m = 2 # 形状参数
        self._Omega = 1 # 平均功率
        self._distance_JR = 50 # Jammer-RIS的距离
        self._distance_AR = 50 # Alice-RIS的距离
        self._distance_RW = 50 # RIS-Willie的距离
        self._distance_step = 15 # 步长
        self._distance_RUk = np.array([self._distance_JR + i * self._distance_step for i in range(self._user_number)]) # RIS-Uk的距离 依次增大的
        self._path_loss_cof = -2.5 # 路径损失系数
        self._L1 = (self._distance_RUk ** self._path_loss_cof) * (self._distance_AR ** self._path_loss_cof)    # Alice-RIS-Uk的路径损失
        self._L2 = (self._distance_RUk ** self._path_loss_cof) * (self._distance_JR ** self._path_loss_cof)   # Jammer-RIS-Uk的路径损失
        self._L3 = self._distance_RW ** self._path_loss_cof * self._distance_JR ** self._path_loss_cof  # Jammer-RIS-Willie的路径损失
        self._L4 = self._distance_RW ** self._path_loss_cof * self._distance_AR ** self._path_loss_cof  # Alice-RIS-Willie的路径损失
        self._interference_cof = 0  # 自干扰系数
        self._noise_variance = 10 **(-80/10)  # 噪声方差
        self._rate_common_lowwer = 0 # 公共消息速率下限
        self._rate_private_lowwer = 0 # 私有消息速率下限
        self._precoder_dim = self._antanna_number*(self._user_number+1) # 预编码器维度
        # 定义观测空间
        self._observation_space = Box(shape=self.state.shape, low=-np.inf, high=np.inf)
        # 定义动作空间
        self._action_space = Discrete(3)
        self._num_steps = 0
        self._terminated = False
        self._laststate = None
        self.last_expert_action = None
        # Define the number of steps per episode
        self._steps_per_episode = 1

    @property
    def observation_space(self):
        # Return the observation space
        return self._observation_space

    @property
    def action_space(self):
        # Return the action space
        return self._action_space


    @property
    def state(self):
        # Provide the current state to the agent
        # rng = np.random.default_rng(seed=0)
        # states1 = rng.uniform(1, 2, 5)
        # states2 = rng.uniform(0, 1, 5)
        # 生成信道增益，返回都是复数形式 
        # 维度 h_AR=N*M h_JR=N*1 h_RW=N*1 h_RUk=N*1*K
        seed = 1 # 种子
        # seed = None
        h_AR = tools.generate_nakagami_channel(self._ris_number, self._user_number, self._m, self._Omega, seed)
        h_JR = tools.generate_nakagami_channel(self._ris_number, 1, self._m, self._Omega, seed)
        h_RW = tools.generate_nakagami_channel(self._ris_number, 1, self._m, self._Omega, seed)
        h_RUk = tools.generate_nakagami_channel(self._ris_number, self._user_number, self._m, self._Omega, seed)
        # 将所有信道增益转换为 N x 1 维度的向量并拼接
        states_AR_real = np.real(h_AR.flatten())
        states_AR_imag = np.imag(h_AR.flatten())
        states_JR_real = np.real(h_JR.flatten())
        states_JR_imag = np.imag(h_JR.flatten())
        states_RW_real = np.real(h_RW.flatten()) # h_RW
        states_RW_imag = np.imag(h_RW.flatten())
        states_RUk_real = np.real(h_RUk.flatten())
        states_RUk_imag = np.imag(h_RUk.flatten())
        reward_in = []
        reward_in.append(0)
        # 拼接所有状态, state维度 2*(N*M+N*1+N*1*K)
        states = np.concatenate([states_AR_real, states_JR_real, states_RUk_real, states_AR_imag, states_JR_imag, states_RUk_imag,  reward_in], axis=0)
        self.channel_gains_dict = {"h_AR":h_AR, "h_JR":h_JR, "h_RW":h_RW, "h_RUk":h_RUk}
        self.channel_gains = np.concatenate([states_AR_real, states_JR_real, states_RUk_real, states_AR_imag, states_JR_imag, states_RUk_imag], axis=0)
        self._laststate = states
        return states

    def action_space_dim(self) -> int:
        '''返回动作空间的维度'''
        return self._ris_number+self._precoder_dim*2

    def step(self, action):
        # Check if episode has ended
        assert not self._terminated, "One episodic has terminated"
        # 区分不同动作的指针
        action_dim_distinguish = [self._ris_number, self._precoder_dim, self._precoder_dim]
        # Calculate reward based on last state and action taken
        reward, expert_action, sub_expert_action, real_action = CompUtility(
            self._laststate,
            action,
            action_dim_distinguish,
            self._user_number,
            self._antanna_number,
            self._ris_number,
            self._power_total,
            self.channel_gains_dict["h_RUk"],
            self.channel_gains_dict["h_AR"],
            self.channel_gains_dict["h_JR"],
            self._L1,
            self._L2,
            self._L3,
            self._L4,
            self._interference_cof,
            self._noise_variance,
            self._AMDEP_cof
            )

        self._laststate[-1] = reward
        # self._laststate[0:-1] = self.channel_gains * real_action # 就算赋值在断点中也不会改变
        # print(self._laststate.base)  # 如果不是 None，说明是视图
        self._laststate[0:-1] = self.channel_gains
        self._num_steps += 1
        # Check if episode should end based on number of steps taken
        if self._num_steps >= self._steps_per_episode:
            self._terminated = True
        # Information about number of steps taken
        info = {'num_steps': self._num_steps, 'expert_action': expert_action, 'sub_expert_action': sub_expert_action}
        # info  = {'num_steps': self._num_steps}
        return self._laststate, reward, self._terminated, info

    def reset(self):
        # Reset the environment to its initial state
        self._num_steps = 0
        self._terminated = False
        state = self.state
        return state, {'num_steps': self._num_steps}

    def seed(self, seed=None):
        # Set seed for random number generation
        np.random.seed(seed)


def make_aigc_env(training_num=0, test_num=0):
    """Wrapper function for AIGC env.
    :return: a tuple of (single env, training envs, test envs).
    """
    env = AIGCEnv()
    env.seed(0)

    train_envs, test_envs = None, None
    if training_num:
        # Create multiple instances of the environment for training
        train_envs = DummyVectorEnv(
            [lambda: AIGCEnv() for _ in range(training_num)])
        train_envs.seed(0)

    if test_num:
        # Create multiple instances of the environment for testing
        test_envs = DummyVectorEnv(
            [lambda: AIGCEnv() for _ in range(test_num)])
        test_envs.seed(0)
    return env, train_envs, test_envs
