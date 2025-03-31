import gym
from gym.spaces import Box, Discrete
from tianshou.env import DummyVectorEnv
from .utility import CompUtility
import numpy as np
import tools

class AIGCEnv(gym.Env):

    def __init__(self):
        
        self._flag = 0
        # 定义环境参数
        self._power_total = 20 # 总功率
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
        self._L3 = self.__distance_RW ** self._path_loss_cof * self._distance_JR ** self._path_loss_cof  # Jammer-RIS-Willie的路径损失
        self._L4 = self._distance_RW ** self._path_loss_cof * self._distance_AR ** self._path_loss_cof  # Alice-RIS-Willie的路径损失
        self._interference_cof = 0  # 自干扰系数
        self._noise_variance = 10 ^(-80/10)  # 噪声方差
        self._rate_common_lowwer = 0 # 公共消息速率下限
        self._rate_private_lowwer = 0 # 私有消息速率下限
        self._precoder_dim = self._antanna_number*(self._user_number+1) # 预编码器维度
        # 定义观测空间
        self._observation_space = Box(shape=self.state.shape, low=0, high=1)
        # 定义动作空间
        self._action_space = Discrete(self.action_space_dim)
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
        h_AR = tools.generate_nakagami_channel(self._ris_number, self._user_number, self._m, self._Omega)
        h_JR = tools.generate_nakagami_channel(self._ris_number, 1, self._m, self._Omega)
        h_RW = tools.generate_nakagami_channel(self._ris_number, 1, self._m, self._Omega)
        h_RUk = [tools.generate_nakagami_channel(self._ris_number, 1, self._m, self._Omega) for _ in range(self._user_number)]
        # 将所有信道增益转换为 N x 1 维度的向量并拼接
        states1 = h_AR.flatten().reshape(-1, 1)
        states2 = h_JR.flatten().reshape(-1, 1)
        states3 = h_RW.flatten().reshape(-1, 1)
        states4 = np.concatenate(h_RUk, axis=1).flatten().reshape(-1, 1)
        reward_in = []
        reward_in.append(0)
        # 拼接所有状态
        states = np.concatenate([states1, states2, states3, states4, reward_in], axis=0)
        self.channel_gains = [h_AR, h_JR, h_RW, h_RUk]
        self._laststate = states
        return states

    def action_space_dim(self) -> int:
        '''返回动作空间的维度'''
        return self._ris_number\
                +self._power_p_cof+self._power_J_cof+self._power_c_cof\
                +self._precoder_dim*2

    def step(self, action):
        # Check if episode has ended
        assert not self._terminated, "One episodic has terminated"
        # Calculate reward based on last state and action taken
        reward, expert_action, sub_expert_action, real_action = CompUtility(self.channel_gains, action)

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
