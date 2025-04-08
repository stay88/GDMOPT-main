import torch
import torch.nn as nn
from .helpers import SinusoidalPosEmb
from tianshou.data import Batch, ReplayBuffer, to_torch

# 多层感知机
class MLP(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        t_dim=16,
        activation='mish'
    ):
        super(MLP, self).__init__()
        _act = nn.Mish if activation == 'mish' else nn.ReLU
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), # 全连接层,输入维度state_dim,输出维度hidden_dim
            _act(), # 激活函数
            nn.Linear(hidden_dim, hidden_dim) # 全连接层,输入维度hidden_dim,输出维度hidden_dim
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim), # 时间嵌入
            nn.Linear(t_dim, t_dim * 2), # 全连接层,输入维度t_dim,输出维度t_dim * 2
            _act(),
            nn.Linear(t_dim * 2, t_dim),
        )
        self.mid_layer = nn.Sequential( # 全连接层
            nn.Linear(hidden_dim + action_dim + t_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x, time, state): # 前向传播
        processed_state = self.state_mlp(state)
        t = self.time_mlp(time)
        x = torch.cat([x, t, processed_state], dim=1)
        x = self.mid_layer(x)
        # x = self.final_layer(x)
        return x

# 评论家网络
class DoubleCritic(nn.Module):
    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_dim=256,
            activation='mish'
    ):
        super(DoubleCritic, self).__init__()
        # _act = nn.Mish if activation == 'mish' else nn.ReLU
        _act = nn.ReLU
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.q1_net = nn.Sequential(nn.Linear(hidden_dim + action_dim, hidden_dim),
                                      _act(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      _act(),
                                      nn.Linear(hidden_dim, 1))
        self.q2_net = nn.Sequential(nn.Linear(hidden_dim + action_dim, hidden_dim),
                                      _act(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      _act(),
                                      nn.Linear(hidden_dim, 1))
    # def forward(self, obs):
    #     return self.q1_net(obs), self.q2_net(obs)
    #
    # def q_min(self, obs):
    #     return torch.min(*self.forward(obs))
    def forward(self, state, action):
        processed_state = self.state_mlp(state)
        x = torch.cat([processed_state, action], dim=-1)
        return self.q1_net(x), self.q2_net(x)

    def q_min(self, obs, action):
        # obs = to_torch(obs, device='cuda:0', dtype=torch.float32)
        # action = to_torch(action, device='cuda:0', dtype=torch.float32)
        # 双Q值网络，取最小值
        return torch.min(*self.forward(obs, action))
    
    # def q_max(self, obs, action):
    #     # 双Q值网络，取最大值
    #     return torch.max(*self.forward(obs, action))
