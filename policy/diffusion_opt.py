import torch
import copy
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
from typing import Any, Dict, List, Type, Optional, Union
from tianshou.data import Batch, ReplayBuffer, to_torch
from tianshou.policy import BasePolicy
from torch.optim.lr_scheduler import CosineAnnealingLR
from .helpers import (
    Losses
)
class DiffusionOPT(BasePolicy):

    def __init__(
            self,
            state_dim: int,
            actor: Optional[torch.nn.Module],
            actor_optim: Optional[torch.optim.Optimizer],
            action_dim: int,
            critic: Optional[torch.nn.Module],
            critic_optim: Optional[torch.optim.Optimizer],
            # dist_fn: Type[torch.distributions.Distribution],
            device: torch.device,
            tau: float = 0.005,
            gamma: float = 1,
            reward_normalization: bool = False,
            estimation_step: int = 1,
            lr_decay: bool = False,
            lr_maxt: int = 1000,
            bc_coef: bool = False,
            exploration_noise: float = 0.1,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        assert 0.0 <= tau <= 1.0, "tau should be in [0, 1]"
        assert 0.0 <= gamma <= 1.0, "gamma should be in [0, 1]"

        # Initialize actor network and optimizer if provided 策略网络，决定在每个状态下采取的动作；
        if actor is not None and actor_optim is not None:
            self._actor: torch.nn.Module = actor  # Actor network
            self._target_actor = deepcopy(actor)  # Target actor network for stable learning
            self._target_actor.eval()  # Set target actor to evaluation mode
            self._actor_optim: torch.optim.Optimizer = actor_optim  # Optimizer for the actor network
            self._action_dim = action_dim  # Dimensionality of the action space

        # Initialize critic network and optimizer if provided 价值网络，评估在每个状态下采取的动作的好坏；
        if critic is not None and critic_optim is not None:
            self._critic: torch.nn.Module = critic  # Critic network
            self._target_critic = deepcopy(critic)  # Target critic network for stable learning
            self._critic_optim: torch.optim.Optimizer = critic_optim  # Optimizer for the critic network
            self._target_critic.eval()  # Set target critic to evaluation mode

        # If learning rate decay is applied, initialize learning rate schedulers for both actor and critic
        if lr_decay:
            self._actor_lr_scheduler = CosineAnnealingLR(self._actor_optim, T_max=lr_maxt, eta_min=0.)
            self._critic_lr_scheduler = CosineAnnealingLR(self._critic_optim, T_max=lr_maxt, eta_min=0.)

        # Initialize other parameters and configurations
        self._tau = tau  # Soft update coefficient for target networks
        self._gamma = gamma  # Discount factor for future rewards
        self._rew_norm = reward_normalization  # If true, normalize rewards
        self._n_step = estimation_step  # Steps for n-step return estimation
        self._lr_decay = lr_decay  # If true, apply learning rate decay
        self._bc_coef = bc_coef  # Coefficient for policy gradient loss
        self._device = device  # Device to run computations on
        self.noise_generator = GaussianNoise(sigma=exploration_noise)

    
    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        ''' 计算目标Q值'''
        batch = buffer[indices]  # batch.obs_next: s_{t+n}
        # Compute the actions for next states with target actor network 
        # self() 函数由于继承了nn.model，
        # 所以实例对象可以直接调用__call__函数，__call__函数又调用了forward函数
        ttt = self(batch, model='_target_actor', input='obs_next').act # 返回动作
        # 使用目标网络评估这些行为
        batch.obs_next = to_torch(batch.obs_next, device=self._device, dtype=torch.float32)
        target_q = self._target_critic.q_min(batch.obs_next, ttt)
        return target_q  # return the minimum of the dual Q values

    def process_fn(self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray) -> Batch:
        # Compute n-step return for transitions in the batch 计算批次中的n步后的累计奖励返回
        return self.compute_nstep_return(
            batch,
            buffer,
            indices,
            self._target_q,
            self._gamma,
            self._n_step,
            self._rew_norm
        )

    # 更新网络参数
    def update(
            self,
            sample_size: int,
            buffer: Optional[ReplayBuffer],
            **kwargs: Any
    ) -> Dict[str, Any]:
        # 如果没有提供重放缓冲区，则返回一个空字典
        if buffer is None: return {}
        self.updating = True # 标识策略正在更新

        #一批次的样本数量
        batch, indices = buffer.sample(sample_size)

        # 计算样本转换的n步返回
        batch = self.process_fn(batch, buffer, indices)
        # 更新网络参数
        result = self.learn(batch, **kwargs)
        if self._lr_decay: # If learning rate decay is enabled, step the learning rate schedulers 如果启用了学习率衰减，则调整学习率调度程序
            self._actor_lr_scheduler.step()
            self._critic_lr_scheduler.step()
        self.updating = False # Indicate that the policy update has finished 标识策略更新已完成
        return result

    def forward(
            self,
            batch: Batch,
            state: Optional[Union[dict, Batch, np.ndarray]] = None,
            input: str = "obs",
            model: str = "actor"
    ) -> Batch:
        '''计算动作'''
        # 将观测空间state转换为PyTorch张量
        obs_ = to_torch(batch[input], device=self._device, dtype=torch.float32)
        # print(obs_)
        # 演员网络，actor或目标actor模型
        model_ = self._actor if model == "actor" else self._target_actor
        # 传入观测空间state到模型，获取动作logits（未归一化的概率）
        logits, hidden = model_(obs_), None
        # print(logits)
        if self._bc_coef:
            acts = logits
        else:
            if np.random.rand() < 0.1:
                # 0.1的概率添加探索高斯噪声
                noise = to_torch(self.noise_generator.generate(logits.shape),
                                 dtype=torch.float32, device=self._device)
                # Add the noise to the action
                acts = logits + noise
                # acts = logits
                acts = torch.clamp(acts, -1, 1)
            else:
                acts = logits

        dist = None  # does not use a probability distribution for actions

        return Batch(logits=logits, act=acts, state=obs_, dist=dist)

    def _to_one_hot(
            self,
            data: np.ndarray,
            one_hot_dim: int
    ) -> np.ndarray:
        # Convert the provided data to one-hot representation
        batch_size = data.shape[0]
        one_hot_codes = np.eye(one_hot_dim)
        # print(data[1])
        one_hot_res = [one_hot_codes[data[i]].reshape((1, one_hot_dim))
                       for i in range(batch_size)]
        return np.concatenate(one_hot_res, axis=0)

    def _update_critic(self, batch: Batch) -> torch.Tensor:
        # Compute the critic's loss and update its parameters
        obs_ = to_torch(batch.obs, device=self._device, dtype=torch.float32)
        acts_ = to_torch(batch.act, device=self._device, dtype=torch.float32)
        target_q = batch.returns # Target Q values are the n-step returns
        # print('target_q',target_q)
        # td, critic_loss = self._mse_optimizer(batch, self.critic, self.critic_optim)
        current_q1, current_q2 = self._critic(obs_,acts_) # Current Q values are the critic's output
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q) # Compute the MSE loss
        # critic_loss = F.mse_loss(current_q1, target_q)

        self._critic_optim.zero_grad() # Zero the critic optimizer's gradients
        critic_loss.backward() # Backpropagate the loss
        self._critic_optim.step() # Perform a step of optimization
        return critic_loss


    def _update_bc(self, batch: Batch, update: bool = False) -> torch.Tensor:
        # Compute the behavior cloning loss
        obs_ = to_torch(batch.obs, device=self._device, dtype=torch.float32)
        # expert_actions = torch.Tensor([info["sub_expert_action"] for info in batch.info]).to(self._device)
        expert_actions = torch.Tensor([info["expert_action"] for info in batch.info]).to(self._device)

        bc_loss = self._actor.loss(expert_actions, obs_).mean()

        if update:  # Update actor parameters if update flag is True
            self._actor_optim.zero_grad()  # Zero the actor optimizer's gradients
            bc_loss.backward()  # Backpropagate the loss
            self._actor_optim.step()  # Perform a step of optimization
        return bc_loss

    def _update_policy(self, batch: Batch, update: bool = False) -> torch.Tensor:
        # Compute the policy gradient loss 计算策略梯度损失
        obs_ = to_torch(batch.obs, device=self._device, dtype=torch.float32)
        acts_ = to_torch(self(batch).act, device=self._device, dtype=torch.float32)
        pg_loss = - self._critic.q_min(obs_, acts_).mean()
        if update:
            self._actor_optim.zero_grad()
            pg_loss.backward()
            self._actor_optim.step()
        return pg_loss

    def _update_targets(self):
        # Perform soft update on target actor and target critic. Soft update is a method of slowly blending
        # the regular and target network to provide more stable learning updates.
        self.soft_update(self._target_actor, self._actor, self._tau)
        self.soft_update(self._target_critic, self._critic, self._tau)

    def learn(
            self,
            batch: Batch,
            **kwargs: Any
    ) -> Dict[str, List[float]]:
        
        #更新评论家网络。评论家网络被更新以最小化Q值预测（current_q1）和实际的目标Q值（target_q）之间的均方误差损失。
        critic_loss = self._update_critic(batch)
        # Update actor network. Here, we first calculate the policy gradient (pg_loss) and
        # behavior cloning loss (bc_loss) but we do not update the actor network yet.
        # The overall loss is a weighted combination of policy gradient loss and behavior cloning loss.
        # 更新演员网络。首先，我们计算策略梯度（pg_loss）和行为克隆损失（bc_loss），但此时还不更新演员网络。、
        # 总体损失是策略梯度损失和行为克隆损失的加权组合。但是这里并没有加权
        if self._bc_coef:
            bc_loss = self._update_bc(batch, update=False)
            overall_loss = bc_loss
        else:
            pg_loss = self._update_policy(batch, update=False)
            overall_loss = pg_loss
        # 这三行代码共同构成了神经网络训练的梯度更新循环，典型流程为：清空旧梯度→计算新梯度→应用梯度更新参数
        self._actor_optim.zero_grad()
        overall_loss.backward()
        self._actor_optim.step()

        # Update the target networks
        self._update_targets()
        return {
            'loss/critic': critic_loss.item(),  # Returns the critic loss as part of the results
            'overall_loss': overall_loss.item()  # Returns the overall loss as part of the results
        }

class GaussianNoise:
    """Generates Gaussian noise."""

    def __init__(self, mu=0.0, sigma=0.1):
        """
        :param mu: Mean of the Gaussian distribution.
        :param sigma: Standard deviation of the Gaussian distribution.
        """
        self.mu = mu
        self.sigma = sigma

    def generate(self, shape):
        """
        Generate Gaussian noise based on a shape.

        :param shape: Shape of the noise to generate, typically the action's shape.
        :return: Numpy array with Gaussian noise.
        """
        noise = np.random.normal(self.mu, self.sigma, shape)
        return noise
