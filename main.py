# Import necessary libraries
import argparse
import os
import pprint
import torch
import numpy as np
from datetime import datetime
from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from tianshou.trainer import offpolicy_trainer
from torch.distributions import Independent, Normal
from tianshou.exploration import GaussianNoise
from env import make_aigc_env
from policy import DiffusionOPT
from diffusion import Diffusion
from diffusion.model import MLP, DoubleCritic
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Define a function to get command line arguments
def get_args():
    # Create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--exploration-noise", type=float, default=0.1) # 控制探索噪声的强度。在训练过程中，添加一定量的噪声可以增加探索性，帮助模型发现更多的状态空间
    parser.add_argument('--algorithm', type=str, default='diffusion_opt')
    parser.add_argument('--seed', type=int, default=2) # 设置种子
    parser.add_argument('--buffer-size', type=int, default=1e6) # 1e6 经验回放缓冲区的大小
    parser.add_argument('-e', '--epoch', type=int, default=1e4) # 训练的总轮数。每一轮包含多个训练步骤。
    parser.add_argument('--step-per-epoch', type=int, default=1) # 每轮训练的步数。每一步对应一次与环境的交互。
    parser.add_argument('--step-per-collect', type=int, default=1) #每次收集数据的步数。通常用于决定何时更新策略
    parser.add_argument('-b', '--batch-size', type=int, default=512) # 每个训练批次的样本数量。较大的批量大小可以加速训练，但需要更多的内存
    parser.add_argument('--wd', type=float, default=1e-5) # 权重衰减（Weight Decay）。用于正则化，防止过拟合。
    parser.add_argument('--gamma', type=float, default=1) # 折扣因子（Discount Factor）。控制未来奖励的重要性。值越小，对未来奖励的关注越少。
    parser.add_argument('--n-step', type=int, default=3) # N-Step 返回。决定了使用多少步的奖励来计算目标价值。
    parser.add_argument('--training-num', type=int, default=2) # 训练环境中并行运行的环境数量。增加这个值可以加速训练。
    parser.add_argument('--test-num', type=int, default=1) # 测试环境中并行运行的环境数量。增加这个值可以更好地评估模型性能。
    parser.add_argument('--logdir', type=str, default='log') # log目录
    parser.add_argument('--log-prefix', type=str, default='default') # # 日志文件的前缀。不同的前缀可以帮助区分不同的实验。
    parser.add_argument('--render', type=float, default=0.1)
    parser.add_argument('--rew-norm', type=int, default=0) # 奖励归一化。如果设置为 1，则对奖励进行归一化处理。
    # parser.add_argument(
    #     '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument(
        '--device', type=str, default='cuda:7')
    parser.add_argument('--resume-path', type=str, default=None) # 恢复训练的路径。如果提供了一个有效的路径，则从该路径加载预训练的模型继续训练。
    parser.add_argument('--watch', action='store_true', default=False) # 是否进入测试模式。如果设置为 `True`，则只进行测试而不进行训练。
    parser.add_argument('--lr-decay', action='store_true', default=False) # 是否启用学习率衰减。如果设置为 `True`，则在训练过程中逐渐降低学习率。
    parser.add_argument('--note', type=str, default='reward = np.sum(data_rate) --step-per-epoch=100 --step-per-collect=1000，环境不固定') # 备注信息。用于记录实验的一些额外信息。

     # for diffusion
    parser.add_argument('--actor-lr', type=float, default=1e-5) # 演员网络的学习率。
    parser.add_argument('--critic-lr', type=float, default=1e-4) # 批评家网络的学习率。
    parser.add_argument('--tau', type=float, default=0.005)  # for soft update
    # adjust
    parser.add_argument('-t', '--n-timesteps', type=int, default=6)  # for diffusion chain 3 & 8 & 12
    parser.add_argument('--beta-schedule', type=str, default='vp',
                        choices=['linear', 'cosine', 'vp'])

    # With Expert: bc-coef True 有专家数据
    # Without Expert: bc-coef False 无专家数据
    parser.add_argument('--bc-coef', default=False) # Apr-04-132705 
    # parser.add_argument('--bc-coef', default=True)

    # for prioritized experience replay 优先经验回放
    parser.add_argument('--prioritized-replay', action='store_true', default=True)
    parser.add_argument('--prior-alpha', type=float, default=0.4)#
    parser.add_argument('--prior-beta', type=float, default=0.4)#

    # Parse arguments and return them
    args = parser.parse_known_args()[0]
    return args


def main(args=get_args()):
    # create environments
    env, train_envs, test_envs = make_aigc_env(args.training_num, args.test_num)
    args.state_shape = env.observation_space.shape[0]
    args.action_shape = env.action_space.n
    args.max_action = 1.

    # 探索噪声*最大动作
    args.exploration_noise = args.exploration_noise * args.max_action
    # seed
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # train_envs.seed(args.seed)
    # test_envs.seed(args.seed)

    # create actor 策略网络
    actor_net = MLP(
        state_dim=args.state_shape,
        action_dim=args.action_shape
    )
    # Actor is a Diffusion model 
    actor = Diffusion(
        state_dim=args.state_shape,
        action_dim=args.action_shape,
        model=actor_net,
        max_action=args.max_action,
        beta_schedule=args.beta_schedule,
        n_timesteps=args.n_timesteps,
        bc_coef = args.bc_coef
    ).to(args.device)
    actor_optim = torch.optim.AdamW(
        actor.parameters(),
        lr=args.actor_lr,
        weight_decay=args.wd
    )

    # Create critic
    critic = DoubleCritic(
        state_dim=args.state_shape,
        action_dim=args.action_shape
    ).to(args.device)
    critic_optim = torch.optim.AdamW(
        critic.parameters(),
        lr=args.critic_lr,
        weight_decay=args.wd
    )

    ## Setup logging
    time_now = datetime.now().strftime('%b%d-%H%M%S') # 当前时间
    log_path = os.path.join(args.logdir, args.log_prefix, "diffusion", time_now) # 存放log目录
    writer = SummaryWriter(log_path) # 创建一个writer对象
    writer.add_text("args", str(args)) # 将args各种参数写入writer
    logger = TensorboardLogger(writer) # 创建一个tensorboard logger对象

    # def dist(*logits):
    #    return Independent(Normal(*logits), 1)

    # Define policy
    policy = DiffusionOPT(
        args.state_shape,
        actor,
        actor_optim,
        args.action_shape,
        critic,
        critic_optim,
        # dist,
        args.device,
        tau=args.tau,
        gamma=args.gamma,
        estimation_step=args.n_step,
        lr_decay=args.lr_decay,
        lr_maxt=args.epoch,
        bc_coef=args.bc_coef,
        action_space=env.action_space,
        exploration_noise = args.exploration_noise,
    )

    # Load a previous policy if a path is provided 载入之前的训练好的模型
    if args.resume_path:
        # 载入模型
        ckpt = torch.load(args.resume_path, map_location=args.device)
        # 载入模型参数nn.module.load_state_dict()固定的方法
        policy.load_state_dict(ckpt)
        print("Loaded agent from: ", args.resume_path)

    # Setup buffer
    if args.prioritized_replay:
        buffer = PrioritizedVectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs),
            alpha=args.prior_alpha,
            beta=args.prior_beta,
        )
    else:
        buffer = VectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs)
        )

    # Setup collector
    train_collector = Collector(policy, train_envs, buffer)
    test_collector = Collector(policy, test_envs)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    # Trainer
    if not args.watch:
        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            args.epoch,
            args.step_per_epoch,
            args.step_per_collect,
            args.test_num,
            args.batch_size,
            save_best_fn=save_best_fn,
            logger=logger,
            test_in_train=False
        )
        pprint.pprint(result)

    # Watch the performance
    # python main.py --watch --resume-path log\default\diffusion\Apr04-132705\policy.pth
    if __name__ == '__main__':
        policy.eval()
        collector = Collector(policy, env)
        print(f"env 参数 state:{env.state}")
        result = collector.collect(n_episode=1) #, render=args.render
        print(result)
        # print(f"Action sequence: {result['act']}") # 打印动作序列
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward mean value: {rews.mean()}, length: {lens.mean()}")
        # 打印缓冲区的数据
        # action = collector.buffer.act
        # action = torch.from_numpy(np.array(action)).float()
        # action = torch.abs(action)
        # action = action.numpy()
        # total_power = 3
        # normalized_weights = action / np.sum(action)
        # a = normalized_weights * total_power
        # print(f"Collected actions: {a}")


if __name__ == '__main__':
    main(get_args())
