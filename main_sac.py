import argparse
import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
from env import Env
from sac_model import SACContinuous, ReplayBuffer as SACReplayBuffer
from utils import lineplot

# Parameters
parser = argparse.ArgumentParser(description='SAC Baseline (DDPG-like Encoder)')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=200, help='Random seed')
parser.add_argument('--env', type=str, default='UAV-v0', help='Environment name')
parser.add_argument('--episodes', type=int, default=1000, help='Total episodes')
parser.add_argument('--test-interval', type=int, default=25, help='Test interval')
parser.add_argument('--test-episodes', type=int, default=10, help='Test episodes')
parser.add_argument('--eval-steps', type=int, default=5000, help='Steps per test session')
parser.add_argument('--max-episode-length', type=int, default=1000, help='Max steps per episode')
parser.add_argument('--action-repeat', type=int, default=2, help='Action repeat')
parser.add_argument('--bit-depth', type=int, default=5, help='Image bit depth')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')

args = parser.parse_args()

# Setup
args.symbolic_env = False
if torch.cuda.is_available() and not args.disable_cuda:
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

results_dir = os.path.join('results', '{}_SAC_{}'.format(args.env, args.id))
os.makedirs(results_dir, exist_ok=True)
summary_name = results_dir + "/SAC_log"
writer = SummaryWriter(summary_name)

# Hyperparameters
sac_actor_lr = 3e-4
sac_critic_lr = 3e-3
sac_alpha_lr = 3e-4
sac_gamma = 0.99
sac_tau = 0.005
sac_buffer_size = 100000
sac_batch_size = 16
start_steps = 0

# Metrics tracking
metrics = {
    'test_episodes': [],
    'test_rewards': [],
    'test_avg_rewards': []
}


def process_sac_obs(obs_dict):
    if isinstance(obs_dict['image'], torch.Tensor):
        img = obs_dict['image'].cpu().numpy().squeeze(0)
        tgt = obs_dict['target'].cpu().numpy().squeeze(0)
    else:
        img = obs_dict['image']
        tgt = obs_dict['target']

    return np.concatenate([img, tgt], axis=0)


# Init Env and Agent
env = Env(args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth)
replay_buffer = SACReplayBuffer(sac_buffer_size)
target_entropy = -env.action_size
agent = SACContinuous(env.action_size, sac_actor_lr, sac_critic_lr, sac_alpha_lr, target_entropy, sac_tau, sac_gamma,
                      args.device, action_bound=1.0)

print(f"Starting SAC Training on {args.device}...")

global_steps = 0
try:
    pbar = tqdm(range(1, args.episodes + 1), desc="SAC Training")

    for episode in pbar:
        # Training Phase
        obs_dict = env.reset()
        state = process_sac_obs(obs_dict)
        done = False
        total_reward = 0

        # Loss variables
        c_loss, a_loss, alpha_val = 0.0, 0.0, 0.0

        while not done:
            if global_steps < start_steps:
                action = env.sample_random_action().numpy()
            else:
                # 训练时 deterministic=False (进行采样)
                action = agent.take_action(state, deterministic=False)

            next_obs_dict, reward, done = env.step(action)

            if torch.is_tensor(reward): reward = reward.item()
            if torch.is_tensor(done): done = done.item()

            next_state = process_sac_obs(next_obs_dict)
            replay_buffer.add(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            global_steps += 1

            if replay_buffer.size() > sac_batch_size and global_steps >= start_steps:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(sac_batch_size)
                transition_dict = {'states': b_s, 'actions': b_a, 'rewards': b_r, 'next_states': b_ns, 'dones': b_d}

                c_loss, a_loss, alpha_val = agent.update(transition_dict)

                if global_steps % 100 == 0:
                    writer.add_scalar('SAC/Critic_Loss', c_loss, global_steps)
                    writer.add_scalar('SAC/Actor_Loss', a_loss, global_steps)
                    writer.add_scalar('SAC/Alpha', alpha_val, global_steps)

            pbar.set_postfix({
                'Step': global_steps,
                'C_L': f'{c_loss:.3f}',
                'Rew': f'{total_reward:.1f}'
            })

        writer.add_scalar('SAC/Train_Reward', total_reward, episode)

        # Testing Phase
        if episode % args.test_interval == 0:
            print(f"Testing at episode {episode}...")
            total_reach_count = 0
            total_accum_reward = 0

            for _ in range(args.test_episodes):
                session_steps = 0
                session_reaches = 0
                session_accum_reward = 0

                while session_steps < args.eval_steps:
                    t_obs_dict = env.reset()
                    t_state = process_sac_obs(t_obs_dict)
                    t_done = False

                    while not t_done:
                        # 测试时必须 deterministic=True
                        t_action = agent.take_action(t_state, deterministic=True)
                        t_next_obs, t_reward, t_done = env.step(t_action)

                        if torch.is_tensor(t_reward): t_reward = t_reward.item()

                        session_accum_reward += t_reward
                        if t_reward >= 100.0:
                            session_reaches += 1

                        t_state = process_sac_obs(t_next_obs)
                        session_steps += 1
                        if session_steps >= args.eval_steps: break

                total_reach_count += session_reaches
                total_accum_reward += session_accum_reward

            avg_reach_count = total_reach_count
            avg_test_reward = total_accum_reward / args.test_episodes

            writer.add_scalar('Average_Reach_Count', avg_reach_count, global_steps)
            writer.add_scalar('Average_Test_Reward', avg_test_reward, global_steps)
            writer.add_scalar('Comparison/Average_Reach_Count', avg_reach_count, global_steps)
            print(
                f"SAC Test Episode {episode}: Reach Count = {avg_reach_count:.2f}, Avg Reward = {avg_test_reward:.2f}")

            metrics['test_episodes'].append(episode)
            metrics['test_rewards'].append(avg_reach_count)
            metrics['test_avg_rewards'].append(avg_test_reward)

            lineplot(metrics['test_episodes'], metrics['test_rewards'], 'Reach Count', results_dir)
            lineplot(metrics['test_episodes'], metrics['test_avg_rewards'], 'Average Test Reward', results_dir)
            torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

except KeyboardInterrupt:
    print("SAC Training interrupted.")

env.close()
writer.close()