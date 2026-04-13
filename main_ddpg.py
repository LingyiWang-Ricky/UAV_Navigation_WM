import argparse
import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
from env import Env
from ddpg_model import DDPG, ReplayBuffer
from utils import lineplot

# Parameters
parser = argparse.ArgumentParser(description='DDPG Baseline')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=300, help='Random seed')
parser.add_argument('--env', type=str, default='UAV-v0', help='Environment name')
parser.add_argument('--episodes', type=int, default=1000, help='Total episodes')
parser.add_argument('--test-interval', type=int, default=25, help='Test interval')
parser.add_argument('--test-episodes', type=int, default=10, help='Test episodes')
parser.add_argument('--eval-steps', type=int, default=1000, help='Steps per test session')
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

# Separate folder for DDPG results
results_dir = os.path.join('results', '{}_DDPG_{}'.format(args.env, args.id))
os.makedirs(results_dir, exist_ok=True)
summary_name = results_dir + "/DDPG_log"
writer = SummaryWriter(summary_name)

# DDPG Parameters
actor_lr = 3e-4
critic_lr = 3e-3
gamma = 0.98
tau = 0.005
sigma = 0.2  # Exploration noise
buffer_size = 1000000
batch_size = 20

# Metrics tracking
metrics = {
    'test_episodes': [],
    'test_rewards': [],
    'test_avg_rewards': []
}


def process_ddpg_obs(obs_dict):
    if isinstance(obs_dict['image'], torch.Tensor):
        img = obs_dict['image'].cpu().numpy().squeeze(0)
        tgt = obs_dict['target'].cpu().numpy().squeeze(0)
    else:
        img = obs_dict['image']
        tgt = obs_dict['target']
    return np.concatenate([img, tgt], axis=0)


# Init Env and Agent
env = Env(args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth)
replay_buffer = ReplayBuffer(buffer_size)
action_dim = env.action_size  # Should be 2 for UAV
agent = DDPG(action_dim, 1.0, sigma, actor_lr, critic_lr, tau, gamma, args.device)

print(f"Starting DDPG Training on {args.device}...")

global_steps = 0
try:
    pbar = tqdm(range(1, args.episodes + 1), desc="DDPG Training")

    for episode in pbar:
        # Training Phase
        obs_dict = env.reset()
        state = process_ddpg_obs(obs_dict)
        done = False
        total_reward = 0
        c_loss, a_loss = 0.0, 0.0

        while not done:
            # Take action with Noise for exploration
            action = agent.take_action(state, noise=True)

            # Step in env
            next_obs_dict, reward, done = env.step(action)

            if torch.is_tensor(reward): reward = reward.item()
            if torch.is_tensor(done): done = done.item()

            next_state = process_ddpg_obs(next_obs_dict)
            replay_buffer.add(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            global_steps += 1

            # Update if buffer is ready
            if replay_buffer.size() > batch_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                transition_dict = {'states': b_s, 'actions': b_a, 'rewards': b_r, 'next_states': b_ns, 'dones': b_d}

                c_loss, a_loss = agent.update(transition_dict)

                pbar.set_postfix({
                    'Step': global_steps,
                    'C_L': f'{c_loss:.3f}',
                    'A_L': f'{a_loss:.3f}',
                    'Rew': f'{total_reward:.1f}'
                })

                if global_steps % 100 == 0:
                    writer.add_scalar('DDPG/Critic_Loss', c_loss, global_steps)
                    writer.add_scalar('DDPG/Actor_Loss', a_loss, global_steps)

        writer.add_scalar('DDPG/Train_Reward', total_reward, episode)

        # Testing Phase
        if episode % args.test_interval == 0:
            print(f"Testing at episode {episode}...")
            total_reach_count = 0
            total_accum_reward = 0

            # Run 'test_episodes' number of Evaluation Sessions
            for _ in range(args.test_episodes):
                session_steps = 0
                session_reaches = 0
                session_accum_reward = 0

                while session_steps < args.eval_steps:
                    t_obs_dict = env.reset()
                    t_state = process_ddpg_obs(t_obs_dict)
                    t_done = False

                    while not t_done:
                        # No noise during testing
                        t_action = agent.take_action(t_state, noise=False)
                        t_next_obs, t_reward, t_done = env.step(t_action)

                        if torch.is_tensor(t_reward): t_reward = t_reward.item()

                        session_accum_reward += t_reward

                        # Check for Reach
                        if t_reward >= 100.0:
                            session_reaches += 1

                        t_state = process_ddpg_obs(t_next_obs)
                        session_steps += 1

                        if session_steps >= args.eval_steps: break

                total_reach_count += session_reaches
                total_accum_reward += session_accum_reward

            # Calculate Average Metrics
            avg_reach_count = total_reach_count
            avg_test_reward = total_accum_reward / args.test_episodes

            writer.add_scalar('Average_Reach_Count', avg_reach_count, global_steps)
            writer.add_scalar('Average_Test_Reward', avg_test_reward, global_steps)
            writer.add_scalar('Comparison/Average_Reach_Count', avg_reach_count, global_steps)
            print(
                f"DDPG Test Episode {episode}: Reach Count = {avg_reach_count:.2f}, Avg Reward = {avg_test_reward:.2f}")

            # Plotting and Saving Metrics
            metrics['test_episodes'].append(episode)
            metrics['test_rewards'].append(avg_reach_count)
            metrics['test_avg_rewards'].append(avg_test_reward)

            lineplot(metrics['test_episodes'], metrics['test_rewards'], 'Reach Count', results_dir)
            lineplot(metrics['test_episodes'], metrics['test_avg_rewards'], 'Average Test Reward', results_dir)

            torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

except KeyboardInterrupt:
    print("DDPG Training interrupted.")

env.close()
writer.close()