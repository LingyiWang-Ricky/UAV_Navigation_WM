import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import collections


# Shared CNN Encoder
class CNNEncoder(nn.Module):
    def __init__(self, in_channels=6):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # Flatten to (B, 1024)
        return x


# Actor (Policy Network)
class PolicyNet(nn.Module):
    def __init__(self, action_dim, action_bound, in_channels=6):
        super(PolicyNet, self).__init__()
        self.encoder = CNNEncoder(in_channels)
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) * self.action_bound


# Critic
class QValueNet(nn.Module):
    def __init__(self, action_dim, in_channels=6):
        super(QValueNet, self).__init__()
        self.encoder = CNNEncoder(in_channels)
        # Critic takes State features + Action as input
        self.fc1 = nn.Linear(1024 + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, 1)

    def forward(self, x, a):
        x = self.encoder(x)
        cat = torch.cat([x, a], dim=1)  # Concatenate state embedding and action
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), np.array(action), reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


# DDPG Agent
class DDPG:
    def __init__(self, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device, in_channels=6):
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.device = device
        self.gamma = gamma
        self.sigma = sigma  # Gaussian noise standard deviation
        self.tau = tau  # Soft update parameter

        # Initialize Networks
        self.actor = PolicyNet(action_dim, action_bound, in_channels).to(device)
        self.critic = QValueNet(action_dim, in_channels).to(device)
        self.target_actor = PolicyNet(action_dim, action_bound, in_channels).to(device)
        self.target_critic = QValueNet(action_dim, in_channels).to(device)

        # Sync target networks initially
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def take_action(self, state, noise=False):
        # state: (6, 64, 64) -> add batch dim -> (1, 6, 64, 64)
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        action = self.actor(state_tensor).item() if self.action_dim == 1 else \
        self.actor(state_tensor).detach().cpu().numpy()[0]

        if noise:
            # Add Gaussian noise for exploration
            action = action + self.sigma * np.random.randn(self.action_dim)
            # Clip to legal range
            action = np.clip(action, -self.action_bound, self.action_bound)

        return action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # Update Critic
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            next_q_values = self.target_critic(next_states, next_actions)
            q_targets = rewards + self.gamma * next_q_values * (1 - dones)

        current_q_values = self.critic(states, actions)
        critic_loss = torch.mean(F.mse_loss(current_q_values, q_targets))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        # Maximize Q value -> Minimize -Q
        actor_loss = -torch.mean(self.critic(states, self.actor(states)))

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft Update Targets
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)

        return critic_loss.item(), actor_loss.item()