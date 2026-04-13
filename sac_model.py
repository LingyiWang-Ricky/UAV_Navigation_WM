import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import random
import collections


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


class ConvolutionalPolicyNetContinuous(nn.Module):
    def __init__(self, action_dim, action_bound, in_channels=6):
        super(ConvolutionalPolicyNetContinuous, self).__init__()
        self.encoder = CNNEncoder(in_channels)

        self.fc1 = nn.Linear(1024, 256)
        self.fc_mu = nn.Linear(256, action_dim)
        self.fc_std = nn.Linear(256, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(self.fc1(x))

        mu = self.fc_mu(x)
        # Reference code uses softplus for std dev
        std = F.softplus(self.fc_std(x))

        dist = Normal(mu, std)
        normal_sample = dist.rsample()  # Reparameterization trick

        # Calculate log_prob
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)

        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        action = action * self.action_bound

        return action, log_prob, torch.tanh(mu) * self.action_bound


class ConvolutionalQValueNetContinuous(nn.Module):
    def __init__(self, action_dim, in_channels=6):
        super(ConvolutionalQValueNetContinuous, self).__init__()
        self.encoder = CNNEncoder(in_channels)
        self.fc1 = nn.Linear(1024 + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, 1)

    def forward(self, x, a):
        x = self.encoder(x)
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


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


class SACContinuous:
    def __init__(self, action_dim, actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma, device, action_bound=1.0):

        self.actor = ConvolutionalPolicyNetContinuous(action_dim, action_bound).to(device)
        self.critic_1 = ConvolutionalQValueNetContinuous(action_dim).to(device)
        self.critic_2 = ConvolutionalQValueNetContinuous(action_dim).to(device)
        self.target_critic_1 = ConvolutionalQValueNetContinuous(action_dim).to(device)
        self.target_critic_2 = ConvolutionalQValueNetContinuous(action_dim).to(device)

        # Initialize target networks
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        # Auto-Alpha Tuning
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau
        self.device = device

    def take_action(self, state, deterministic=False):
        # state: (C, H, W) -> (1, C, H, W)
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        action, _, mean_action = self.actor(state)

        if deterministic:
            return mean_action.detach().cpu().numpy()[0]
        else:
            return action.detach().cpu().numpy()[0]

    def calc_target(self, rewards, next_states, dones):
        next_actions, log_prob, _ = self.actor(next_states)
        entropy = -log_prob

        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)

        # Min(Q1, Q2) + alpha * entropy
        next_value = torch.min(q1_value, q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)

        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(states.size(0), -1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # Update Critic (Q-Networks)
        td_target = self.calc_target(rewards, next_states, dones)

        # Current Q
        current_q1 = self.critic_1(states, actions)
        current_q2 = self.critic_2(states, actions)

        critic_1_loss = F.mse_loss(current_q1, td_target.detach())
        critic_2_loss = F.mse_loss(current_q2, td_target.detach())

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # Update Actor (Policy)
        new_actions, log_prob, _ = self.actor(states)
        entropy = -log_prob

        q1_pi = self.critic_1(states, new_actions)
        q2_pi = self.critic_2(states, new_actions)

        # Maximize (min(Q) + alpha * entropy)  => Minimize -(min(Q) + alpha * entropy)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - torch.min(q1_pi, q2_pi))

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update Alpha (Temperature)
        # Minimize: alpha * (entropy - target_entropy)
        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())

        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # Soft Update Targets
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

        return critic_1_loss.item(), actor_loss.item(), self.log_alpha.exp().item()