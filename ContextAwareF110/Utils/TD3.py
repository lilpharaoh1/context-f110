from os import stat
import numpy as np 
from matplotlib import pyplot as plt
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ContextAwareF110.Utils.Buffer import SmartBuffer

MEMORY_SIZE = 100000


# hyper parameters
BATCH_SIZE = 100
GAMMA = 0.99
tau = 0.005
NOISE = 0.2
NOISE_CLIP = 0.5
EXPLORE_NOISE = 0.1
POLICY_FREQUENCY = 2
POLICY_NOISE = 0.2
WINDOW_SIZE = 5

class Actor(nn.Module):   
    def __init__(self, state_dim, action_dim, max_action, h_size, window_in=1):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim*window_in, h_size)
        self.l2 = nn.Linear(h_size, h_size)
        self.l3 = nn.Linear(h_size, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x)) 
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, h_size, window_in=1):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear((state_dim)*window_in + action_dim, h_size)
        self.l2 = nn.Linear(h_size, h_size)
        self.l3 = nn.Linear(h_size, 1)

        # Q2 architecture
        self.l4 = nn.Linear((state_dim)*window_in + action_dim, h_size)
        self.l5 = nn.Linear(h_size, h_size)
        self.l6 = nn.Linear(h_size, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1



class TD3(object):
    def __init__(self, state_dim, action_dim, name, max_action=1, window_in=1, window_out=1):
        self.name = name
        self.state_dim = state_dim
        self.max_action = max_action
        self.act_dim = action_dim
        self.window_in = window_in
        self.window_out = window_out
        self.state_buff = None

        self.actor = None
        self.actor_target = None
        self.actor_optimizer = None

        self.critic = None
        self.critic_target = None
        self.critic_optimizer = None

        self.replay_buffer = SmartBuffer(state_dim=state_dim, act_dim=action_dim, window_in=window_in, window_out=window_out)

    def create_agent(self, h_size):
        state_dim = self.state_dim
        action_dim = self.act_dim
        max_action = self.max_action
        self.actor = Actor(state_dim, action_dim, max_action, h_size, window_in=self.window_in)
        self.actor_target = Actor(state_dim, action_dim, max_action, h_size, window_in=self.window_in)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = Critic(state_dim, action_dim, h_size, window_in=self.window_in)
        self.critic_target = Critic(state_dim, action_dim, h_size, window_in=self.window_in)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

    def select_action(self, state, noise=0.1):
        return self.act(state, noise=noise)

    def act(self, state, noise=0.1):
        if self.state_buff is None:
            self.state_buff = np.tile(state, (self.window_in, 1))
        else:
            self.state_buff[:-1] = self.state_buff[1:]
            self.state_buff[-1] = state
        state = torch.FloatTensor(self.state_buff.reshape(1, -1))

        action = self.actor(state).data.numpy().flatten()
        if noise != 0: 
            action = (action + np.random.normal(0, noise, size=self.act_dim))
            
        return action.clip(-self.max_action, self.max_action)

    def get_critic_value(self, state, action):
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)

        current_Q1, current_Q2 = self.critic(state[None, :], action[None, :])
        ret = current_Q1.detach().item()

        return ret

    def train(self, iterations=2):
        if self.replay_buffer.size() < BATCH_SIZE * 5:
            return 0
        for it in range(iterations):
            # Sample replay buffer 
            x, u, y, r, d = self.replay_buffer.sample(BATCH_SIZE)

            # Flatten windows
            x = x.reshape(x.shape[0], -1)
            u = u.reshape(u.shape[0], -1)
            y = y.reshape(y.shape[0], -1)
            d = d.reshape(d.shape[0], -1)
            r = r.reshape(r.shape[0], -1)

            # Turn into tensors
            state = torch.FloatTensor(x)
            action = torch.FloatTensor(u)
            next_state = torch.FloatTensor(y)
            done = torch.FloatTensor((1 - d))
            reward = torch.FloatTensor(r)

            # Select action according to policy and add clipped noise (for exploration) 
            noise = torch.FloatTensor(u[:, -self.act_dim:]).data.normal_(0, POLICY_NOISE)
            noise = noise.clamp(-NOISE_CLIP, NOISE_CLIP)
            adpt_next_state = torch.cat((state[:, self.window_out*self.state_dim:], next_state), 1) if self.window_out < self.window_in else next_state
            next_action = (self.actor_target(adpt_next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(adpt_next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * GAMMA * target_Q).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action[:, -self.act_dim:])

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % POLICY_FREQUENCY == 0:
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        total_loss = actor_loss + critic_loss
        
        return total_loss

    def save(self, directory="./saves", best=False):
        if best:
            filename = "best_" + self.name
        else:
            filename = self.name

        torch.save(self.actor, '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic, '%s/%s_critic.pth' % (directory, filename))
        torch.save(self.actor_target, '%s/%s_actor_target.pth' % (directory, filename))
        torch.save(self.critic_target, '%s/%s_critic_target.pth' % (directory, filename))
        self.scan_buff = None

    def load(self, directory="./saves", best=False):
        if best:
            filename = "best_" + self.name
        else:
            filename = self.name

        self.actor = torch.load('%s/%s_actor.pth' % (directory, filename))
        self.critic = torch.load('%s/%s_critic.pth' % (directory, filename))
        self.actor_target = torch.load('%s/%s_actor_target.pth' % (directory, filename))
        self.critic_target = torch.load('%s/%s_critic_target.pth' % (directory, filename))
        self.scan_buff = None

    def try_load(self, load=True, h_size=300, path=None):
        if load:
            try:
                self.load(path)
            except Exception as e:
                print(f"Exception: {e}")
                print(f"Unable to load model")
                pass
        else:
            print(f"Not loading - restarting training")
            self.create_agent(h_size)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
