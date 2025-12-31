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
EPSILON = 1e-6

# hyper parameters
BATCH_SIZE = 100
GAMMA = 0.85 # 0.99
tau = 0.005
ALPHA = 0.2

NOISE = 0.2
NOISE_CLIP = 0.5
EXPLORE_NOISE = 0.1
POLICY_FREQUENCY = 2
POLICY_NOISE = 0.5
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
WINDOW_SIZE = 5

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, window_in=1, window_out=1):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(window_in*num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(window_in*num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, max_action, hidden_dim, action_space=None, window_in=1, window_out=1):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs*window_in, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        self.max_action = max_action
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, max_action, hidden_dim, action_space=None, window_in=1, window_out=1):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(window_in*num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        self.max_action=max_action
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-NOISE_CLIP, NOISE_CLIP)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)




class SAC(object):
    def __init__(self, state_dim, action_dim, name, max_action=1, window_in=1, window_out=1, policy_type="Gaussian", entropy_tuning=True, lr=None, gamma=None):
        self.name = name
        self.policy_type=policy_type # From pytorch-soft-actor-critic
        self.automatic_entropy_tuning  = entropy_tuning # From pytorch-soft-actor-critic 
        self.state_dim = state_dim
        self.max_action = max_action
        self.act_dim = action_dim
        self.window_in = window_in
        self.window_out = window_out
        self.lr = 1e-3 if lr is None else lr
        self.gamma = GAMMA if gamma is None else gamma
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

        self.critic = QNetwork(state_dim, action_dim, h_size, window_in=self.window_in, window_out=self.window_out)
        self.critic_target = QNetwork(state_dim, action_dim, h_size, window_in=self.window_in, window_out=self.window_out)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.alpha = ALPHA
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor((action_dim, 1))).item() # EMRAN, may have to be (2,1) instead of (2,)
                self.log_alpha = torch.zeros(1, requires_grad=True)
                self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.lr)

            self.policy = GaussianPolicy(state_dim, action_dim, max_action, h_size, window_in=self.window_in, window_out=self.window_out)
            self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(state_dim, action_dim, max_action, h_size, window_in=self.window_in, window_out=self.window_out)
            self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    def select_action(self, state, noise=0.1):
        return self.act(state, noise=noise)

    def act(self, state, noise=0.1):
        if self.state_buff is None:
            self.state_buff = np.tile(state, (self.window_in, 1))
        else:
            self.state_buff[:-1] = self.state_buff[1:]
            self.state_buff[-1] = state
        state = torch.FloatTensor(self.state_buff.reshape(1, -1))

        action, _, _ = self.policy.sample(state) # action, log_prob, mean
        action = action.data.numpy().flatten()
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
        # Sample a batch from memory
        if self.replay_buffer.size() < BATCH_SIZE * self.window_in:
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

            with torch.no_grad():
                # Select action according to policy and add clipped noise (for exploration) 
                noise = torch.FloatTensor(u[:, -self.act_dim:]).data.normal_(0, POLICY_NOISE)
                noise = noise.clamp(-NOISE_CLIP, NOISE_CLIP)                    
                adpt_next_state = torch.cat((state[:, self.window_out*self.state_dim:], next_state), 1) if self.window_out < self.window_in else next_state
                next_action, next_state_log_pi, _ = self.policy.sample(adpt_next_state)
                next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(adpt_next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_state_log_pi
                target_Q = reward + (done * self.gamma * target_Q).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action[:, -self.act_dim:])

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            pi, log_pi, _ = self.policy.sample(state)

            qf1_pi, qf2_pi = self.critic(state, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            pi, log_pi, _ = self.policy.sample(state)


            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()


                self.alpha = self.log_alpha.exp()
                alpha_tlogs = self.alpha.clone() # For TensorboardX logs
            else:
                alpha_loss = torch.tensor(0.)
                alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

            pi, log_pi, _ = self.policy.sample(state)

            # Every POLICY FREQUENCY, update critic weights
            if it % POLICY_FREQUENCY == 0:
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                pi, log_pi, _ = self.policy.sample(state)

        total_loss = policy_loss + critic_loss
        
        return total_loss


    def save(self, directory="./saves", best=False):
        if best:
            filename = "best_" + self.name
        else:
            filename = self.name
        
        torch.save(self.policy, '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic, '%s/%s_critic.pth' % (directory, filename))
        torch.save(self.critic_target, '%s/%s_critic_target.pth' % (directory, filename))
        self.scan_buff = None

    def load(self, directory="./saves", best=False):
        if best:
            filename = "best_" + self.name
        else:
            filename = self.name

        self.policy = torch.load('%s/%s_actor.pth' % (directory, filename))
        self.critic = torch.load('%s/%s_critic.pth' % (directory, filename))
        self.critic_target = torch.load('%s/%s_critic_target.pth' % (directory, filename))
        self.scan_buff = None

        print("Agent Loaded")

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

        self.policy_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
