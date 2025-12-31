from os import stat
import numpy as np 
from matplotlib import pyplot as plt
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SmartBuffer(object):
    def __init__(self, max_size=1000000, state_dim=14, act_dim=1, window_in=1, window_out=1):     
        self.max_size = max_size
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.window_in = window_in
        self.window_out = window_out
        self.window_size = window_in + window_out
        self.window_entries = 0
        self.ptr = 0

        self.state_buff = np.empty((self.window_size, state_dim))
        self.act_buff = np.empty((self.window_size, act_dim))
        self.reward_buff = np.empty((self.window_size, 1))
        self.done_buff = np.empty((self.window_size, 1))

        self.states = np.empty((max_size, *self.state_buff.shape))
        self.actions = np.empty((max_size, *self.act_buff.shape))
        self.rewards = np.empty((max_size, *self.reward_buff.shape))
        self.dones = np.empty((max_size, *self.done_buff.shape))

    def add(self, s, a, r, d):
        self.fill_buff(s, a, r, d)
        if self.window_entries == self.window_size:
            self.states[self.ptr] = self.state_buff
            self.actions[self.ptr] = self.act_buff
            self.rewards[self.ptr] = self.reward_buff
            self.dones[self.ptr] = self.done_buff
            self.ptr += 1
        if d:
            self.clear_window()
        
        if self.ptr == self.max_size-1: self.ptr = 0 #! crisis
    
    def fill_buff(self, s, a, r, d):
        # move buff back one
        self.state_buff[:-1] = self.state_buff[1:]
        self.act_buff[:-1] = self.act_buff[1:]
        self.reward_buff[:-1] = self.reward_buff[1:]
        self.done_buff[:-1] = self.done_buff[1:]

        # Fill in new values
        self.state_buff[-1] = s
        self.act_buff[-1] = a
        self.reward_buff[-1] = r
        self.done_buff[-1] = d

        # increment window_entries
        self.window_entries = min(self.window_entries + 1, self.window_size)

    def clear_window(self):
        # Reset window
        self.state_buff = np.empty_like(self.state_buff)
        self.act_buff = np.empty_like(self.act_buff)
        self.reward_buff = np.empty_like(self.reward_buff)
        self.done_buff = np.empty_like(self.done_buff)

        # Reset window entries
        self.window_entries = 0

    def sample(self, batch_size, return_next_actions=False):
        ind = np.random.randint(0, self.ptr-1, size=batch_size)
        states = np.empty((batch_size, self.window_in, self.state_dim))
        actions = np.empty((batch_size, self.window_in, self.act_dim))
        next_states = np.empty((batch_size, self.window_out, self.state_dim))
        next_actions = np.empty((batch_size, self.window_out, self.act_dim))
        rewards = np.empty((batch_size, self.window_out, 1))
        dones = np.empty((batch_size, self.window_out, 1))

        for i, j in enumerate(ind):
            states[i] = self.states[j, :self.window_in]
            actions[i] = self.actions[j, :self.window_in]
            next_states[i] = self.states[j, self.window_in:]
            next_actions[i] = self.actions[j, self.window_in:]
            rewards[i] = self.rewards[j, self.window_in:]
            dones[i] = self.dones[j, self.window_in:]


        if return_next_actions:
            return states, actions, next_states, next_actions, rewards, dones
        return states, actions, next_states, rewards, dones

    def size(self):
        return self.ptr