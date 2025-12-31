import numpy as np 
from ContextAwareF110.Utils.TD3 import TD3
from ContextAwareF110.Utils.HistoryStructs import TrainHistory
import torch
from numba import njit

from ContextAwareF110.Utils.utils import init_file_struct
from matplotlib import pyplot as plt


class FastTransform:
    def __init__(self, run, conf):
        self.state_space = conf.n_beams 
        self.action_space = 2
        self.range_finder_scale = conf.range_finder_scale
        self.n_beams = conf.n_beams
        self.max_speed = run.max_speed
        self.max_steer = conf.max_steer


    def transform_obs(self, obs):
        """
        Transforms the observation received from the environment into a vector which can be used with a neural network.
    
        Args:
            obs: observation from env

        Returns:
            nn_obs: observation vector for neural network
        """
            
        scan = np.array(obs['scan']) 

        scaled_scan = scan/self.range_finder_scale
        scan = np.clip(scaled_scan, 0, 1)

        return scan


    def transform_action(self, nn_action):
        steering_angle = nn_action[0] * self.max_steer
        speed = (nn_action[1] + 1) * (self.max_speed  / 2 - 0.5) + 1
        speed = min(speed, self.max_speed) # cap the speed

        action = np.array([steering_angle, speed])

        return action


