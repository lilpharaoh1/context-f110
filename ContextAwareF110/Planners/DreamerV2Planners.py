import numpy as np 
from ContextAwareF110.Utils.dreamerv2.dreamerv2 import DreamerV2
from ContextAwareF110.Utils.HistoryStructs import TrainHistory
from ContextAwareF110.Utils.FastTransform import FastTransform
import torch
from numba import njit

from ContextAwareF110.Utils.utils import init_file_struct
from matplotlib import pyplot as plt

class DreamerV2Trainer: 
    def __init__(self, run, conf):
        self.run, self.conf = run, conf
        self.name = run.run_name
        self.path = conf.vehicle_path + run.path + run.run_name 
        init_file_struct(self.path)

        self.v_min_plan =  conf.v_min_plan

        self.state = None
        self.action = None
        self.nn_state = None
        self.nn_rssm = None
        self.nn_act = None

        self.transform = FastTransform(run, conf)

        self.agent = DreamerV2(self.transform.state_space, self.transform.action_space, run.run_name, max_action=1, window_in=run.window_in, window_out=run.window_out)
        self.agent.create_agent()

        self.t_his = TrainHistory(run, conf)

        self.train = self.agent.train # alias for sss
        # self.save = self.agent.save # alias for sss

    def plan(self, obs, add_mem_entry=True, **kwargs):
        nn_state = self.transform.transform_obs(obs)
        if add_mem_entry:
            self.add_memory_entry(obs)
            
        if obs['state'][3] < self.v_min_plan:
            self.action = np.array([0, 7])
            return {
                'action': self.action
            }  

        if np.isnan(nn_state).any():
            print(f"NAN in state: {nn_state}")

        self.nn_state = nn_state # after to prevent call before check for v_min_plan
        self.nn_act = self.agent.act(self.nn_state, self.nn_act, self.nn_rssm).squeeze(0)

        if np.isnan(self.nn_act).any():
            print(f"NAN in act: {nn_state}")
            raise Exception("Unknown NAN in act")

        self.transform.transform_obs(obs) # to ensure correct PP actions
        self.action = self.transform.transform_action(self.nn_act)

        return {
            'action': self.action
        } 

    def add_memory_entry(self, obs):
        if self.nn_state is not None:
            self.t_his.add_step_data(obs['reward'])

            self.agent.buffer.add(self.nn_state, self.nn_act, obs['reward'], False)

    def intervention_entry(self, obs):
        """
        To be called when the supervisor intervenes.
        The lap isn't complete, but it is a terminal state
        """
        nn_s_prime = self.transform.transform_obs(obs)
        if self.nn_state is None:
            return
        self.t_his.add_step_data(obs['reward'])

        self.agent.buffer.add(self.nn_state, self.nn_act, obs['reward'], True)

    def done_entry(self, obs):
        """
        To be called when ep is done.
        """
        nn_s_prime = self.transform.transform_obs(obs)

        self.t_his.lap_done(obs['reward'], obs['progress'], False)
        if self.nn_state is None:
            print(f"Crashed on first step: RETURNING")
            return
        
        self.agent.save(self.path)
        if np.isnan(self.nn_act).any():
            print(f"NAN in act: {self.nn_act}")
            raise Exception("NAN in act")
        if np.isnan(nn_s_prime).any():
            print(f"NAN in state: {nn_s_prime}")
            raise Exception("NAN in state")

        self.agent.buffer.add(self.nn_state, self.nn_act, obs['reward'], True)
        self.nn_state = None

    def lap_complete(self):
        pass

    def save_training_data(self):
        self.t_his.print_update(True)
        self.t_his.save_csv_data()
        self.agent.save(self.path)

class DreamerV2Tester:
    def __init__(self, run, conf):
        """
        Testing vehicle using the reference modification navigation stack

        Args:
            agent_name: name of the agent for saving and reference
            conf: namespace with simulation parameters
            mod_conf: namespace with modification planner parameters
        """
        self.run, self.conf = run, conf
        self.v_min_plan = conf.v_min_plan
        self.path = conf.vehicle_path + run.path + run.run_name 

        self.actor = torch.load(self.path + '/' + run.run_name + "_actor.pth")

        self.transform = FastTransform(run, conf)
        self.window_in = run.window_in
        self.n_beams = conf.n_beams
        self.scan_buffer = np.zeros((self.window_in, self.n_beams))

        print(f"Agent loaded: {run.run_name}")

    def plan(self, obs, **kwargs):
        if obs['state'][3] < self.v_min_plan:
            self.action = np.array([0, 7])
            return {
                'action': self.action
            }  
        
        nn_obs = self.transform.transform_obs(obs)


        if self.scan_buffer.all() ==0: # first reading
            for i in range(self.window_in):
                self.scan_buffer[i, :] = nn_obs 
        else:
            self.scan_buffer = np.roll(self.scan_buffer, 1, axis=0)
            self.scan_buffer[0, :] = nn_obs

        nn_obs = np.reshape(self.scan_buffer, (self.window_in * self.n_beams))
        nn_obs = torch.FloatTensor(nn_obs.reshape(1, -1))
        nn_action = self.actor(nn_obs).data.numpy().flatten()
        self.nn_act = nn_action

        self.action = self.transform.transform_action(nn_action)

        return {
            'action': self.action
        }    

    def lap_complete(self):
        pass
