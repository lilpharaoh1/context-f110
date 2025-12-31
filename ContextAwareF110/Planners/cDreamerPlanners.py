import numpy as np 
from ContextAwareF110.Utils.cdreamer.dreamerv3 import DreamerV3
from ContextAwareF110.Utils.HistoryStructs import TrainHistory
from ContextAwareF110.Utils.FastTransform import FastTransform
import torch
from numba import njit

from ContextAwareF110.Utils.utils import init_file_struct
from matplotlib import pyplot as plt

class cDreamerTrainer: 
    def __init__(self, run, conf, init=False):
        self.run, self.conf = run, conf
        self.name = run.run_name
        self.path = conf.vehicle_path + run.path + run.run_name 
        if init:
            init_file_struct(self.path)

        self.v_min_plan =  conf.v_min_plan

        self.state = None
        self.action = None
        self.nn_state = None
        self.nn_rssm = None
        self.nn_act = None
        self.nn_context = None

        self.transform = FastTransform(run, conf)

        # Multiagent Info
        self.num_agents = run.num_agents
        self.agent = DreamerV3(self.transform.state_space, self.transform.action_space, run.run_name, max_action=1, window_in=run.window_in, window_out=run.window_out, lr=run.lr)

        self.t_his = TrainHistory(run, conf, cont=not init)
        if not init:
            self.agent.load(self.path)

        self.train = self.agent.train # alias for sss
        # self.save = self.agent.save # alias for sss

    def plan(self, obs, context=None, add_mem_entry=True, **kwargs):
        self.nn_context = np.array(context) if context else np.zeros((1, 2))
        nn_state = self.transform.transform_obs(obs)
        if add_mem_entry:
            self.add_memory_entry(obs)
            
        if obs['state'][3] < self.v_min_plan:
            self.action = np.array([0, 7])
            return self.action

        if np.isnan(nn_state).any():
            print(f"NAN in state: {nn_state}")

        self.nn_state = nn_state # after to prevent call before check for v_min_plan
        # if self.nn_act is None:
        #     print("self.nn_act is None!!!!!")
        self.nn_act, self.nn_rssm = self.agent.act(self.nn_state, self.nn_act, self.nn_rssm, context=self.nn_context, is_first=True if self.nn_act is None else False)
        self.nn_act = self.nn_act.cpu().squeeze(0)

        if np.isnan(self.nn_act).any():
            print(f"NAN in act: {nn_state}")
            raise Exception("Unknown NAN in act")

        self.transform.transform_obs(obs) # to ensure correct PP actions
        self.action = self.transform.transform_action(self.nn_act)

        return self.action 

    def add_memory_entry(self, obs, done=False):
        if self.nn_state is not None:
            self.t_his.add_step_data(obs['reward'])

            eps_name = 'eps_' + str(self.agent.buffer_ptr)
            
            transition = {}
            transition['is_first'] = np.array(1.0 if eps_name not in self.agent.buffer_eps else 0.0) # EMRAN hacky fix
            transition['image'] = np.array(self.nn_state)
            transition['action'] = np.array(self.nn_act)
            transition['reward'] = np.array(obs['reward'])
            transition['context'] = np.array(self.nn_context)
            transition['is_terminal'] = np.array(0.0 if not done else 1.0)
            
            if eps_name not in self.agent.buffer_eps:
                self.agent.buffer_eps[eps_name] = {}
                for k in transition.keys():
                    self.agent.buffer_eps[eps_name][k] = np.array([transition[k]]).reshape(1, -1)
            else:
                for k in self.agent.buffer_eps[eps_name].keys():
                    self.agent.buffer_eps[eps_name][k] = np.append(self.agent.buffer_eps[eps_name][k], transition[k].copy().reshape(1, -1), axis=0)
            # self.agent.buffer.add(self.nn_state, self.nn_act, obs['reward'], False)

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
        
        # self.agent.save(self.path)
        if np.isnan(self.nn_act).any():
            print(f"NAN in act: {self.nn_act}")
            raise Exception("NAN in act")
        if np.isnan(nn_s_prime).any():
            print(f"NAN in state: {nn_s_prime}")
            raise Exception("NAN in state")

        self.add_memory_entry(obs, done=True)
        self.nn_state = None
        self.nn_rssm = None
        self.nn_act = None

    def lap_complete(self):
        pass

    def save_training_data(self):
        self.t_his.print_update(True)
        self.t_his.save_csv_data()
        self.agent.save(self.path)
        if self.t_his.new_best:
            self.agent.save(self.path, best=True)

class cDreamerTester:
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

        self.transform = FastTransform(run, conf)

        self.agent = DreamerV3(self.transform.state_space, self.transform.action_space, run.run_name, max_action=1, window_in=run.window_in, window_out=run.window_out, lr=run.lr)
        checkpoint = torch.load(self.path + '/best_' + run.run_name + ".pth")
        self.agent.load_state_dict(checkpoint['agent_state_dict'])
        self.nn_state = None
        self.nn_act = None
        self.nn_rssm = None
        self.nn_context = None

        self.window_in = run.window_in
        self.n_beams = conf.n_beams
        self.scan_buffer = np.zeros((self.window_in, self.n_beams))

        print(f"Agent loaded: {run.run_name}")

    def plan(self, obs, context=None, **kwargs):
        self.nn_context = np.array(context) if not context is None else np.zeros((1, 2))
        if obs['state'][3] < self.v_min_plan:
            self.action = np.array([0, 7])
            return self.action, np.zeros((108,))
        
        nn_obs = self.transform.transform_obs(obs)


        # if self.scan_buffer.all() ==0: # first reading
        #     for i in range(self.window_in):
        #         self.scan_buffer[i, :] = nn_obs 
        # else:
        #     self.scan_buffer = np.roll(self.scan_buffer, 1, axis=0)
        #     self.scan_buffer[0, :] = nn_obs

        # nn_obs = np.reshape(self.scan_buffer, (self.window_in * self.n_beams))
        # nn_obs = torch.FloatTensor(nn_obs.reshape(1, -1))


        self.nn_state = nn_obs # after to prevent call before check for v_min_plan
        self.nn_act, self.nn_rssm, recon = self.agent.act(self.nn_state, self.nn_act, self.nn_rssm, context=self.nn_context, is_first=self.nn_act is None, video=True)
        self.nn_act = self.nn_act.cpu().squeeze(0)

        if np.isnan(self.nn_act).any():
            print(f"NAN in act: {nn_state}")
            raise Exception("Unknown NAN in act")

        self.transform.transform_obs(obs) # to ensure correct PP actions
        self.action = self.transform.transform_action(self.nn_act)


        return self.action, recon

    def lap_complete(self):
        self.nn_state = None
        self.nn_rssm = None
        self.nn_act = None
