import torch
import numpy as np
import time
import argparse

from ContextAwareF110.f110_gym.f110_env import F110Env
from ContextAwareF110.Utils.utils import *
from ContextAwareF110.Utils.loaders import select_reward_function, select_agent

from ContextAwareF110.Planners.PurePursuit import PurePursuit
from ContextAwareF110.Planners.FollowTheGap import FTG
from ContextAwareF110.Planners.TD3Planners import TD3Trainer, TD3Tester
from ContextAwareF110.Planners.SACPlanners import SACTrainer, SACTester
from ContextAwareF110.Planners.DreamerV2Planners import DreamerV2Trainer, DreamerV2Tester
from ContextAwareF110.Planners.DreamerV3Planners import DreamerV3Trainer, DreamerV3Tester
from ContextAwareF110.Planners.cDreamerPlanners import cDreamerTrainer, cDreamerTester
from ContextAwareF110.Planners.cbDreamerPlanners import cbDreamerTrainer, cbDreamerTester

from ContextAwareF110.Utils.RewardSignals import *
from ContextAwareF110.Utils.StdTrack import StdTrack

from ContextAwareF110.Utils.HistoryStructs import VehicleStateHistory
from ContextAwareF110.TestSimulation import TestSimulation

# settings
SHOW_TRAIN = False
# SHOW_TRAIN = True
VERBOSE = True
NON_TRAINABLE = []

class TrainSimulation(TestSimulation):
    def __init__(self, run_file):
        super().__init__(run_file)

        self.reward = None
        self.previous_observation = None


    def run_training_evaluation(self):
        # print(self.run_data)
        for run in self.run_data:
            print("Performing run :", run)
            seed = run.random_seed + 10*run.n
            np.random.seed(seed) # repetition seed
            # torch.set_deterministic(True)
            torch.use_deterministic_algorithms(True)
            torch.manual_seed(seed)

            self.env = F110Env(
                map=run.map_name,
                num_agents=run.num_agents
                )
            self.map_name = run.map_name
            self.num_agents = run.num_agents
            self.target_position = run.target_position
            self.start_train_steps = run.start_train_steps
            self.start_poses = run.start_poses
            self.n_train_steps = run.n_train_steps
            assert self.num_agents == len(run.adversaries) + 1, "Number of agents != number of adversaries + 1"

            #train
            self.std_track = StdTrack(run.map_name, num_agents=run.num_agents)
            self.reward = select_reward_function(run, self.conf, self.std_track)

            self.target_planner = select_agent(run, self.conf, run.architecture, init=(not self.start_train_steps > 0))

            self.vehicle_state_history = [VehicleStateHistory(run, f"Training/agent_{agent_id}") for agent_id in range(self.num_agents)]

            self.completed_laps = 0
            self.places = []
            self.progresses = []

            self.run_training(run)

            #Test
            self.target_planner = select_agent(run, self.conf, run.architecture, train=False, init=False)

            self.vehicle_state_history = [VehicleStateHistory(run, f"Testing/agent_{agent_id}") for agent_id in range(self.num_agents)]

            self.n_test_laps = run.n_test_laps

            self.lap_times = []
            self.completed_laps = 0
            self.places = []
            self.progresses = []
            self.start_poses = "ordered" # start poses should be ordered for testing

            self.run_testing(run)

            conf = vars(self.conf)
            conf['path'] = run.path
            conf['run_name'] = run.run_name
            save_conf_dict(conf, "TrainingConfig")

            self.env.close_rendering()

    def run_training(self, run):
        assert self.env != None, "No environment created"
        start_time = time.time()
        print(f"Starting Baseline Training: {self.target_planner.name}")
        # assert not type(agent) in NON_TRAINABLE, f"{type(agent)} is a non-trainable agent type"
        
        lap_counter, crash_counter = 0, 0
        observations = self.reset_simulation()
        target_obs = observations[0]

        
        if len(run.adversaries) == 0:
            context_info = [0.0, 0.0]
        else:
            speed_val, steer_val = run.context_info[:2]
            speed_c, steer_c = np.random.uniform(-speed_val, speed_val), np.random.uniform(-steer_val, steer_val)
            context_info = [speed_c, steer_c] 
        self.adv_planners = [select_agent(run, self.conf, architecture, init=False, context_info=context_info) for architecture in run.adversaries] 

        context = context_info #if len(run.adversaries) > 0 else None
        for i in range(self.start_train_steps, self.n_train_steps):
            self.prev_obs = observations # used for calculating reward, so only wanst target_obs
            target_action = self.target_planner.plan(target_obs, context=context)
            # target_action['action'] = np.array([0.0, 1.8]) + np.random.normal(scale=np.array([0.025, 0.2]))
            # target_action['action'] = np.array([0.0, 0.0])
            if len(self.adv_planners) > 0:
                adv_actions = np.array([adv.plan(obs)['action'] if not obs['colision_done'] else [0.0, 0.0] for (adv, obs) in zip(self.adv_planners, observations[1:])])
                # adv_actions = np.array([np.array([0.0, 1.8]) + np.random.normal(scale=np.array([0.025, 0.2])) if not obs['colision_done'] else [0.0, 0.0] for (adv, obs) in zip(self.adv_planners, observations[1:])])
                # adv_actions = np.array([np.array([0.0, 0.0]) if not obs['colision_done'] else [0.0, 0.0] for (adv, obs) in zip(self.adv_planners, observations[1:])])
                actions = np.concatenate((target_action['action'].reshape(1, -1), adv_actions), axis=0)
            else:
                actions = target_action['action'].reshape(1, -1)
            observations = self.run_step(actions)
            target_obs = observations[0]

            self.target_planner.t_his.add_overtaking(target_obs['overtaking'])

            if lap_counter > 0: # don't train on first lap.
                self.target_planner.agent.train()

            if SHOW_TRAIN: self.env.render('human_fast')

            if target_obs['lap_done'] or target_obs['colision_done'] or target_obs['current_laptime'] > self.conf.max_laptime:
                self.target_planner.done_entry(target_obs)

                if target_obs['lap_done']:
                    if VERBOSE: print(f"{i}::Lap Complete {self.completed_laps} -> FinalR: {target_obs['reward']:.2f} -> LapTime {target_obs['current_laptime']:.2f} -> TotalReward: {self.target_planner.t_his.rewards[self.target_planner.t_his.ptr-1]:.2f} -> Progress: {target_obs['progress']:.2f}")

                    self.completed_laps += 1

                elif target_obs['colision_done'] or self.std_track.check_done(0): # target agent_id = 0

                    if VERBOSE: print(f"{i}::Crashed -> FinalR: {target_obs['reward']:.2f} -> LapTime {target_obs['current_laptime']:.2f} -> TotalReward: {self.target_planner.t_his.rewards[self.target_planner.t_his.ptr-1]:.2f} -> Progress: {target_obs['progress']:.2f}")
                    crash_counter += 1
                
                else:
                    print(f"{i}::LapTime Exceeded -> FinalR: {target_obs['reward']:.2f} -> LapTime {target_obs['current_laptime']:.2f} -> TotalReward: {self.target_planner.t_his.rewards[self.target_planner.t_his.ptr-1]:.2f} -> Progress: {target_obs['progress']:.2f}")

                if self.vehicle_state_history:
                    for vsh in self.vehicle_state_history: 
                        vsh.save_history(f"train_{lap_counter}", test_map=self.map_name)
                lap_counter += 1

                observations = self.reset_simulation()
                self.target_planner.save_training_data()

                # Reinstatiate adversaries with new context (if necessary)
                if len(run.adversaries) == 0:
                    context_info = [0.0, 0.0]
                else:
                    speed_val, steer_val = run.context_info[:2]
                    speed_c, steer_c = np.random.uniform(-speed_val, speed_val), np.random.uniform(-steer_val, steer_val)
                    context_info = [speed_c, steer_c] 
                self.adv_planners = [select_agent(run, self.conf, architecture, init=False, context_info=context_info) for architecture in run.adversaries] 
                context = context_info # if len(run.adversaries) > 0 else None

        train_time = time.time() - start_time
        print(f"Finished Training: {self.target_planner.name} in {train_time} seconds")
        print(f"Crashes: {crash_counter}")


        print(f"Training finished in: {time.time() - start_time}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run-file', default='dev', type=str)
    args = parser.parse_args()

    sim = TrainSimulation(args.run_file)
    sim.run_training_evaluation()


if __name__ == '__main__':
    main()



