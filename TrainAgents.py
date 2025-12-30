import torch
import numpy as np
import time

from TrajectoryAidedLearning.f110_gym.f110_env import F110Env
from TrajectoryAidedLearning.Utils.utils import *

from TrajectoryAidedLearning.Planners.PurePursuit import PurePursuit
from TrajectoryAidedLearning.Planners.DisparityExtender import DispExt
from TrajectoryAidedLearning.Planners.TD3Planners import TD3Trainer, TD3Tester
from TrajectoryAidedLearning.Planners.SACPlanners import SACTrainer, SACTester
from TrajectoryAidedLearning.Planners.DreamerV2Planners import DreamerV2Trainer, DreamerV2Tester
from TrajectoryAidedLearning.Planners.DreamerV3Planners import DreamerV3Trainer, DreamerV3Tester
from TrajectoryAidedLearning.Planners.cDreamerPlanners import cDreamerTrainer, cDreamerTester
from TrajectoryAidedLearning.Planners.cbDreamerPlanners import cbDreamerTrainer, cbDreamerTester
from TrajectoryAidedLearning.Planners.cobDreamerPlanners import cobDreamerTrainer, cobDreamerTester
from TrajectoryAidedLearning.Planners.cfDreamerPlanners import cfDreamerTrainer, cfDreamerTester

from TrajectoryAidedLearning.Utils.RewardSignals import *
from TrajectoryAidedLearning.Utils.StdTrack import StdTrack

from TrajectoryAidedLearning.Utils.HistoryStructs import VehicleStateHistory
from TrajectoryAidedLearning.TestSimulation import TestSimulation

# settings
# SHOW_TRAIN = False
SHOW_TRAIN = True
VERBOSE = True

NON_TRAINABLE = []

# TODO move to utils
def select_reward_function(run, conf, std_track):
    reward = run.reward
    if reward == "Progress":
        reward_function = ProgressReward(std_track)
    elif reward == "Cth": 
        reward_function = CrossTrackHeadReward(std_track, conf)
    elif reward == "TAL":
        reward_function = TALearningReward(conf, run)
    else: raise Exception("Unknown reward function: " + reward)
        
    return reward_function

# TODO move to utils
def select_agent(run, conf, architecture, train=True, init=False, ma_info=[0.0, 0.0]):
    agent_type = architecture if architecture is not None else "TD3"
    if agent_type == "PP":
        agent = PurePursuit(run, conf, init=init, ma_info=ma_info) 
    elif agent_type == "TD3":
        agent = TD3Trainer(run, conf, init=init) if train else TD3Tester(run, conf)
    elif agent_type == "SAC":
        agent = SACTrainer(run, conf, init=init) if train else SACTester(run, conf)
    elif agent_type == "DreamerV2":
        agent = DreamerV2Trainer(run, conf) if train else DreamerV2Tester(run, conf)
    elif agent_type == "DreamerV3":
        agent = DreamerV3Trainer(run, conf, init=init) if train else DreamerV3Tester(run, conf)
    elif agent_type == "cDreamer":
        agent = cDreamerTrainer(run, conf, init=init) if train else cDreamerTester(run, conf)
    elif agent_type == "cbDreamer":
        agent = cbDreamerTrainer(run, conf, init=init) if train else cbDreamerTester(run, conf)
    elif agent_type == "cobDreamer":
        agent = cobDreamerTrainer(run, conf, init=init) if train else cobDreamerTester(run, conf)
    elif agent_type == "cfDreamer":
        agent = cfDreamerTrainer(run, conf, init=init) if train else cfDreamerTester(run, conf)
    elif agent_type == "DispExt":
        agent = DispExt(run, conf, ma_info=ma_info)
    else: raise Exception("Unknown agent type: " + agent_type)

    return agent


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
            ma_info = [0.0, 0.0]
        else:
            speed_val, la_val = run.ma_info[:2]
            speed_c, la_c = np.random.uniform(-speed_val, speed_val), np.random.uniform(-la_val, la_val)
            ma_info = [speed_c, la_c] 
        self.adv_planners = [select_agent(run, self.conf, architecture, init=False, ma_info=ma_info) for architecture in run.adversaries] 

        context = ma_info #if len(run.adversaries) > 0 else None
        for i in range(self.start_train_steps, self.n_train_steps):
            self.prev_obs = observations # used for calculating reward, so only wanst target_obs
            target_action = self.target_planner.plan(target_obs, context=context)
            # target_action = np.array([0.0, 1.8]) + np.random.normal(scale=np.array([0.025, 0.2]))
            # target_action = np.array([0.0, 0.0])
            # print(f"colision_done : {[obs['colision_done'] for obs in observations]}")
            if len(self.adv_planners) > 0:
                adv_actions = np.array([adv.plan(obs) if not obs['colision_done'] else [0.0, 0.0] for (adv, obs) in zip(self.adv_planners, observations[1:])])
                # adv_actions = np.array([np.array([0.0, 1.8]) + np.random.normal(scale=np.array([0.025, 0.2])) if not obs['colision_done'] else [0.0, 0.0] for (adv, obs) in zip(self.adv_planners, observations[1:])])
                # adv_actions = np.array([np.array([0.0, 0.0]) if not obs['colision_done'] else [0.0, 0.0] for (adv, obs) in zip(self.adv_planners, observations[1:])])
                actions = np.concatenate((target_action.reshape(1, -1), adv_actions), axis=0)
            else:
                actions = target_action.reshape(1, -1)
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
                    ma_info = [0.0, 0.0]
                else:
                    speed_val, la_val = run.ma_info[:2]
                    speed_c, la_c = np.random.uniform(-speed_val, speed_val), np.random.uniform(-la_val, la_val)
                    ma_info = [speed_c, la_c] 
                self.adv_planners = [select_agent(run, self.conf, architecture, init=False, ma_info=ma_info) for architecture in run.adversaries] 
                context = ma_info # if len(run.adversaries) > 0 else None

        train_time = time.time() - start_time
        print(f"Finished Training: {self.target_planner.name} in {train_time} seconds")
        print(f"Crashes: {crash_counter}")


        print(f"Training finished in: {time.time() - start_time}")



def main():
    # run_file = "dev"
    # run_file = "SAC_lr"
    # run_file = "SAC_gamma"
    # run_file = "SAC_singleagent"
    # run_file = "SAC_multiagent_stationary"
    # run_file = "SAC_multiagent_nonstationary"
    # run_file = "sac_multiagent_classic"
    # run_file = "sac_multiagent_classic_gbr"
    # run_file = "sac_multiagent_dispext"
    # run_file = "dreamerv3_lr"
    # run_file = "dreamerv3_singleagent"
    # run_file = "dreamerv3_multiagent_stationary"
    # run_file = "dreamerv3_multiagent_nonstationary"
    # run_file = "dreamerv3_multiagent_classic"
    # run_file = "dreamerv3_multiagent_classic_gbr"
    # run_file = "dreamerv3_multiagent_dispext"
    # run_file = "cdreamer_singleagent"
    # run_file = "cdreamer_multiagent_stationary"
    # run_file = "cdreamer_multiagent_nonstationary"
    # run_file = "cdreamer_multiagent_classic"
    # run_file = "cdreamer_multiagent_classic_gbr"
    # run_file = "cdreamer_multiagent_dispext"
    # run_file = "cfdreamer_multiagent_nonstationary"
    # run_file = "cbdreamer_multiagent_nonstationary"
    # run_file = "cbdreamer_multiagent_classic"
    # run_file = "cbdreamer_multiagent_dispext"
    # run_file = "cbdreamer_multiagent_classic2"
    # run_file = "cobdreamer_multiagent_nonstationary"

    run_file = "dev"
    sim = TrainSimulation(run_file)
    sim.run_training_evaluation()


if __name__ == '__main__':
    main()



