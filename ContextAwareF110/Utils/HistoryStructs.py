import os, shutil
import csv
import glob
import numpy as np
from matplotlib import pyplot as plt
from ContextAwareF110.Utils.utils import *
from matplotlib.ticker import MultipleLocator


SIZE = 20000


def plot_data(values, moving_avg_period=10, title="Results", figure_n=2):
    plt.figure(figure_n)
    plt.clf()        
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)

    out = ewma(values)
    plt.plot(out)
    
    # if len(values) >= moving_avg_period:
    #     moving_avg = true_moving_average(values, moving_avg_period)
    #     plt.plot(moving_avg)    
    # if len(values) >= moving_avg_period*5:
    #     moving_avg = true_moving_average(values, moving_avg_period * 5)
    #     plt.plot(moving_avg)    

    # plt.pause(0.001)


class TrainHistory():
    def __init__(self, run, conf, cont=False, save_key=None) -> None:
        self.path = conf.vehicle_path + run.path +  run.run_name 

        # training data
        self.ptr = 0
        self.lengths = np.zeros(SIZE)
        self.rewards = np.zeros(SIZE) 
        self.progresses = np.zeros(SIZE) 
        self.pos_overtaking = np.zeros(SIZE) 
        self.neg_overtaking = np.zeros(SIZE) 
        self.overtaking = np.zeros(SIZE) 
        self.laptimes = np.zeros(SIZE) 
        self.t_counter = 0 # total steps
        
        # best performance
        self.best_reward = -999.0
        self.best_progress = -999.0
        self.best_overtaking = -999.0
        self.save_key = save_key if not save_key is None else "reward"
        self.new_best = False

        # espisode data
        self.ep_counter = 0 # ep steps
        self.ep_reward = 0
        self.ep_pos_overtaking = 0
        self.ep_neg_overtaking = 0

        if cont:
            self.load_history()

    def add_overtaking(self, new_o):
        if new_o > 0:
            self.ep_pos_overtaking += new_o
        elif new_o < 0:
            self.ep_neg_overtaking += new_o
    
    def add_step_data(self, new_r):
        self.ep_reward += new_r
        self.ep_counter += 1
        self.t_counter += 1 

    def lap_done(self, reward, progress, show_reward=False):
        self.add_step_data(reward)
        self.lengths[self.ptr] = self.ep_counter
        self.rewards[self.ptr] = self.ep_reward
        self.progresses[self.ptr] = progress
        self.pos_overtaking[self.ptr] = self.ep_pos_overtaking
        self.neg_overtaking[self.ptr] = self.ep_neg_overtaking
        self.overtaking[self.ptr] = self.ep_pos_overtaking + self.ep_neg_overtaking
        self.ptr += 1

        if show_reward:
            plt.figure(8)
            plt.clf()
            plt.plot(self.ep_rewards)
            plt.plot(self.ep_rewards, 'x', markersize=10)
            plt.title(f"Ep rewards: total: {self.ep_reward:.4f}")
            plt.ylim([-1.1, 1.5])
            plt.pause(0.0001)

        self.ep_counter = 0
        self.ep_reward = 0
        self.ep_rewards = []
        self.ep_pos_overtaking = 0
        self.ep_neg_overtaking = 0

    def print_update(self, plot_reward=True):
        if self.ptr < 10:
            return
        
        mean10 = np.mean(self.rewards[self.ptr-10:self.ptr])
        mean100 = np.mean(self.rewards[max(0, self.ptr-100):self.ptr])
        # print(f"Run: {self.t_counter} --> Moving10: {mean10:.2f} --> Moving100: {mean100:.2f}  ")
        
        if plot_reward:
            # raise NotImplementedError
            plot_data(self.rewards[0:self.ptr], figure_n=2)

    def save_csv_data(self):
        data = []
        ptr = self.ptr  #exclude the last entry
        for i in range(ptr): 
            data.append([i, self.rewards[i], self.lengths[i], self.progresses[i], self.overtaking[i], self.laptimes[i]])
        save_csv_array(data, self.path + "/training_data_episodes.csv")

        plot_data(self.rewards[0:ptr], figure_n=2)
        plt.figure(2)
        plt.savefig(self.path + "/training_rewards_episodes.png")

        t_steps = np.cumsum(self.lengths[0:ptr])/100
        plt.figure(3)
        plt.clf()

        # Plot & Save Reward
        plt.plot(t_steps, self.rewards[0:ptr], '.', color='darkblue', markersize=4)
        reward_ewma = ewma(self.rewards[0:ptr])
        plt.plot(t_steps, reward_ewma, linewidth='4', color='r')
        max_reward, idx_reward = np.round(np.max(reward_ewma), 5), np.argmax(reward_ewma)
        self.new_best = max_reward > self.best_reward if self.save_key == "reward" else self.new_best
        self.best_reward = max_reward
        plt.plot(t_steps[idx_reward], max_reward, 'x', color='black', markersize=8)
                
        plt.xlabel("Training Steps (x100)")
        plt.ylabel("Reward per Episode")

        plt.tight_layout()
        plt.grid()
        plt.savefig(self.path + "/training_rewards_steps.png")

        plt.figure(4)
        plt.clf()


        # Plot & Save Progress
        plt.plot(t_steps, self.progresses[0:ptr], '.', color='darkblue', markersize=4)
        progress_ewma = ewma(self.progresses[0:ptr])
        plt.plot(t_steps, progress_ewma, linewidth='4', color='r')
        max_progress, idx_progress = np.round(np.max(progress_ewma), 5), np.argmax(progress_ewma)
        self.new_best = max_progress > self.best_progress if self.save_key == "progress" else self.new_best
        self.best_progress = max_progress
        plt.plot(t_steps[idx_progress], max_progress, 'x', color='black', markersize=8)

        plt.xlabel("Training Steps (x100)")
        plt.ylabel("Progress")

        plt.tight_layout()
        plt.grid()
        plt.savefig(self.path + "/training_progress_steps.png")

        plt.figure(5)
        plt.clf()


        # Plot & Save Overtaking
        plt.plot(t_steps, self.overtaking[0:ptr], '.', color='darkblue', markersize=4)
        overtaking_ewma = ewma(self.overtaking[0:ptr])
        plt.plot(t_steps, overtaking_ewma, linewidth='4', color='r')
        max_overtaking, idx_overtaking = np.round(np.max(overtaking_ewma), 5), np.argmax(overtaking_ewma)
        self.new_best = max_overtaking > self.best_overtaking if self.save_key == "overtaking" else self.new_best
        self.best_overtaking = max_overtaking
        plt.plot(t_steps[idx_overtaking], max_overtaking, 'x', color='black', markersize=8)
                

        plt.xlabel("Training Steps (x100)")
        plt.ylabel("Overtaking per Episode")

        plt.tight_layout()
        plt.grid()
        plt.savefig(self.path + "/training_overtaking_steps.png")

        # plt.close()
    
    def load_history(self):
        filename = self.path + "/training_data_episodes.csv"
        with open(filename, 'r') as file:
            data = csv.reader(file)
            for i, row in enumerate(data):
                # data.append([i, self.rewards[i], self.lengths[i], self.progresses[i], self.overtaking[i], self.laptimes[i]])
                self.ptr = int(row[0]) + 1
                self.rewards[i] = float(row[1])
                self.lengths[i] = float(row[2])
                self.progresses[i] = float(row[3])
                self.overtaking[i] = float(row[4])
                self.laptimes[i] = float(row[5])


class VehicleStateHistory:
    def __init__(self, run, folder):
        self.vehicle_name = run.run_name
        tmp_path = "Data/Vehicles/" + run.path + run.run_name
        indv_dirs = folder.split('/')
        for indv_dir in indv_dirs:
            tmp_path = tmp_path + "/" + indv_dir
            if not os.path.exists(tmp_path):
                os.mkdir(tmp_path)
        self.path = "Data/Vehicles/" + run.path + run.run_name + "/" + folder + "/"
        self.states = []
        self.masks = []
        self.actions = []
        self.progresses = []
    

    def add_state(self, state):
        self.states.append(state)
        
    def add_mask(self, mask):
        self.masks.append(mask)
    
    def add_action(self, action):
        self.actions.append(action)

    def add_progress(self, progress):
        self.progresses.append(progress)
    
    def save_history(self, lap_n=0, test_map=None):
        states = np.array(self.states)
        self.actions.append(np.array([0, 0])) # last action to equal lengths
        actions = np.array(self.actions)
        progresses = np.array(self.progresses).reshape(-1, 1)

        lap_history = np.concatenate((states, actions, progresses), axis=1)
        if len(self.masks) > 2: # One is automatically added
            self.masks.insert(0, np.array(self.masks[0]))
            masks = np.concatenate(self.masks, axis=0)#.reshape(-1, 2)
            lap_history = np.concatenate((lap_history, masks), axis=1)

        if test_map is None:
            np.save(self.path + f"Lap_{lap_n}_history_{self.vehicle_name}.npy", lap_history)
        else:
            np.save(self.path + f"Lap_{lap_n}_history_{self.vehicle_name}_{test_map}.npy", lap_history)

        self.states = []
        self.actions = []
        self.progresses = []
        self.masks = []



class SafetyHistory:
    def __init__(self, run):
        self.vehicle_name = run.run_name
        self.path = "Data/Vehicles/" + run.path + self.vehicle_name + "/SafeHistory/"
        os.mkdir(self.path)

        self.planned_actions = []
        self.safe_actions = []
        self.interventions = []
        self.lap_n = 0

        self.interval_counter = 0
        self.inter_intervals = []
        self.ep_interventions = 0
        self.intervention_list = []

    def add_actions(self, planned_action, safe_action=None):
        self.planned_actions.append(planned_action)
        if safe_action is None:
            self.safe_actions.append(planned_action)
            self.interventions.append(False)
        else:
            self.safe_actions.append(safe_action)
            self.interventions.append(True)

    def add_planned_action(self, planned_action):
        self.planned_actions.append(planned_action)
        self.safe_actions.append(planned_action)
        self.interventions.append(False)
        self.interval_counter += 1

    def add_intervention(self, planned_action, safe_action):
        self.planned_actions.append(planned_action)
        self.safe_actions.append(safe_action)
        self.interventions.append(True)
        self.inter_intervals.append(self.interval_counter)
        self.interval_counter = 0
        self.ep_interventions += 1

    def train_lap_complete(self):
        self.intervention_list.append(self.ep_interventions)

        print(f"Interventions: {self.ep_interventions} --> {self.inter_intervals}")

        self.ep_interventions = 0
        self.inter_intervals = []

    def plot_safe_history(self):
        planned = np.array(self.planned_actions)
        safe = np.array(self.safe_actions)
        plt.figure(5)
        plt.clf()
        plt.title("Safe History: steering")
        plt.plot(planned[:, 0], color='blue')
        plt.plot(safe[:, 0], '-x', color='red')
        plt.legend(['Planned Actions', 'Safe Actions'])
        plt.ylim([-0.5, 0.5])
        # plt.show()
        plt.pause(0.0001)

        plt.figure(6)
        plt.clf()
        plt.title("Safe History: velocity")
        plt.plot(planned[:, 1], color='blue')
        plt.plot(safe[:, 1], '-x', color='red')
        plt.legend(['Planned Actions', 'Safe Actions'])
        plt.ylim([-0.5, 0.5])
        # plt.show()
        plt.pause(0.0001)

        self.planned_actions = []
        self.safe_actions = []

    def save_safe_history(self, training=False):
        planned_actions = np.array(self.planned_actions)
        safe_actions = np.array(self.safe_actions)
        interventions = np.array(self.interventions)
        data = np.concatenate((planned_actions, safe_actions, interventions[:, None]), axis=1)

        if training:
            np.save(self.path + f"Training_safeHistory_{self.vehicle_name}.npy", data)
        else:
            np.save(self.path + f"Lap_{self.lap_n}_safeHistory_{self.vehicle_name}.npy", data)

        self.lap_n += 1

        self.planned_actions = []
        self.safe_actions = []
        self.interventions = []
