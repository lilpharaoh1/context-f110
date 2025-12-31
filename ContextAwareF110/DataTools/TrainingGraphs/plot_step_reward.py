import os
import sys
sys.path.insert(0, os.getcwd()) # hacky fix

import matplotlib.pyplot as plt
import numpy as np
import glob
import argparse
import yaml

from ContextAwareF110.Utils.utils import *
from matplotlib.ticker import MultipleLocator
from ContextAwareF110.DataTools.plotting_utils import *

from ContextAwareF110.Utils.utils import *
from ContextAwareF110.DataTools.TrainingGraphs.TrainingUtils import *
from ContextAwareF110.DataTools.plotting_utils import *

SAVE_NAME = "classic_stepreward_dist_1"
FILL = True

def plot_progress():
    names = [
        "SAC",
        "DreamerV3",
        "cRSSM",
        "cMask",
    ]
    runs = [
        "sac_multiagent_classic/SAC_1_15153030_Std_Cth_f1_esp_6_0_850_",
        "dreamerv3_multiagent_classic/DreamerV3_1_15153030_Std_Cth_f1_esp_6_0_850_",
        "cdreamer_multiagent_classic/cDreamer_1_15153030_Std_Cth_f1_esp_6_0_850_",
        "cbdreamer_multiagent_classic2/cbDreamer_1_15153030_Std_Cth_f1_esp_6_0_850_",

    ]
    colors = [
        'gray',
        'red',
        'blue',
        'green',
    ]
    
    def find_progress(folder):
        rewards, lengths, progresses, _ = load_csv_data(folder)
        print(lengths[-20:])
        steps = np.cumsum(lengths[:-1]) / 1000
        # avg_progress = true_moving_average(rewards[:-1], 20)
        avg_progress = ewma(rewards[:-1] / lengths[:-1])

        return avg_progress, steps    

    def explore_folder(run_name, run_idx=0):
        run_folders = glob.glob(f"Data/Vehicles/{run_name}*/")
        print("run_folder :", run_folders)

        progresses = []
        steps_list = []    
        for idx, folder in enumerate(run_folders):
            print(f"Vehicle folder being opened: {folder}")
            prog, steps = find_progress(folder)
            progresses.append(prog)
            steps_list.append(steps)
    
        xs = np.linspace(0, 100, 300)
        min, max, mean = convert_to_min_max_avg(steps_list, progresses, xs)
        print("min, max, mean:", min.shape, max.shape, mean.shape)
        
        plt.cla()
        plt.clf()
        
        plt.plot(xs, mean, '-', color=colors[run_idx], linewidth=2, label=names[run_idx])
        if FILL:
            plt.gca().fill_between(xs, min, max, color=colors[run_idx], alpha=0.2)

        plt.xlabel("Training Steps (x1000)")
        plt.ylabel("Reward per Episode")
        plt.ylim(0, 1)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.grid()

        std_img_saving(f"Data/Vehicles/{run_name.split('/')[0]}/{SAVE_NAME}")

        return min, max, mean

    mins, maxs, means = [], [], []
    for idx, run in enumerate(runs):
        min, max, mean = explore_folder(run, run_idx=idx)
        mins.append(min)
        maxs.append(max)
        means.append(mean)
    
    plt.cla()
    plt.clf()
    xs = np.linspace(0, 100, 300)
    for run_idx in range(len(runs)):
        plt.plot(xs, means[run_idx], '-', color=colors[run_idx], linewidth=2, label=names[run_idx])
        if FILL:
            plt.gca().fill_between(xs, mins[run_idx], maxs[run_idx], color=colors[run_idx], alpha=0.2)
    plt.xlabel("Training Steps (x1000)")
    plt.ylabel("Reward per Episode")
    plt.ylim(0, 1)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.grid()
    std_img_saving(f"Data/Vehicles/{SAVE_NAME}")




plot_progress()