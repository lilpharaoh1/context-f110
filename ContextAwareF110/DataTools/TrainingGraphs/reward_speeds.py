import os
import sys
sys.path.insert(0, os.getcwd()) # hacky fix

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter
import argparse

from ContextAwareF110.Utils.utils import *
from ContextAwareF110.DataTools.TrainingGraphs.TrainingUtils import *
from ContextAwareF110.DataTools.plotting_utils import *

def plot_reward_speeds(run_file):
    run_data = setup_run_list(run_file)
    data_path = "Data/Vehicles/" + run_file + "/"

    print(run_data)
    print("Run List :")
    for run in run_data:
        print(f"    {run}")

    n_repeats = run_data[-1].n + 1
    n_runs = len(run_data) // n_repeats
    n_train_steps = run_data[0].n_train_steps
    steps_list = [[] for _ in range(n_runs)]
    progresses_list = [[] for _ in range(n_runs)]
    
    for run_num, run in enumerate(run_data):
        path = data_path + run.run_name + "/"
        rewards, lengths, progresses, _ = load_csv_data(path)
        steps = np.cumsum(lengths[:-1]) / 1000
        avg_progress = true_moving_average(progresses[:-1], 20)* 100

        steps_list[run_num % n_repeats].append(steps)
        progresses_list[run_num % n_repeats].append(avg_progress)

    plt.figure(2, figsize=(4.5, 2.3))

    labels = [f"{run_data[i].max_speed} m/s" for i in range(n_runs)]

    xs = np.linspace(0, n_train_steps / 1000, 300)
    for i in range(len(steps_list)):
        min, max, mean = convert_to_min_max_avg(steps_list[i], progresses_list[i], xs)
        plt.plot(xs, mean, '-', color=pp[i], linewidth=2, label=labels[i])
        plt.gca().fill_between(xs, min, max, color=pp[i], alpha=0.2)

    plt.gca().get_yaxis().set_major_locator(MultipleLocator(25))

    plt.xlabel("Training Steps (x1000)")
    plt.ylabel("Track Progress %")
    plt.ylim(0, n_train_steps / 1000)
    plt.legend(loc='center', bbox_to_anchor=(0.5, -0.52), ncol=3)
    plt.tight_layout()
    plt.grid()

    name = data_path + "training_progress"
    std_img_saving(name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run-file', default='dev', type=str)
    args = parser.parse_args()

    plot_reward_speeds(args.run_file)
    
if __name__ == "__main__":
    main()
