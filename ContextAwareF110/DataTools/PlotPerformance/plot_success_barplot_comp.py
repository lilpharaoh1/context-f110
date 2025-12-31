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

def plot_progress():
    names = [
        "SAC",
        "DreamerV3",
        "SAC",
        "DreamerV3",
    ]
    runs = [
        "SAC_singleagent/SAC_0_0000_Std_Cth_f1_esp_6_10_850_",
        "dreamerv3_singleagent/DreamerV3_0_0000_Std_Cth_f1_esp_6_1_850_",
        "dreamerv3_singleagent/DreamerV3_0_0000_Std_Cth_f1_esp_6_1_850_",
        "SAC_singleagent/SAC_0_0000_Std_Cth_f1_esp_6_10_850_",
    ]
    colors = [
        'red',
        'blue',
        'red',
        'blue',
    ]

    split = 2
    
    def find_progress(folder):
        yaml_path = f"{folder}RunConfig_0_record.yaml"
        print(yaml_path)
        with open(yaml_path) as file:
            run_config = yaml.safe_load(file)
            indv_succ = run_config['success_rate']
            print("indc_succ :", indv_succ)

        print("indc_succ :", indv_succ)
        return indv_succ   

    def explore_folder(run_name):
        run_folders = glob.glob(f"Data/Vehicles/{run_name}*/")
        print("run_folder :", run_folders)

        progresses = []
        for idx, folder in enumerate(run_folders):
            print(f"Vehicle folder being opened: {folder}")
            prog  = find_progress(folder)
            progresses.append(prog)

        print("prog :", progresses)
    
        min, max, mean = np.min(progresses), np.max(progresses), np.mean(progresses)
        print("min, max, mean:", min, max, mean)

        return min, max, mean

    def plot_minmax(x_base, mins, maxes, color, w):
        for i in range(len(x_base)):
            xs = [x_base[i], x_base[i]]
            ys = [mins, maxes]
            plt.plot(xs, ys, color=color, alpha=0.8, linewidth=2)
            xs = [x_base[i]-w, x_base[i]+w]
            y1 = [mins, mins]
            y2 = [maxes, maxes]
            plt.plot(xs, y1, color=color, alpha=0.8, linewidth=2)
            plt.plot(xs, y2, color=color, alpha=0.8, linewidth=2)

    mins, maxs, means = [], [], []
    for idx, run in enumerate(runs):
        min, max, mean = explore_folder(run)
        mins.append(min)
        maxs.append(max)
        means.append(mean)
    mins = np.array(mins).reshape((-1, split))
    maxs = np.array(maxs).reshape((-1, split))
    means = np.array(means).reshape((-1, split))


    fig = plt.figure(figsize=(4.5, 2.6))
    xs = np.arange(1, len(runs) // split) # How many comparisons?
    
    barWidth = 0.4
    w = 0.05
    br = xs - barWidth/2
        
    plt.cla()
    plt.clf()

    for i in range(len(runs) // split):
        br = [x + barWidth for x in br]
        plt.bar(br, means[start], color=colors[start], width=barWidth, label=names[start])
        plot_minmax(br, mins[start], maxs[start], 'black', w)

        for run_idx in range(1, len(runs) // split):
            br = [x + barWidth for x in br]

            plt.bar(br, means[start + run_idx], color=colors[start + run_idx], width=barWidth, label=names[start + run_idx])
            plot_minmax(br, mins[start + run_idx], maxs[start + run_idx], 'black', w)
        
    plt.gca().get_xaxis().set_major_locator(MultipleLocator(1))
    plt.gca().get_yaxis().set_major_locator(MultipleLocator(25))
    plt.xticks([0, 1], ["label one", "label two"])
    # plt.xlabel("Maximum speed (m/s)")
    plt.ylabel("Average Track \nProgress (%)")
    plt.ylim(0, 100)

    plt.legend(ncol=2, loc="center", bbox_to_anchor=(0.50, -0.15))
    plt.tight_layout()
    plt.grid()
    std_img_saving(f"Data/Vehicles/success_barplot_comp")

plot_progress()
