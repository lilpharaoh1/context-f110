import os
import sys
sys.path.insert(0, os.getcwd()) # hacky fix

from matplotlib import pyplot as plt
# plt.rc('font', family='serif')
# plt.rc('pdf',fonttype = 42)
# plt.rc('text', usetex=True)
plt.rcParams['pdf.use14corefonts'] = True

import numpy as np
import glob
import os
import argparse

import glob
from matplotlib.ticker import PercentFormatter
from matplotlib.collections import LineCollection

from ContextAwareF110.DataTools.MapData import MapData
from ContextAwareF110.Utils.StdTrack import StdTrack 
from ContextAwareF110.Utils.RacingTrack import RacingTrack
from ContextAwareF110.Utils.utils import *
from matplotlib.ticker import MultipleLocator
from ContextAwareF110.DataTools.plotting_utils import *

# SAVE_PDF = False
SAVE_PDF = True

RUN_FOLDER = 9

CMAP_SIZE = {
    "f1_esp" : 0.35,
    "f1_aut" : 0.4,
    "f1_gbr" : 0.4,
    "f1_mco" : 0.4
}

names = [
    "SAC",
    "DreamerV3"
    ]

runs = [
    "sac_singleagent/SAC_0_0000_Std_Cth_f1_esp_6_10_850_0",
    "dreamerv3_singleagent/DreamerV3_0_0000_Std_Cth_f1_esp_6_0_850_0"
]
cmaps = [
    'Blues',
    'Oranges',
]

def ensure_path_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def process_folder():
    map_data = MapData('f1_esp')
    map_data.plot_map_img()

    for idx, run in enumerate(runs):
        folder = f"Data/Vehicles/{run}/"
        run_folder = glob.glob(f"{folder}" + f"Testing/Testing_0/")
        states = load_raceline(run_folder)
        plot_raceline(states, map_data, idx)

    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.legend(names)

    plt.show()

    # name = folder + "advline_whole"
    # whole_limits(self.map_name)
    # std_img_saving(name)

    # name = folder + "advline_highlight"
    # highlight1_limits(self.map_name)
    # std_img_saving(name)


def load_raceline(run_folder):
    path = run_folder[0]
    vehicle_name = path.split("/")[-4]
    best_lap = 0
    shortest = 10000
    for lap_n in range(50):
        data = np.load(path + f"agent_0/Lap_{lap_n}_history_{vehicle_name}_f1_esp.npy")
        if data.shape[0] < shortest and data[-1, 9] >= 1.0:
            best_lap = lap_n
            shortest = data.shape[0]
    data = np.load(path + f"agent_0/Lap_{best_lap}_history_{vehicle_name}_f1_esp.npy")
    print(data.shape)
    states = data[:, :7]

    return states # to say success


def plot_raceline(states, map_data, i): 
    points = states[:, 0:2]
    vs = states[:, 3]
    
    xs, ys = map_data.pts2rc(points)
    points = np.concatenate([xs[:, None], ys[:, None]], axis=1)
    points = points.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = plt.Normalize(0, 5)
    lc = LineCollection(segments, cmap=cmaps[i], norm=norm)
    lc.set_array(vs)
    lc.set_linewidth(0.5)
    line = plt.gca().add_collection(lc)
    plt.gca().set_aspect('equal', adjustable='box')


def whole_limits(map_name):
    if map_name == "f1_esp":
        plt.xlim(20, 1500)
        plt.ylim(50, 520)
    elif map_name == "f1_aut":
        plt.xlim(20, 1500)
        plt.ylim(50, 520)
    elif map_name == "f1_aut":
        plt.xlim(20, 1500)
        plt.ylim(50, 520)
    else: # "f1_mco":
        plt.xlim(0, 1350)
        plt.ylim(0, 900)

def highlight1_limits(map_name):
    if map_name == "f1_esp":
        # ESP Start
        plt.xlim(625, 1485)
        plt.ylim(50, 520)
    elif map_name == "f1_aut":
        plt.xlim(20, 1500)
        plt.ylim(50, 520)
    elif map_name == "f1_aut":
        plt.xlim(20, 1500)
        plt.ylim(50, 520)
    else: # "f1_mco":
        plt.xlim(20, 1500)
        plt.ylim(50, 520)

def highlight2_limits(map_name):
    if map_name == "f1_esp":
        plt.xlim(900, 1500)
        plt.ylim(50, 520)
    elif map_name == "f1_aut":
        plt.xlim(20, 1500)
        plt.ylim(50, 520)
    elif map_name == "f1_aut":
        plt.xlim(20, 1500)
        plt.ylim(50, 520)
    else: # "f1_mco":
        plt.xlim(20, 1500)
        plt.ylim(50, 520)

def main():
    process_folder()
    
if __name__ == "__main__":
    main()


    
