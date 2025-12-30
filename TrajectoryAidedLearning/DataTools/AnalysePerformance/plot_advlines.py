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

from TrajectoryAidedLearning.DataTools.MapData import MapData
from TrajectoryAidedLearning.Utils.StdTrack import StdTrack 
from TrajectoryAidedLearning.Utils.RacingTrack import RacingTrack
from TrajectoryAidedLearning.Utils.utils import *
from matplotlib.ticker import MultipleLocator
from TrajectoryAidedLearning.DataTools.plotting_utils import *

# SAVE_PDF = False
SAVE_PDF = True

RUN_FOLDER = 3

CMAP_SIZE = {
    "f1_esp" : 0.35,
    "f1_aut" : 0.4,
    "f1_gbr" : 0.4,
    "f1_mco" : 0.4
}

cmaps = ['Blues', 'Greens', 'Oranges']
colors = ['coral', 'mediumseagreen', 'royalblue']

def ensure_path_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


class AnalyseTestLapData:
    def __init__(self):
        self.path = None
        self.vehicle_name = None
        self.map_name = None
        self.states = None
        self.actions = None
        self.map_data = None
        self.std_track = None
        self.summary_path = None
        self.run_data = None
        self.num_agents = None
        self.n_test_laps = None
        self.lap_n = 0

    def explore_folder(self, run_file):
        self.run_data = setup_run_list(run_file)
        path = "Data/Vehicles/" + run_file + "/"

        vehicle_folders = glob.glob(f"{path}*/")
        print(vehicle_folders)
        vehicle_folder = vehicle_folders[RUN_FOLDER]
        self.num_agents = self.run_data[RUN_FOLDER].num_agents
        self.n_test_laps = self.run_data[RUN_FOLDER].n_test_laps
        print(f"Selecting vehicle folder {vehicle_folder} for analysis...")

        self.process_folder(vehicle_folder)

        # set = 1
        # for j, folder in enumerate(vehicle_folders):
            # print(f"Vehicle folder being opened: {folder}")
            # self.process_folder(folder)

    def process_folder(self, folder):
        self.vehicle_name = folder.split("/")[-2]
                
        self.map_name = self.vehicle_name.split("_")[5]
        if self.map_name == "f1":
            self.map_name += "_" + self.vehicle_name.split("_")[6]
        self.map_data = MapData(self.map_name)
        self.std_track = StdTrack(self.map_name)
        self.racing_track = RacingTrack(self.map_name)

        plt.figure(1)
        plt.clf()
        self.map_data.plot_map_img()

        for idx, run_num in enumerate([21, 24, 27]):
        # for idx, run_num in enumerate([40, 43, 48]):
            run_folder = glob.glob(f"{folder}" + f"Testing/Testing_{run_num}/")
            print("run_folder :", run_folder)
            self.load_advline(run_folder, run_num)
            self.plot_advline(idx)

        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        name = folder + "advline_whole"
        whole_limits(self.map_name)
        std_img_saving(name)

        name = folder + "advline_highlight"
        highlight1_limits(self.map_name)
        std_img_saving(name)

        # name = save_path + f"{self.vehicle_name}_velocity_map_{self.lap_n}_highlight1"
        # highlight1_limits(self.map_name)
        # std_img_saving(name)

        # name = save_path + f"{self.vehicle_name}_velocity_map_{self.lap_n}_highlight2"
        # highlight2_limits(self.map_name)
        # std_img_saving(name)

        # for j, run_folder in enumerate(runs_folders):
        #     print("processing ", run_folder)
        #     print(f"{j}) {run_folder}")
        #     if not os.path.exists(run_folder + "TestingVelocities/"): 
        #         os.mkdir(run_folder + "TestingVelocities/") 
        #     self.path = run_folder
        #     for self.lap_n in range(self.n_test_laps):
        #         if not self.load_lap_data(): break # no moreSAC_0_0000_Std_Cth_f1_esp_6_10_850_0 laps
        #         print(f"Processing test lap {self.lap_n}...")
        #         self.plot_velocity_heat_map()


    def load_advline(self, run_folder, run_num):
        self.path = run_folder[0]
        best_lap = 0
        longest = 0
        for lap_n in range(1):
            data = np.load(self.path + f"agent_1/Lap_{lap_n}_history_{self.vehicle_name}_{self.map_name}.npy")
            if data.shape[0] > longest:
                best_lap = lap_n
                longest = data.shape[0]
        data = np.load(self.path + f"agent_1/Lap_{best_lap}_history_{self.vehicle_name}_{self.map_name}.npy")
        print(data.shape)
        self.states = data[:, :7]
        self.actions = data[:, 7:]

        return 1 # to say success

    
    def plot_advline(self, i): 
        # save_path  = self.path + "TestingVelocities/"


        for agent_id in range(self.num_agents):
            points = self.states[:, 0:2]
            vs = self.states[:, 3]
            
            xs, ys = self.map_data.pts2rc(points)
            points = np.concatenate([xs[:, None], ys[:, None]], axis=1)
            points = points.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            norm = plt.Normalize(0, 5)
            # lc = LineCollection(segments, cmap=cmaps[i], norm=norm)
            lc = LineCollection(segments, alpha=0.7, color=colors[i], norm=norm)
            lc.set_array(vs)
            lc.set_linewidth(2.4)
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
        plt.xlim(1150, 1400)
        plt.ylim(200, 425)
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

def analyse_folder(run_file):    
    TestData = AnalyseTestLapData()
    TestData.explore_folder(run_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run-file', default='dev', type=str)
    args = parser.parse_args()

    analyse_folder(args.run_file)
    
if __name__ == "__main__":
    main()


    
