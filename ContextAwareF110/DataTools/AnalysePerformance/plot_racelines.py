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

CTX_RUN = 17

CMAP_SIZE = {
    "f1_esp" : 0.35,
    "f1_aut" : 0.4,
    "f1_gbr" : 0.4,
    "f1_mco" : 0.4
}

cmaps = ["Reds", "Blues", "Greens", "Purples"]


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
        self.num_agents = self.run_data[1].num_agents
        self.n_test_laps = self.run_data[0].n_test_laps
        path = "Data/Vehicles/" + run_file + "/"

        vehicle_folders = glob.glob(f"{path}*/")
        print(vehicle_folders)
        print(f"{len(vehicle_folders)} folders found")

        set = 1
        for j, folder in enumerate(vehicle_folders):
            print(f"Vehicle folder being opened: {folder}")
            self.process_folder(folder)

    def process_folder(self, folder):
        self.vehicle_name = folder.split("/")[-2]
                
        self.map_name = self.vehicle_name.split("_")[5]
        if self.map_name == "f1":
            self.map_name += "_" + self.vehicle_name.split("_")[6]
        self.map_data = MapData(self.map_name)
        self.std_track = StdTrack(self.map_name)
        self.racing_track = RacingTrack(self.map_name)


        runs_folders = glob.glob(f"{folder}" + f"Testing/Testing_{CTX_RUN}/")
        print("run_folders :", runs_folders)
        for j, run_folder in enumerate(runs_folders):
            print("processing ", run_folder)
            print(f"{j}) {run_folder}")
            if not os.path.exists(run_folder + "TestingVelocities/"): 
                os.mkdir(run_folder + "TestingVelocities/") 
            self.path = run_folder
            for self.lap_n in range(self.n_test_laps):
                if not self.load_lap_data(): break # no moreSAC_0_0000_Std_Cth_f1_esp_6_10_850_0 laps
                print(f"Processing test lap {self.lap_n}...")
                self.plot_velocity_heat_map()


    def load_lap_data(self):
        try:
            data = np.array([np.load(self.path + f"agent_{agent_id}/Lap_{self.lap_n}_history_{self.vehicle_name}_{self.map_name}.npy") for agent_id in range(self.num_agents)])
        except Exception as e:
            print(e)
            print(f"No data for: " + f"Lap_{self.lap_n}_history_{self.vehicle_name}_{self.map_name}.npy")
            return 0
        self.states = data[:, :, :7]
        self.actions = data[:, :, 7:]

        return 1 # to say success

    
    def plot_velocity_heat_map(self): 
        save_path  = self.path + "TestingVelocities/"
        
        plt.figure(1)
        plt.clf()

        for agent_id in range(self.num_agents):
            points = self.states[agent_id, :, 0:2]
            vs = self.states[agent_id, :, 3]
            
            self.map_data.plot_map_img()

            xs, ys = self.map_data.pts2rc(points)
            points = np.concatenate([xs[:, None], ys[:, None]], axis=1)
            points = points.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            norm = plt.Normalize(0, 6)
            lc = LineCollection(segments, cmap=cmaps[agent_id], norm=norm)
            lc.set_array(vs)
            lc.set_linewidth(0.5)
            line = plt.gca().add_collection(lc)
            cbar = plt.colorbar(line,fraction=0.046, pad=0.04, shrink=CMAP_SIZE[self.map_name])
            cbar.ax.tick_params(labelsize=12)
            plt.gca().set_aspect('equal', adjustable='box')

        
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        name = save_path + f"{self.vehicle_name}_velocity_map_{self.lap_n}_whole"
        whole_limits(self.map_name)
        std_img_saving(name)

        # name = save_path + f"{self.vehicle_name}_velocity_map_{self.lap_n}_highlight1"
        # highlight1_limits(self.map_name)
        # std_img_saving(name)

        # name = save_path + f"{self.vehicle_name}_velocity_map_{self.lap_n}_highlight2"
        # highlight2_limits(self.map_name)
        # std_img_saving(name)

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
        plt.xlim(20, 1500)
        plt.ylim(50, 520)

def highlight1_limits(map_name):
    if map_name == "f1_esp":
        # ESP Start
        plt.xlim(650, 1200)
        plt.ylim(300, 520)
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


    
