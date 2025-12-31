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
from matplotlib.patches import Rectangle, Arrow, Patch

from ContextAwareF110.DataTools.MapData import MapData
from ContextAwareF110.Utils.StdTrack import StdTrack 
from ContextAwareF110.Utils.RacingTrack import RacingTrack
from ContextAwareF110.Utils.utils import *
from matplotlib.ticker import MultipleLocator
from ContextAwareF110.DataTools.plotting_utils import *

# SAVE_PDF = False
SAVE_PDF = True

# LEGEND = False
LEGEND = True
LEGEND_LOC = "lower right"

ARCH_MAP = {
    0: None,
    1: "PP",
    2: "DispExt",
    3: "TD3",
    4: "SAC",
    5: "DreamerV2",
    6: "Director"
}

colors = ['red', 'blue', 'green', 'purple']


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
        self.num_agents = self.run_data[0].num_agents
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
        # self.path = folder

        self.vehicle_name = folder.split("/")[-2]
                
        self.map_name = self.vehicle_name.split("_")[5]
        if self.map_name == "f1":
            self.map_name += "_" + self.vehicle_name.split("_")[6]
        self.map_data = MapData(self.map_name)
        self.std_track = StdTrack(self.map_name)
        self.racing_track = RacingTrack(self.map_name)

                    
        runs_folders = glob.glob(f"{folder}" + "Testing/*/")
        for j, run_folder in enumerate(runs_folders):
            print(f"{j}) {run_folder}")
            self.num_agents = 2 if j < 3 else 4
            if not os.path.exists(run_folder + "TestingCollisions/"): 
                os.mkdir(run_folder + "TestingCollisions/") 
            self.path = run_folder
            self.states = np.empty((0, self.num_agents, 7))
            self.actions = np.empty((0, self.num_agents, 3))
            for self.lap_n in range(self.n_test_laps):
                if not self.load_lap_data(): break # no moreSAC_0_0000_Std_Cth_f1_esp_6_10_850_0 laps
                print(f"Processing test lap {self.lap_n}...")
            self.plot_velocity_heat_map(run_num=j)


    def load_lap_data(self):
        try:
            data = np.array([np.load(self.path + f"agent_{agent_id}/Lap_{self.lap_n}_history_{self.vehicle_name}_{self.map_name}.npy") for agent_id in range(self.num_agents)])
        except Exception as e:
            print(e)
            print(f"No data for: " + f"Lap_{self.lap_n}_history_{self.vehicle_name}_{self.map_name}.npy")
            return 0
        last_states = data[:, -1, :7].reshape(1, self.num_agents, 7)
        last_actions = data[:, -1, 7:].reshape(1, self.num_agents, 3)
        self.states = np.concatenate([self.states, last_states], axis=0)
        self.actions = np.concatenate([self.actions, last_actions], axis=0)

        return 1 # to say success

    
    def plot_velocity_heat_map(self, run_num=0): 
        save_path  = self.path + "TestingCollisions/"
        
        plt.figure(1)
        plt.clf()
        lap_length = self.states.shape[1]
        legend_patches = []

        agent_id = 0 # target agent
        points = self.states[:, agent_id, 0:2]
        angles = self.states[:, agent_id, 4]

        xs, ys = self.map_data.pts2rc(points)
        points = np.concatenate([xs[:, None], ys[:, None]], axis=1)
        points = points.reshape(-1, 2)
        
        self.map_data.plot_map_img()

        # Plot vehicle
        width, length = 10, 5
        for t, ((x, y), angle) in enumerate(zip(points, angles)):
            # plt.text(x + 5, y + 5, f"Lap {t}", fontsize=3, ha='left', color=colors[agent_id])
            car = Rectangle(
                (x - (width/2), y - (length/2)),
                width, length,
                angle=(angle * 180) / np.pi,
                rotation_point='center',
                color=colors[agent_id], 
                fill=False,
                linewidth=0.6,
                alpha=0.85,
                )
            arr = Arrow(
                x, y,
                (width)*np.cos(angle), (width)*np.sin(angle),
                width=length,
                color=colors[agent_id], 
                fill=False,
                linewidth=0.2,
                alpha=0.85
                )
            _ = plt.gca().add_patch(car)
            _ = plt.gca().add_patch(arr)

            plt.gca().set_aspect('equal', adjustable='box')
        
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        name = save_path + f"{self.vehicle_name}_collisions_{run_num} "
        set_limits(self.map_name)
        std_img_saving(name)

def set_limits(map_name):
    # # ESP Full
    plt.xlim(20, 1500)
    plt.ylim(50, 520)

    # # ESP Start
    # plt.xlim(650, 1200)
    # plt.ylim(300, 520)



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


    
