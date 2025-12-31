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
    2: "FTG",
    3: "TD3",
    4: "SAC",
    5: "DreamerV2",
    6: "Director"
}

colors = ['red', 'blue', 'green', 'purple']
DT = 0.01 # defined by self.timestep in f110_env


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
        self.sim_steps = load_conf("config_file").sim_steps

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
            if not os.path.exists(run_folder + "TestingProgress/"): 
                os.mkdir(run_folder + "TestingProgress/") 
            self.path = run_folder
            for self.lap_n in range(self.n_test_laps):
                if not self.load_lap_data(): break # no moreSAC_0_0000_Std_Cth_f1_esp_6_10_850_0 laps
                print(f"Processing test lap {self.lap_n}...")
                self.plot_progress()


    def load_lap_data(self):
        try:
            data = np.array([np.load(self.path + f"agent_{agent_id}/Lap_{self.lap_n}_history_{self.vehicle_name}_{self.map_name}.npy") for agent_id in range(self.num_agents)])
        except Exception as e:
            print(e)
            print(f"No data for: " + f"Lap_{self.lap_n}_history_{self.vehicle_name}_{self.map_name}.npy")
            return 0
        self.states = data[:, :, :7]
        self.actions = data[:, :, 7:9]
        self.progresses = data[:, :, 9]

        return 1 # to say success

    
    def plot_progress(self): 
        save_path  = self.path + "TestingProgress/"
        
        plt.figure(1, figsize=(4.5, 2.3))
        plt.clf()
        target = self.path[:-1].split('/')[-1].split('_')[0]
        advs = [ARCH_MAP[int(str_adv)] for str_adv in self.path[:-1].split("/")[-1].split('_')[1]]
        agent_names = [f"{adv} (Adversary #{adv_idx + 1})" for adv_idx, adv in enumerate(advs)]
        agent_names.insert(0, f"{target} (Target)") 

        total_steps = self.progresses.shape[1]
        xs = np.linspace(0, total_steps * DT * self.sim_steps, total_steps)
        legend_patches = [] 

        for agent_id in range(self.num_agents):
            progress = self.progresses[agent_id]
            plt.plot(xs, progress, '-', color=colors[agent_id], linewidth=1, alpha=0.85)
        

            if LEGEND:
                legend_patches.append(Patch(color=colors[agent_id], label=agent_names[agent_id], fill=False, linewidth=3))

        plt.xlabel("Time (s)")
        plt.ylabel("Track Progress %")
        plt.xlim(0.0, total_steps * DT * self.sim_steps)
        if LEGEND:
            plt.legend(handles=legend_patches, loc=LEGEND_LOC, fontsize=10)
        # plt.legend(loc='center', bbox_to_anchor=(0.5, -0.52), ncol=3)
        plt.tight_layout()
        plt.grid()

        name = save_path + f"{self.vehicle_name}_progress_{self.lap_n}_relay"
        std_img_saving(name)

        plt.ylim(0.0, 1.0)
        name = save_path + f"{self.vehicle_name}_progress_{self.lap_n}_fully"
        std_img_saving(name)

def set_limits(map_name):
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


    
