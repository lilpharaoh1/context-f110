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

RUN_FOLDER = 4

CMAP_SIZE = {
    "f1_esp" : 0.35,
    "f1_aut" : 0.4,
    "f1_gbr" : 0.4,
    "f1_mco" : 0.4
}

cmaps = ["Reds", "Blues", "Greens", "Purples"]
colors = ['mediumseagreen', 'lightseagreen', 'coral']


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
        self.progresses = None 
        self.masks = None
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
        vehicle_folder = vehicle_folders[RUN_FOLDER]
        self.num_agents = 2
        self.n_test_laps = self.run_data[RUN_FOLDER].n_test_laps
        print(f"Selecting vehicle folder {vehicle_folder} for analysis...")

        self.process_folder(vehicle_folder)

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

        # for idx, run_num in enumerate([2, 24, 46]):
        # for idx, run_num in enumerate([40, 43, 48]):
        run_num = 24
        run_folder = glob.glob(f"{folder}" + f"Testing/Testing_{run_num}/")
        print("run_folder :", run_folder)
        self.load_masks(run_folder, run_num)
        self.plot_masks()

        name = folder + "masks_map"
        std_img_saving(name)


    def load_masks(self, run_folder, run_num):
        self.path = run_folder[0]
        best_lap = 0
        longest = 0
        for lap_n in range(50):
            data = np.load(self.path + f"agent_0/Lap_{lap_n}_history_{self.vehicle_name}_{self.map_name}.npy")
            opp_data = np.load(self.path + f"agent_1/Lap_{lap_n}_history_{self.vehicle_name}_{self.map_name}.npy")
            if data.shape[0] > longest and data[-30, 9] > opp_data[-30, 9]:
                best_lap = lap_n
                longest = data.shape[0]
        best_lap = 44
        data = np.expand_dims(np.load(self.path + f"agent_0/Lap_{best_lap}_history_{self.vehicle_name}_{self.map_name}.npy"), axis=0)
        opp_data = np.expand_dims(np.load(self.path + f"agent_1/Lap_{best_lap}_history_{self.vehicle_name}_{self.map_name}.npy"), axis=0)
        self.states = np.concatenate((data[:, :, :7], opp_data[:, :, :7]), axis=0)
        self.actions = np.concatenate((data[:, :, 7:9], opp_data[:, :, 7:9]), axis=0)
        self.progresses = np.concatenate((data[:, :, 9], opp_data[:, :, 9]), axis=0)
        self.prog_differences = opp_data[:, :, 9] - data[0, :, 9]
        self.overtake_idx = [np.argwhere(pd > 0)[-1][0] for pd in self.prog_differences]
        self.masks = data[0, :, 10:]

        return 1 # to say success

    
    def plot_masks(self): 
        fig, ax = plt.subplots()

        points = self.states[0, :, 0:2]
        steering_masks = self.masks[:, 0]
        speed_masks = self.masks[:, 1]
        
        self.map_data.plot_map_img()

        xs, ys = self.map_data.pts2rc(points)
        points = np.concatenate([xs[:, None], ys[:, None]], axis=1)
        points = points.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(0.0, 1.0)
        lc = LineCollection(segments, cmap=cmaps[0], norm=norm)
        # lc.set_array(steering_masks)
        lc.set_array(speed_masks)
        lc.set_linewidth(0.5)
        line = plt.gca().add_collection(lc)
        cbar = plt.colorbar(line,fraction=0.046, pad=0.04, shrink=CMAP_SIZE[self.map_name])
        cbar.ax.tick_params(labelsize=12)
        plt.gca().set_aspect('equal', adjustable='box')

        for agent_id in range(1, self.num_agents):
            points = self.states[agent_id, :, 0:2]
            print(points.shape)
            
            self.map_data.plot_map_img()

            xs, ys = self.map_data.pts2rc(points)
            points = np.concatenate([xs[:, None], ys[:, None]], axis=1)
            points = points.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            norm = plt.Normalize(0.0, 1.0)
            lc = LineCollection(segments, alpha=0.5)
            lc.set_linewidth(0.5)
            line = plt.gca().add_collection(lc)
            plt.gca().set_aspect('equal', adjustable='box')
        
        width, length = 10, 5
        for agent_id, o_idx in enumerate(self.overtake_idx):
            tar_xy = self.states[0, o_idx, 0:2].reshape(1, -1)
            opp_xy = self.states[agent_id + 1, o_idx, 0:2].reshape(1, -1)
            tar_x, tar_y = self.map_data.pts2rc(tar_xy)
            opp_x, opp_y = self.map_data.pts2rc(opp_xy)
            tar_angle = self.states[0, o_idx, 4]
            opp_angle = self.states[agent_id + 1, o_idx, 4]
            tar_car = Rectangle(
                (tar_x - (width/2), tar_y - (length/2)),
                width, length,
                angle=(tar_angle * 180) / np.pi,
                rotation_point='center',
                color='orangered', 
                fill=False,
                linewidth=0.6,
                alpha=0.85,
                )
            opp_car = Rectangle(
                (opp_x - (width/2), opp_y - (length/2)),
                width, length,
                angle=(opp_angle * 180) / np.pi,
                rotation_point='center',
                color='RoyalBlue', 
                fill=False,
                linewidth=0.6,
                alpha=0.85,
                )
            ax.add_patch(tar_car)
            ax.add_patch(opp_car)

        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)


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


    
