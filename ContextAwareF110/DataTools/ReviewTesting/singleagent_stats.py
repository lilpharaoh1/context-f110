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


class AnalyseTestLapData:
    def __init__(self):
        self.run_data = None
        self.conf = None

    def explore_folder(self, run_file):
        self.run_data = setup_run_list(run_file)
        self.conf = load_conf("config_file")
        self.n = self.run_data[-1].n + 1
        self.n_test_laps = self.run_data[-1].n_test_laps
        self.num_agents = self.run_data[-1].num_agents
        self.map_name = self.run_data[-1].map_name

        path = "Data/Vehicles/" + run_file + "/"
        run_folders = sorted(glob.glob(f"{path}*/"))

        # print(f"\nrun_folders: {run_folders}\n")

        total_laps = 0
        progresses = np.zeros(len(run_folders))
        velocities = np.zeros(len(run_folders))
        times = np.zeros(len(run_folders))
        for run_num, run_folder in enumerate(run_folders):
            laps, progress, velocity, time = self.load_stats(run_folder)
            total_laps += int(laps)
            progresses[run_num] = progress
            velocities[run_num] = velocity
            times[run_num] = time

        print(f"Total Laps: {total_laps}")
        print(f"Progress: {np.mean(progresses)} +- {np.std(progresses)}") 
        print(f"Velocity: {np.mean(velocities)} +- {np.std(velocities)}") 
        print(f"Lap Times: {np.mean(times)} +- {np.std(times)}") 


    def load_stats(self, folder):
        run_path = f"{folder}Testing/Testing_0/"
        vehicle_name = folder.split("/")[-2]
        speeds = []
        for lap_n in range(self.n_test_laps):
            data = np.load(run_path + f"agent_0/Lap_{lap_n}_history_{vehicle_name}_{self.map_name}.npy")
            speeds.append(np.mean(data[:, 3]))
        
        yaml_read = f"{folder}RunConfig_0_record.yaml"
        with open(yaml_read) as file:
            run_config = yaml.safe_load(file)
        
        laps = run_config['success_rate'] * (self.n_test_laps / 100)
        progress = run_config['avg_progress']
        velocity = np.mean(speeds)
        time = run_config['avg_times']

        return laps, progress, velocity, time

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