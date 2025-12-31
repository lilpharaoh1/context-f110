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
        self.map_name = self.run_data[-1].map_name

        path = "Data/Vehicles/" + run_file + "/"

        vehicle_folders = sorted(glob.glob(f"{path}*/"))
        run_names = [folder[:-2] for folder in vehicle_folders[::self.n]]
        run_folders = [sorted(glob.glob(f"{run_name}*/")) for run_name in run_names]
        print(f"{len(vehicle_folders)} folders found")

        for run_num, run_folder in enumerate(run_folders):
            print("run_num", run_num)
            self.num_agents = self.run_data[len(run_folders) - 1 - run_num].num_agents
            self.n_test_laps = self.run_data[len(run_folders) - 1 - run_num].n_test_laps

            for idx, folder in enumerate(run_folder):
                print(f"Vehicle folder being opened: {folder}")
                for run in range(0, 49):
                    print(f" - Processing Run {run} - ")
                    data = self.load_testing_stats(folder, run)
                    self.edit_run(folder, run, data)
                    # exit()

            
    def load_testing_stats(self, folder, run):
        run_path = f"{folder}Testing/Testing_{run}/"
        vehicle_name = folder.split("/")[-2]
        place = []
        for lap_n in range(50):
            max_progresses = np.zeros((self.num_agents))
            for agent_idx in range(self.num_agents):
                data = np.load(run_path + f"agent_{agent_idx}/Lap_{lap_n}_history_{vehicle_name}_{self.map_name}.npy")
                for i in range(1, len(data)):
                    data[i, 9] = data[i-1, 9] + 0.005 if data[i, 9] == 0. else data[i, 9]
                for i in range(1, len(data)):
                    data[i, 9] = data[i-1, 9] + 0.005 if data[i, 9] <= data[i-1, 9] and data[i-1, 9] >= 0.97 else data[i, 9]
                max_progresses[agent_idx] = data[-1, 9]
            places = sorted(range(len(max_progresses)), key=lambda i: max_progresses[i], reverse=True)
            place.append(places[0] + 1 if max_progresses[0] > 1.0 else self.num_agents)
        
        place = np.array(place)

        return place

    def edit_run(self, folder, run, data):
        yaml_read = f"{folder}RunConfig_{run}_record.yaml"
        with open(yaml_read) as file:
            run_config = yaml.safe_load(file)
        
        run_config['avg_place'] = float(np.mean(data))
        run_config['place_std_dev'] = float(np.std(data))

        yaml_dump = f"{folder}TestRunConfig_{run}_record.yaml"
        with open(yaml_dump, 'w') as file:
            yaml.dump(run_config, file)

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