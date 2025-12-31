import yaml 
import csv 
import os 
from argparse import Namespace
import shutil
import numpy as np
from numba import njit
from matplotlib import pyplot as plt

ARCH_MAP = {
    "PP": 1,
    "DispExt": 2,
    "TD3": 3,
    "SAC": 4,
    "DreamerV2": 5,
    "Director": 6
}

def save_conf_dict(dictionary, save_name=None):
    if save_name is None:
        save_name  = dictionary["run_name"]
    path = "Data/Vehicles/" + dictionary["path"] + dictionary["run_name"] + f"/{save_name}_record.yaml"
    with open(path, 'w') as file:
        yaml.dump(dictionary, file)

def load_conf(fname):
    full_path =  "config/" + fname + '.yaml'
    with open(full_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)

    conf = Namespace(**conf_dict)

    return conf

def load_yaml_dict(fname):
    full_path =  "config/" + fname + '.yaml'
    with open(full_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)

    return conf_dict



def init_file_struct(path):
    if os.path.exists(path):
        try:
            os.rmdir(path)
        except:
            shutil.rmtree(path)
    os.mkdir(path)

@njit(cache=True)
def limit_phi(phi):
    while phi > np.pi:
        phi = phi - 2*np.pi
    while phi < -np.pi:
        phi = phi + 2*np.pi
    return phi

def init_file_struct(path):
    if os.path.exists(path):
        try:
            os.rmdir(path)
        except:
            shutil.rmtree(path)
    os.mkdir(path)

def init_reward_struct(path):
    if os.path.exists(path):
        return 
    os.mkdir(path)

plotting_pallete2 = ["#922B21", "#1F618D", "#117A65", "#B7950B", "#6C3483", "#A04000", "#117A65"]
pp = ["#CB4335", "#2874A6", "#229954", "#D4AC0D", "#884EA0", "#BA4A00", "#17A589"]
path_orange = "#E67E22"

pp_light = ["#EC7063", "#5499C7", "#58D68D", "#F4D03F", "#AF7AC5"]            
pp_dark = ["#943126", "#1A5276", "#1D8348", "#9A7D0A", "#633974"]
pp_darkest = ["#78281F", "#154360", "#186A3B", "#7D6608", "#512E5F"]


def plot_pallet():
    plt.figure(1)
    for i in range(len(pp)):
        plt.plot([i,i], [0,1], color=pp[i], linewidth=10)
    plt.figure(2)
    for i in range(len(plotting_pallete2)):
        plt.plot([i,i], [0,1], color=plotting_pallete2[i], linewidth=10)
    plt.show()

def ewma(data, alpha=0.2):
    if len(data) < 2:
        return np.zeros_like(data)

    alpha_rev = 1-alpha
    n = data.shape[0]

    pows = alpha_rev**(np.arange(n+1))

    scale_arr = 1/pows[:-1]
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out

def true_moving_average(data, period):
    if len(data) < period:
        return np.zeros_like(data)
    ret = np.convolve(data, np.ones(period), 'same') / period
    # t_end = np.convolve(data, np.ones(period), 'valid') / (period)
    # t_end = t_end[-1] # last valid value
    for i in range(period): # start
        t = np.convolve(data, np.ones(i+2), 'valid') / (i+2)
        ret[i] = t[0]
    for i in range(period):
        length = int(round((i + period)/2))
        t = np.convolve(data, np.ones(length), 'valid') / length
        ret[-i-1] = t[-1]
    return ret


def setup_run_list(run_file):
    full_path =  "config/" + run_file + '.yaml'
    with open(full_path) as file:
        run_dict = yaml.load(file, Loader=yaml.FullLoader)


    run_list = []
    try:
        start_n = run_dict['start_n']
    except KeyError:
        start_n = 0
    for rep in range(start_n, run_dict['n']):
        for run in run_dict['runs']:
            # base is to copy everything from the original
            for key in run_dict.keys():
                if key not in run.keys() and key != "runs":
                    run[key] = run_dict[key]

            assert run['target_position'] > 0 and run['target_position'] <= run['num_agents'], "Invalid target_position in runfile"
            # only have to add what isn't already there
            adversaries = [adv for adv in run["adversaries"]] if not run["adversaries"] is None else []
            ma_info = [info for info in run["ma_info"]] if not run["ma_info"] is None else [0.0, 0.0] 
            run["adversaries"] = adversaries
            set_n = run['set_n']
            max_speed = run['max_speed']
            run["n"] = rep
            
            run["start_train_steps"] = run["start_train_steps"] if "start_train_steps" in run else 0
            run["start_poses"] = run["start_poses"] if "start_poses" in run else "ordered"

            # Handcrafted Agents
            if run['architecture'] == "PP":
                run['run_name'] = f"{run['architecture']}_PP_{run['map_mode']}_PP_{run['map_name']}_{max_speed}_{set_n}_{rep}"
            elif run['architecture'] == "DispExt":
                run['run_name'] = f"{run['architecture']}_DispExt_{run['map_mode']}_DispExt_{run['map_name']}_{max_speed}_{set_n}_{rep}"
            # RL Agents
            else:
                run['run_name'] = f"{run['architecture']}_{str_adv(adversaries)}_{str_ma(run['ma_info'])}_{run['map_mode']}_{run['reward']}_{run['map_name']}_{max_speed}_{int(run['lr'] * 1e4)}_{int(run['gamma'] * 1e3)}_{rep}"
            run['path'] = f"{run['test_name']}/"

            run_list.append(Namespace(**run))

    init_reward_struct("Data/Vehicles/" + run_list[0].path)

    return run_list

def str_adv(adversaries):
    if len(adversaries) == 0:
        return "0"
    out = ""
    for adv in adversaries:
        out += str(ARCH_MAP[adv])
    return out

def str_ma(ma_info):
    ma_info = [int(info * 100) for info in ma_info]
    out = ""
    for info in ma_info:
        out += str(info)
    return out

@njit(cache=True)
def calculate_speed(delta, f_s=0.8, max_v=7):
    b = 0.523
    g = 9.81
    l_d = 0.329

    if abs(delta) < 0.03:
        return max_v
    if abs(delta) > 0.4:
        return 0

    V = f_s * np.sqrt(b*g*l_d/np.tan(abs(delta)))

    V = min(V, max_v)

    return V

def calculate_steering(v):
    b = 0.523
    g = 9.81
    L = 0.329

    d = np.arctan(L*b*g/(v**2))

    d = np.clip(d, 0, 0.4)

    # note always positive d return

    return d


def save_csv_array(data, filename):
    with open(filename, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def moving_average(data, period):
    return np.convolve(data, np.ones(period), 'same') / period

if __name__ == '__main__':

    plot_pallet()
