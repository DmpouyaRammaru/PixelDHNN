# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import numpy as np
import scipy, scipy.misc
from tqdm import tqdm
import argparse
import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_PROC_DIR = os.path.join(os.path.dirname(__file__), 'img_proc')
sys.path.append(IMG_PROC_DIR)
sys.path.append(PARENT_DIR)
from preproc import preproc_main
from utils import to_pickle, from_pickle


def get_args():
    SAVE_DIR = os.path.join(THIS_DIR, 'tar_pkl')
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--save_dir', default=SAVE_DIR, type=str, help='where to save the trained model')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    return parser.parse_args()


def sample_dataset(timesteps=100, trials=200, side=28):
    settings = locals()
    realframes = preproc_main(timesteps,trials)
    frames = np.stack(realframes).reshape(trials*timesteps, -1)
    return frames, settings


def make_dataset(test_split=0.2, **kwargs):
    '''Constructs a dataset of observations from an OpenAI Gym env'''
    frames, settings = sample_dataset(**kwargs)
    
    pixels, dpixels = [], [] # position and velocity data (pixel space)
    next_pixels, next_dpixels = [], [] # (pixel space measurements, 1 timestep in future)

    trials = settings['trials']
    for pix in np.split(frames, trials):
        # concat adjacent frames to get velocity information
        # now the pixel arrays have same information as canonical coords
        # ...but in a different (highly nonlinear) basis
        p = np.concatenate([pix[:-1], pix[1:]], axis=-1)
        
        dp = p[1:] - p[:-1]
        p = p[1:]

        # calculate the same quantities, one timestep in the future
        next_p, next_dp = p[1:], dp[1:]
        p, dp = p[:-1], dp[:-1]

        # append to lists
        pixels.append(p) ; dpixels.append(dp)
        next_pixels.append(next_p) ; next_dpixels.append(next_dp)

    # concatenate across trials
    data = {'pixels': pixels, 'dpixels': dpixels, 'next_pixels': next_pixels, 'next_dpixels': next_dpixels}
    data = {k: np.concatenate(v) for k, v in data.items()}
    data_notsplit = data 
    # make a train/test split
    split_ix = int(data['pixels'].shape[0]* test_split)
    split_data = {}
    for k, v in data.items():
      split_data[k], split_data['test_' + k] = v[split_ix:], v[:split_ix]
    data = split_data

    settings['timesteps'] -= 3 # from all the offsets computed above
    data['meta'] = settings
    data_notsplit['meta'] = settings 

    return data, data_notsplit


def install_dataset(experiment_name, save_dir, **kwargs):
  path = '{}/{}-pixels-dataset.pkl'.format(save_dir, experiment_name)
  path2 = '{}/{}-not-split-pixels-dataset.pkl'.format(save_dir, experiment_name)
  data, data_notsplit = make_dataset(**kwargs)
  to_pickle(data, path)
  to_pickle(data_notsplit,path2)


def get_dataset(experiment_name, save_dir):
  path = '{}/{}-pixels-dataset.pkl'.format(save_dir, experiment_name)
  data = from_pickle(path)
  return data


def hamiltonian_fn(coords):
  k = 1.9  # this coefficient must be fit to the data
  q, p = np.split(coords,2)
  H = k*(1-np.cos(q)) + p**2 # pendulum hamiltonian
  return H


def dynamics_fn(t, coords):
  dcoords = autograd.grad(hamiltonian_fn)(coords)
  dqdt, dpdt = np.split(dcoords,2)
  S = -np.concatenate([dpdt, -dqdt], axis=-1)
  return S


if __name__ == '__main__':
    args = get_args()
    install_dataset('pendulum', args.save_dir)