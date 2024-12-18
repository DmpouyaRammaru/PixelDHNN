# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import numpy as np
import gym
import scipy, scipy.misc
from tqdm import tqdm
import argparse

import os, sys
this_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils import to_pickle, from_pickle
from Dataset import datasetTool

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--save_dir', default=this_dir, type=str, help='where to save the trained model')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    return parser.parse_args()

def get_theta(obs):
    '''Transforms coordinate basis from the defaults of the gym pendulum env.'''
    theta = np.arctan2(obs[0], -obs[1])
    theta = theta + np.pi/2
    theta = theta + 2*np.pi if theta < -np.pi else theta
    theta = theta - 2*np.pi if theta > np.pi else theta
    return theta
    
def preproc(X, side):
    '''Crops, downsamples, desaturates, etc. the rgb pendulum observation.'''
    X = X[...,0][240:-120,120:-120] - X[...,1][240:-120,120:-120]
    return scipy.misc.imresize(X, [int(side), side]) / 255.


# def preproc(X, side):
#     '''Crops, downsamples, desaturates, etc. the rgb pendulum observation.'''
#     X = X[...,0][440:-220,330:-330] - X[...,1][440:-220,330:-330]
#     return scipy.misc.imresize(X, [int(side), side]) / 255.

def sample_gym(seed=0, timesteps=103, trials=200, side=28, min_angle=0., max_angle=np.pi/6, 
              verbose=False, env_name='Pendulum-v0'):

    gym_settings = locals()
    if verbose:
        print("Making a dataset of pendulum pixel observations.")
        print("Edit 5/20/19: you may have to rewrite the `preproc` function depending on your screen size.")
    env = gym.make(env_name)
    env.reset() ; env.seed(seed)

    canonical_coords = []
    for step in tqdm (range(trials*timesteps)):

        if step % timesteps == 0:
            angle_ok = False

            while not angle_ok:
                obs = env.reset()
                theta_init = np.abs(get_theta(obs))
                if verbose:
                    print("\tCalled reset. Max angle= {:.3f}".format(theta_init))
                if theta_init > min_angle and theta_init < max_angle:
                    angle_ok = True
                  
            if verbose:
                print("\tRunning environment...")
                
        # frames.append(preproc(env.render('rgb_array'), side))
        obs, _, _, _ = env.step([0.])
        theta, dtheta = get_theta(obs), obs[-1]

        # The constant factor of 0.25 comes from saying plotting H = PE + KE*c
        # and choosing c such that total energy is as close to constant as
        # possible. It's not perfect, but the best we can do.
        # canonical_coords.append( np.array([theta, 0.25 * dtheta]) )
    
    print(f"cc_before_type:{type(canonical_coords)}")
    print(f"cc_bfore:{len(canonical_coords)}")
    
    # canonical_coords = np.stack(canonical_coords).reshape(trials*timesteps, -1)
    # print(f"cc_type:{type(canonical_coords)}")
    # print(f"cc:{canonical_coords.shape}")

    # frames = np.stack(frames).reshape(trials*timesteps, -1)
    dataset = datasetTool

    path = "/Users/ranmaru/Documents/卒検_結果/結果＿余分/結果18_dhnn_simulation/d-1120_6_01-free.pkl"
    raw_dataset = dataset.load_pkl(path)
    qputx = raw_dataset["bef"] # qpux=dif

    theta, dtheta, anot, anot2, frames, = dataset.splitter(raw_dataset,qputx)
    theta, dtheta = np.squeeze(theta), np.squeeze(dtheta)
    theta = theta.to('cpu').detach().numpy().copy()
    dtheta = dtheta.to('cpu').detach().numpy().copy()
    frames = frames.to('cpu').detach().numpy().copy()
    frames = frames.reshape([-1,784])
    q = np.ravel(theta)
    p = np.ravel(dtheta)
    canonical_coords = np.append(p, p, axis=0).reshape(20600,2)
    print(f"theta_type:{type(q)}")
    print(f"theta_shape:{q.shape}")
    print(f"canonical:{canonical_coords.shape}")
    print(f"frames:{frames.shape}")

    return canonical_coords, frames, gym_settings

def make_gym_dataset(test_split=0.2, **kwargs):
    '''Constructs a dataset of observations from an OpenAI Gym env'''
    canonical_coords, frames, gym_settings = sample_gym(**kwargs)
    
    coords, dcoords = [], [] # position and velocity data (canonical coordinates)
    pixels, dpixels = [], [] # position and velocity data (pixel space)
    next_pixels, next_dpixels = [], [] # (pixel space measurements, 1 timestep in future)

    trials = gym_settings['trials']
    for cc, pix in zip(np.split(canonical_coords, trials), np.split(frames, trials)):
        # calculate cc offsets
        cc = cc[1:]
        dcc = cc[1:] - cc[:-1]
        cc = cc[1:]

        # concat adjacent frames to get velocity information
        # now the pixel arrays have same information as canonical coords
        # ...but in a different (highly nonlinear) basis
        p = np.concatenate([pix[:-1], pix[1:]], axis=-1)
        
        dp = p[1:] - p[:-1]
        p = p[1:]

        # calculate the same quantities, one timestep in the future
        next_p, next_dp = p[1:], dp[1:]
        p, dp = p[:-1], dp[:-1]
        cc, dcc = cc[:-1], dcc[:-1]

        # append to lists
        coords.append(cc) ; dcoords.append(dcc)
        pixels.append(p) ; dpixels.append(dp)
        next_pixels.append(next_p) ; next_dpixels.append(next_dp)

    # concatenate across trials
    data = {'coords': coords, 'dcoords': dcoords,
            'pixels': pixels, 'dpixels': dpixels, 
            'next_pixels': next_pixels, 'next_dpixels': next_dpixels}
    data = {k: np.concatenate(v) for k, v in data.items()}

    # make a train/test split
    split_ix = int(data['coords'].shape[0]* test_split)
    split_data = {}
    for k, v in data.items():
      split_data[k], split_data['test_' + k] = v[split_ix:], v[:split_ix]
    data = split_data

    gym_settings['timesteps'] -= 3 # from all the offsets computed above
    data['meta'] = gym_settings

    return data

def install_dataset(experiment_name, save_dir, **kwargs):
  if experiment_name == "pendulum":
    env_name = "Pendulum-v0"
  else:
    assert experiment_name in ['pendulum']
  
  path = '{}/{}-pixels-dataset.pkl'.format(save_dir, experiment_name)

  data = make_gym_dataset(**kwargs)
  to_pickle(data, path)

def get_dataset(experiment_name, save_dir, **kwargs):
  '''Returns a dataset bult on top of OpenAI Gym observations. Also constructs
  the dataset if no saved version is available.'''
  
  if experiment_name == "pendulum":
    env_name = "Pendulum-v0"
  elif experiment_name == "acrobot":
    env_name = "Acrobot-v1"
  else:
    assert experiment_name in ['pendulum']

  path = '{}/{}-pixels-dataset.pkl'.format(save_dir, 'forDHNN')

  try:
      data = from_pickle(path)
      print("Successfully loaded data from {}".format(path))
  except:
      print("Had a problem loading data from {}. Rebuilding dataset...".format(path))
      # data = make_gym_dataset(**kwargs)
      # to_pickle(data, path)

  return data


### FOR DYNAMICS IN ANALYSIS SECTION ###
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
    install_dataset('pendulum', args.save_dir, verbose=True, seed=args.seed)