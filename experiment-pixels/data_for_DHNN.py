import numpy as np
import torch, io
import matplotlib.pyplot as plt
from IPython import display
from PIL import Image, ImageDraw, ImageSequence, ImageFont
import scipy, scipy.misc, scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)
EXPERIMENT_DIR = './experiment-pixels'
sys.path.append(EXPERIMENT_DIR)

from data_diss_simulation import get_dataset, make_gym_dataset, sample_gym, hamiltonian_fn
from nn_models import MLPAutoencoder
from hnn import PixelHNN
from utils import to_pickle

DPI = 300
LINE_SEGMENTS = 20
LINE_WIDTH = 2
FORMAT = 'pdf'

def get_args():
    return {'input_dim': 2*784,
         'hidden_dim': 200,
         'latent_dim': 2,
         'learn_rate': 1e-3,
         'nonlinearity': 'tanh',
         'total_steps': 2000,
         'print_every': 200,
         'num_frames': 400,
         'name': 'pixels',
         'seed': 0,
         'save_dir': './{}'.format(EXPERIMENT_DIR),
         'fig_dir': './figures'}

class ObjectView(object):
    def __init__(self, d): self.__dict__ = d

args = ObjectView(get_args())

# load dataset
data = get_dataset('pendulum', args.save_dir, verbose=False)
side = data['meta']['side']
trials = data['meta']['trials']
timesteps = data['meta']['timesteps']
coords = data['coords']
test_split = 0.2
train_split = 0.8
setting = {'side':side, 'trials':trials, 'timesteps':timesteps}

frames = data['pixels'][:,:side**2].reshape(-1, side, side)[:args.num_frames]
name = '{}-dataset'.format(args.name)

def load_model(args, baseline=False):
    autoencoder = MLPAutoencoder(args.input_dim, args.hidden_dim,
                                     args.latent_dim, nonlinearity='relu')
    model = PixelHNN(args.latent_dim, args.hidden_dim, autoencoder,
                     nonlinearity=args.nonlinearity, baseline=baseline)

    path = "{}/pixels-pixels-{}.tar".format(args.save_dir,'AE')
    model.load_state_dict(torch.load(path))
    return model

AE_model = load_model(args)
print(type(data['pixels']))
print(data['pixels'].shape)
x = torch.tensor( data['pixels'], dtype=torch.float32)
x_next = torch.tensor( data['next_pixels'], dtype=torch.float32)
# x = x[:100]
# x_next = x_next[:100]
latents = AE_model.encode(x)
latents_next = AE_model.encode(x_next)
# tもいれるなら，ここで，リストやタプルを作成


latents_np = latents.detach().numpy() 
k = 3000
fig = plt.figure(figsize=(3.25, 3), facecolor='white', dpi=DPI)
ax = fig.add_subplot(1, 1, 1, frameon=True)
plt.plot(latents_np[:k,0], latents_np[:k,1], '*', markersize=2)
ax.set_xlabel("$z_0$ (analogous to $\\theta$)")
ax.set_ylabel("$z_1 \\approx \dot z_0$ (analogous to $\dot \\theta$)")
plt.title("Latent representation of data ($z$)")
plt.tight_layout() ; plt.show()


print(f'latents_next.shape:{latents.shape}')
print(f'type(x):{type(latents)}')

data = {'latents':latents, 'latents_next':latents_next, 'coords':coords}

# make a train/test split
# split_ix = int(19400 * train_split) # train / test split
# data['latents'], data['latents_test'] = latents[:split_ix], latents[split_ix:]
# data['latents_next'], data['latents_next_test'] = latents_next[:split_ix], latents_next[split_ix:]
# data['coords'], data['coords_test'] = coords[:split_ix], coords[split_ix:]
data['meta'] = setting


to_pickle(data, path = '{}/{}-pixels-dataset.pkl'.format(args.save_dir, 'forDHNN'))

