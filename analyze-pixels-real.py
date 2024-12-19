import numpy as np
import torch, sys, io
import matplotlib.pyplot as plt
import os
import scipy, scipy.misc, scipy.integrate
import csv
import cv2
from IPython import display
from PIL import Image, ImageDraw, ImageSequence, ImageFont
from scipy.interpolate import UnivariateSpline
from tqdm import tqdm

EXPERIMENT_DIR = './experiment-pixels'
sys.path.append(EXPERIMENT_DIR)

from data import  make_dataset, get_dataset, hamiltonian_fn
from nn_models import MLPAutoencoder, MLP
from hnn import HNN, PixelHNN, DHNN
from utils import make_gif, L2_loss, integrate_model
solve_ivp = scipy.integrate.solve_ivp

DPI = 300
LINE_SEGMENTS = 20
LINE_WIDTH = 2
FORMAT = 'pdf'

def get_args():
    data_dir = os.path.join(EXPERIMENT_DIR, 'tar_pkl')
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
         'data_dir': './{}'.format(data_dir),
         'fig_dir': './figures'}

class ObjectView(object):
    def __init__(self, d): self.__dict__ = d
args = ObjectView(get_args())

# load dataset
data = get_dataset('forDHNN', args.data_dir)
pixel_data = get_dataset('pendulum', args.data_dir)
side = data['meta']['side']
trials = data['meta']['trials']
timesteps = data['meta']['timesteps']

frames = pixel_data['pixels'][:,:side**2].reshape(-1, side, side)[:args.num_frames]

def load_model(args):
    model = DHNN(args.latent_dim, args.hidden_dim)
    AE = MLPAutoencoder(args.input_dim, args.hidden_dim, args.latent_dim,nonlinearity='relu')
    path = "{}/pixels-pixels-{}.tar".format(args.data_dir, 'hnn')
    path_AE = "{}/pixels-pixels-{}.tar".format(args.data_dir, 'AE')
    model.load_state_dict(torch.load(path))
    AE.load_state_dict(torch.load(path_AE))
    return model, AE
hnn_model = load_model(args)[0]
AE_model = load_model(args)[1]

# ## How does the latent space look?
k = 100
fig = plt.figure(figsize=(3.25, 3), facecolor='white', dpi=DPI)
ax = fig.add_subplot(1, 1, 1, frameon=True)
latents = data['latents'].detach().numpy()
plt.plot(latents[:k,0], latents[:k,1], linestyle='-', marker=None)

ax.set_xlabel("$z_0$ (analogous to $\\theta$)")
ax.set_ylabel("$z_1 \\approx \dot z_0$ (analogous to $\dot \\theta$)")
plt.title("Latent representation of data ($z$)")

plt.tight_layout() ; plt.show()
fig.savefig('{}/latents-hnn.{}'.format(args.fig_dir, FORMAT))

# Plot total energy
k = 1000
fig = plt.figure(figsize=[5,3], dpi=DPI)
latents_tensor = torch.tensor( data['latents'], dtype=torch.float32, requires_grad=True)
z_values = latents_tensor[:k]
F2 = hnn_model.mlp_h(z_values)
F1  = hnn_model.mlp_d(z_values)
energy = F2.detach().numpy() + F1.detach().numpy()
plt.plot(energy)
plt.title("Total energy")
plt.xlabel("Timestep (every {} steps is a different trial)".format(timesteps))
plt.tight_layout() ; plt.show()
fig.savefig('{}/total-energy.{}'.format(args.fig_dir, FORMAT))

#Plotting the learned Hamiltonian on phase space
xmin, xmax = latents[:,0].min(), latents[:,0].max()
vmin, vmax = latents[:,1].min(), latents[:,1].max()

def Hplot(model,name='',save=False,dir_name=''):

    def get_field(xmin=-1.2, xmax=1.2, ymin=-1.2, ymax=1.2, gridsize=20):
        field = {'meta': locals()}

        # meshgrid to get vector field
        q, p = np.meshgrid(np.linspace(xmin, xmax, gridsize), np.linspace(ymin, ymax, gridsize))
        ys = np.stack([q, p], axis=0)  # (2,2R,2R)

        # feed through HNN model
        np_mesh_inputs = np.stack([q.flatten(), p.flatten()]).T
        mesh_inputs = torch.tensor( np_mesh_inputs, requires_grad=True, dtype=torch.float32)

        field['x'] = ys
        return mesh_inputs,q,p
    
    def get_h_gfield(model, **kwargs):
        field,_,_ = get_field(**kwargs)
        h_field = model.mlp_h(field)
        h_field = h_field.detach().cpu().numpy()
        return h_field

    R = np.pi/6
    kwargs = {'xmin': -R, 'xmax': R, 'ymin': -R, 'ymax': R, 'gridsize': 50}
    field,q,p = get_field(**kwargs)
    
    kwargs = {'xmin': xmin, 'xmax': xmax, 'ymin': vmin, 'ymax': vmax, 'gridsize': 50}
    field,q2,p2 = get_field(**kwargs)
    h_field = get_h_gfield(model,**kwargs)
    h_field = h_field.reshape(50,50)

    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=plt.figaspect(0.5),dpi=300,tight_layout=True)

    # ハミルトニアンの真値を算出
    def hamiltonian_fn(q,p):
        k = 1.9  # this coefficient must be fit to the data
        H = k*(1-np.cos(q)) + p**2 # pendulum hamiltonian
        return H

    def damping_potential_fn(p, b = 0.16):
        return 0.5 * b * p**2

    ax1 = fig.add_subplot(1,2,1, projection='3d')
    ax1.plot_surface(q, p, hamiltonian_fn(q,p), rstride=1, cstride=1, cmap='plasma')
    # ax1.plot_surface(q, p, damping_potential_fn(p), rstride=1, cstride=1, cmap='plasma')
    ax1.set_xlabel("$q$", fontsize=14)
    ax1.set_ylabel("$p$", rotation=0, fontsize=14)
    ax1.set_title('True Dissipation function', pad=10)
    ax1.set_xlim(-R, R)
    ax1.set_ylim(-R, R)
    # fig.savefig('{}/3Dhamiltonian_True_local.{}'.format(dir_name,FORMAT)) if save else plt.show()

    ax2 = fig.add_subplot(1,2,2, projection='3d')
    ax2.plot_surface(q2, p2,h_field, rstride=1, cstride=1, cmap='plasma')
    ax2.set_xlabel("$q$", fontsize=14)
    ax2.set_ylabel("$p$", rotation=0, fontsize=14)
    ax2.set_title(name, pad=10)
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(vmin, vmax)
    fig.savefig('{}/3Dhamiltonian_{}_local.{}'.format(dir_name,name,FORMAT)) if save else plt.show()
    
    return 

Hplot(hnn_model,name='DHNN',save=False,dir_name='')
# Integrate models in latent space
# Start by getting a real trajectory. Then see how well our models match it
num_frames = 101 #default is 200
test_data = make_dataset(timesteps=num_frames+3, trials=1, test_split=0)[0]
test_data_latents = get_dataset('forDHNN-not-split', args.data_dir)

# load dataset
side = test_data['meta']['side']

real_pixel_traj = test_data['pixels'][:,:side**2].reshape(-1, side, side)[:num_frames]

# Integrate models to get a simulated trajectories
# get initial values of latents
x0 = test_data_latents['latents'][19303:19304]
hnn_y0 =  x0.detach().numpy().squeeze()

# integrator settings
t_span = [0, 0.98*num_frames]
point_density = num_frames / t_span[-1]
t_eval = np.linspace(t_span[0], t_span[1], num_frames)

# integrate models
# Here we don't compute t_eval[0] because it's the initial state of the system
#   and thus is already known.
hnn_traj = integrate_model(hnn_model, t_span, hnn_y0, t_eval=t_eval[1:])

plt.plot(hnn_traj['y'][0], hnn_traj['y'][1], label='Dissipative Hamiltonian NN')
plt.legend(loc='upper right')
path = '{}/hnn_trafectory.{}'.format(args.fig_dir, FORMAT)
plt.savefig(path, dpi=300, bbox_inches='tight')  
plt.show()


# Decode into latent space and animate
# get initial values of latents
def toGif(name: str = "output", images: list = [], fps=20) -> None:
    assert len(images) > 0, "Visualizer.toGif require images list. but its empty."
    imgs = [Image.fromarray(img) for img in images]
    imgs[0].save(f"{args.fig_dir}/{name}.gif", save_all=True, append_images=imgs[1:], duration=1000 / fps, loop=0)
    print(f"saved {name}.gif")


def funs(X_0):
    print(f"input shape: {tuple(X_0.shape)}")  # (99,1568)
    X_1 = np.reshape(X_0, (X_0.shape[0], -1, 28))
    print(f"reshaped shape: {tuple(X_1.shape)}")  # (99,28,56)
    X_2 = np.split(X_1, 2, axis=1)[0]
    print(f"split shape: {tuple(X_2.shape)}")  # (99,28,28)
    # X_3 = [x for x in X_2]
    X_3 = [x for x in X_2]
    print(f"list shape: {len(X_3)}, {tuple(X_3[0].shape)}")  # 99, (28,28)
    return X_3

hnn_zs = torch.tensor( hnn_traj['y'].T, dtype=torch.float32)

# hnn_pixel_traj = AE_model.decode(hnn_zs).detach().numpy().astype(np.uint8) 
hnn_pixel_traj = (AE_model.decode(hnn_zs).detach().numpy() * 255).astype(np.uint8)
realdata_pixel = (pixel_data['pixels'][0:99,] * 255).astype(np.uint8)
toGif('EstimatedFromRealdata',funs(hnn_pixel_traj) )
toGif('Realdata',funs(realdata_pixel))

def merge_gifs(gif1_path, gif2_path, output_path):
    # GIFを開く
    gif1 = Image.open(gif1_path)
    gif2 = Image.open(gif2_path)
    
    # フレームを取得
    frames1 = [frame.copy() for frame in ImageSequence.Iterator(gif1)]
    frames2 = [frame.copy() for frame in ImageSequence.Iterator(gif2)]
    
    # フレーム数の調整
    min_frames = min(len(frames1), len(frames2))
    frames1 = frames1[:min_frames]
    frames2 = frames2[:min_frames]
    
    # 横に結合
    merged_frames = []
    for frame1, frame2 in zip(frames1, frames2):
        array1 = np.array(frame1)
        array2 = np.array(frame2)
        merged_array = np.hstack((array1, array2))
        merged_frame = Image.fromarray(merged_array)
        merged_frames.append(merged_frame)
    
    # 新しいGIFを保存
    merged_frames[0].save(
        output_path,
        save_all=True,
        append_images=merged_frames[1:],
        duration=gif1.info.get('duration', 100),  # デフォルト100ms
        loop=0
    )
    print(f"Saved merged GIF to {output_path}")

merge_gifs(f"{args.fig_dir}/EstimatedFromRealdata.gif", f"{args.fig_dir}/Realdata.gif", f"{args.fig_dir}/compare.gif")


frame_numbers = []
angles = []
csv_path = os.path.join(EXPERIMENT_DIR, 'csv', 'positions.csv')
with open(csv_path, "r") as f:
    reader = csv.reader(f)
    next(reader)  # ヘッダーをスキップ
    for row in reader:
        frame_numbers.append(int(row[0]))  # フレーム番号
        angles.append(-float(row[3]))  # 角度を逆転させて追加
frame_numbers = np.array(frame_numbers)
angles = np.array(angles)

# フレーム番号を時間ベースに変換
fps = 20
time_step = 1 / fps  
time_stamps = frame_numbers * time_step

# スプライン補間
smoothing_factor = 400  # 小さいほど荒い
spline = UnivariateSpline(time_stamps, angles, s=smoothing_factor)

# 初期値が測定値，予測値と同様になるようにスケーリングを行った．AEが正確に機能していないため，比較が難しい．
hnn_angles = hnn_traj['y'][0]
hnn_angles_scaled = ((hnn_angles - np.min(hnn_angles)) / (np.max(hnn_angles) - np.min(hnn_angles))) * (30 - (-50)) + (-50)

# 新しい時間ベースのサンプリング
new_time_stamps = np.linspace(time_stamps.min(), time_stamps.max(), 99)
new_angles = spline(time_stamps)

plt.figure(figsize=(10, 6))
plt.plot(time_stamps, new_angles, 'k-', label="Ground truth (Angle)", linestyle='--')
plt.plot(time_stamps, hnn_angles_scaled, 'b-', label='DHNN (Angle)', linewidth=2)

# グラフの装飾
plt.xlabel("Time (s)")
plt.ylabel("Angle ")
plt.legend(loc='upper right')
plt.grid(False)

# プロットの保存と表示
plt.savefig('{}/time_vs_angle.{}'.format(args.fig_dir, FORMAT), dpi=300, bbox_inches='tight')
plt.show()

