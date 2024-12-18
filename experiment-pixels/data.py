# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import numpy as np
import gym
import scipy, scipy.misc
from tqdm import tqdm

# 追加 2024/ 6/24
import cv2 
import time

# 2024/ 7/03----------------------------------------
import argparse
# --------------------------------------------------

import os, sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# 2024/06/28--------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)
# from preproc3 import main
from preproc_gray import main
# from preproc_test2 import main
# 2024/06/28--------------------------------------------------

from utils import to_pickle, from_pickle

# 2024/7/03追加-----------------------------
# 各引数を設定している．
def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    return parser.parse_args()
# -------------------------------------------

#角度の初期設定，モデルに適した形式に変換
def get_theta(obs):
    '''Transforms coordinate basis from the defaults of the gym pendulum env.'''
    #obsの0(x),1(y)の要素を取り出している．
    theta = np.arctan2(obs[0], -obs[1])
    theta = theta + np.pi/2
    theta = theta + 2*np.pi if theta < -np.pi else theta
    theta = theta - 2*np.pi if theta > np.pi else theta
    #obsのx,y情報からthetaを計算する．
    return theta

#RGB画像に対するの前処理，これにより入力画像をモデルに適した形式に変換    
# 71で画像のリサイズと正規化を行っている．画像が，縦×横(side×side)の２次元行列になる(resize)．/255の部分が正規化という作業．これは行列に含まれる要素の最大値を1にしている．0~255→0~1にしている（正規化）．70行目：クロップ（画像のある範囲の切り取り）と，赤色の抽出を行っている．
def preproc(X, side):
    '''Crops, downsamples, desaturates, etc. the rgb pendulum observation.'''
    X = X[...,0][240:-120,120:-120] - X[...,1][240:-120,120:-120]
    return scipy.misc.imresize(X, [int(side), side]) / 255.


# def preproc(X, side):
#     '''Crops, downsamples, desaturates, etc. the rgb pendulum observation.'''
#     X = X[...,0][440:-220,330:-330] - X[...,1][440:-220,330:-330]
#     return scipy.misc.imresize(X, [int(side), side]) / 255.

#gymから環境をサンプリングする関数timesteps=103, trials=200
def sample_gym(seed=0, timesteps=100, trials=200, side=28, min_angle=0., max_angle=np.pi/6, 
              verbose=False, env_name='Pendulum-v0'):

    #locals()はこの関数内で定義されている変数をすべて抽出する
    gym_settings = locals()
    #verboseの初期値はfalseであるため，関数に引数が設定されていないとき，実行されない
    if verbose:
        print("Making a dataset of pendulum pixel observations.")
        print("Edit 5/20/19: you may have to rewrite the `preproc` function depending on your screen size.")
    #env_name=振り子のゲームであるため，gymで振り子が起動
    env = gym.make(env_name)
    #振り子の環境を初期値に戻す，ここで，seed値を決定する
    env.reset() ; env.seed(seed)


    #canonical_coords,framesデータを格納するためのリスト作成
    #canonical coords:正規座標，frames:rgb値を格納，ここでframesの型はlist
    canonical_coords, frames = [], []
    
    #Video Source 2024/6/24---------------------------------------
    #cap = cv2.VideoCapture('videos/traffic.mp4') #自分のmp4のpathを入力
    #cap = cv2.VideoCapture(0)
    # Video:gymの画像サイズに合わせる．
    #frameWidth = 500
    #frameHeight = 500
    #-------------------------------------------------------------
    
    #指定数ループ
    #tqdm：処理の進捗をプログレスバーで表示させる
    #for文の繰り返し範囲はrange(trials*timesteps)の部分．
    for step in tqdm(range(trials*timesteps)):

        #stepをtimestepsで割ったあまりが0と等しいとき実行．timesteps=103で素数であるため，この操作は必ず実行される．
        if step % timesteps == 0:
            angle_ok = False

            #角度の初期化:angle_okがTrueになるまで処理を繰り返す．
            #env.restしたときの初期環境が，min~maxの間の角度に収まるようにする．
            while not angle_ok:
                #env.resetは初期環境を返す
                obs = env.reset()
                #np.abs()，()内の配列の各要素を絶対値に変換
                #初期環境(observation)の配列(cos,sin,ang velocity)を整数に変換してtheta_initに代入
                theta_init = np.abs(get_theta(obs))
                #if verbose:
                #    print("\tCalled reset. Max angle= {:.3f}".format(theta_init))

                #初期環境の値が，最小値と最大値の間に存在する時，angle_ok=Trueとなり，処理を抜け出す．
                if theta_init > min_angle and theta_init < max_angle:
                    angle_ok = True
                  
            #if verbose:
            #    print("\tRunning environment...")



        # env.render('rgb_array')で現在の環境のrgb値を返す．それを前処理し，framesのリストの末尾に付け加える．
        # env.render('rgb_array')は(1000, 1000, 3),(奥行き，縦，横)この配列のrgb情報を実動画像から抽出したい．
        # env.render('rgb_array')→rgb 2.24/6/24
        
        # ※※frames.append(preproc(env.render('rgb_array'), side))
        
        #env.step([0.])で環境に対してアクションを一つ取る．その戻り値として，obs,_,_,_がある．_は報酬，終了フラグ，追加情報
        obs, _, _, _ = env.step([0.])
        #thetaにenv.stepの環境角度を，dthetaにはobsのリストの末尾(角速度)を挿入している．
        theta, dtheta = get_theta(obs), obs[-1]

        # The constant factor of 0.25 comes from saying plotting H = PE + KE*c
        # and choosing c such that total energy is as close to constant as
        # possible. It's not perfect, but the best we can do.
        #thetaとdthetaをnumpyで同一の配列にしてcanonical_coordsの末尾に付け加える．
        canonical_coords.append( np.array([theta, 0.25 * dtheta]) )

    # ----------------------2024/07/02 ()→(timesteps,trials)
    realframes = main(timesteps,trials)
    # ----------------------

    #np.stack，.reshapeで配列を変換，ここでframesはndarrayにする．その前まではlist
    #np.stack(frames)→np.stack(realframes)に変更
    canonical_coords = np.stack(canonical_coords).reshape(trials*timesteps, -1)
    frames = np.stack(realframes).reshape(trials*timesteps, -1)
    #最終的な返り値はcanonical_coords()とframes()とgym_settings()
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
        # np.concatenateは配列の結合を行う．axis=-1より，左側に結合される．
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

#2024/7/26追加----------------------------
def install_dataset(experiment_name, save_dir, **kwargs):
  if experiment_name == "pendulum":
    env_name = "Pendulum-v0"
  else:
    assert experiment_name in ['pendulum']
  
  path = '{}/{}-pixels-dataset.pkl'.format(save_dir, experiment_name)

  data = make_gym_dataset(**kwargs)
  to_pickle(data, path)
#-----------------------------------------

def get_dataset_pixels(experiment_name, save_dir, **kwargs):
  '''Returns a dataset bult on top of OpenAI Gym observations. Also constructs
  the dataset if no saved version is available.'''
  
  if experiment_name == "pendulum":
    env_name = "Pendulum-v0"
  elif experiment_name == "acrobot":
    env_name = "Acrobot-v1"
  else:
    assert experiment_name in ['pendulum']

  path = '{}/{}-pixels-dataset.pkl'.format(save_dir, experiment_name)

  try:
      data = from_pickle(path)
      print("Successfully loaded data from {}".format(path))
  except:
      print("Had a problem loading data from {}. Rebuilding dataset...".format(path))
      # 2024/6/27コメントアウト------------------------------------
      #data = make_gym_dataset(**kwargs)
      #to_pickle(data, path)
      # -----------------------------------------------------------
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

#if __name__ == "__main__":
#    output = sample_gym()
#    print(output[1])

# 2024/6/27追加----------------------------------
if __name__ == '__main__':
    args = get_args()
    install_dataset('pendulum', args.save_dir, verbose=True, seed=args.seed)
# 2024/6/27追加----------------------------------
