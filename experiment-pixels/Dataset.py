# generate dataset from trajectory or image data
import random

import cv2
import numpy as np
import torch
import pickle


class datasetTool:
    def make(raw_data: list, dimensions: int) -> dict:
        bef_qputx, aft_qputx, dif_qputx = [], [], []
        for batch in raw_data:
            q, p, u, t, x = np.split(batch, [dimensions, 2 * dimensions, 3 * dimensions, 3 * dimensions + 1], axis=1)

            bef_q, bef_p, bef_u, bef_t, bef_x = q[:-1], p[:-1], u[:-1], t[:-1], x[:-1]
            bef_qputx.append(np.concatenate((bef_q, bef_p, bef_u, bef_t, bef_x), axis=1))

            aft_q, aft_p, aft_u, aft_t, aft_x = q[1:], p[1:], u[1:], t[1:], x[1:]
            aft_qputx.append(np.concatenate((aft_q, aft_p, aft_u, aft_t, aft_x), axis=1))

            dif_q, dif_p, dif_u, dif_t, dif_x = aft_q - bef_q, aft_p - bef_p, aft_u - bef_u, aft_t - bef_t, np.concatenate((aft_x, bef_x), axis=1)
            dif_qputx.append(np.concatenate((dif_q, dif_p, dif_u, dif_t, dif_x), axis=1))

        return {"bef": torch.Tensor(np.array(bef_qputx)), "aft": torch.Tensor(np.array(aft_qputx)), "dif": torch.Tensor(np.array(dif_qputx)), "dim": dimensions, "batch": len(raw_data)}

    def randomTTSplit(raw_dataset: dict, split_rate: float = 0.8):
        """
        split dataset to train and test randomly.
        return tuple of (train, test) dataset.
        split_rate is ratio of train dataset.
        """
        bef_qputx, aft_qputx, dif_qputx, trials = raw_dataset["bef"], raw_dataset["aft"], raw_dataset["dif"], raw_dataset["batch"]

        train_trial_num = int(trials * split_rate)

        rnd_index = random.sample(range(trials), trials)
        tr_sel, te_sel = rnd_index[:train_trial_num], rnd_index[train_trial_num:]

        return {"bef": bef_qputx[tr_sel], "aft": aft_qputx[tr_sel], "dif": dif_qputx[tr_sel], "dim": raw_dataset["dim"], "batch": train_trial_num}, {"bef": bef_qputx[te_sel], "aft": aft_qputx[te_sel], "dif": dif_qputx[te_sel], "dim": raw_dataset["dim"], "batch": trials - train_trial_num}

    def sequentialTTSplit(raw_dataset: dict):
        """
        split dataset to train and test sequentially.
        return tuple of (train, test) dataset.
        split rate is fixed to 0.5.
        """
        bef_qputx, aft_qputx, dif_qputx, trials = raw_dataset["bef"], raw_dataset["aft"], raw_dataset["dif"], raw_dataset["batch"]
        rnd_index = np.arange(trials)
        tr_sel, te_sel = rnd_index[::2], rnd_index[1::2]

        return {"bef": bef_qputx[tr_sel], "aft": aft_qputx[tr_sel], "dif": dif_qputx[tr_sel], "dim": raw_dataset["dim"], "batch": len(tr_sel)}, {"bef": bef_qputx[te_sel], "aft": aft_qputx[te_sel], "dif": dif_qputx[te_sel], "dim": raw_dataset["dim"], "batch": len(te_sel)}

    def splitter(raw_dataset: dict, qputx: torch.Tensor):
        """
        split qputx to q, p, u, t, x(torch)
        not used in this project.
        """
        dim = raw_dataset["dim"]
        out = qputx[:, :, 0:dim], qputx[:, :, dim : dim * 2], qputx[:, :, dim * 2 : dim * 3], qputx[:, :, dim * 3 : dim * 3 + 1], qputx[:, :, dim * 3 + 1 :]
        return out

    def load_pkl(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def splitter2D(raw_dataset: dict, qputx: torch.Tensor, concat: int = 0):
        """
        split qputx to q, p, u, t, x(torch)
        returns tuple of (q<points, dim>, p<points, dim>, u<points, dim>, t<points,>, x<points, height*width>)
        concat = <0,1,2>. 0 returns (q, p, u, t, x), 1 returns (qp, u, t, x), 2 returns (qpu, t, x). default is 0.
        """
        dim = raw_dataset["dim"]
        q, p, u, t, x = qputx[:, 0:dim], qputx[:, dim : dim * 2], qputx[:, dim * 2 : dim * 3], torch.squeeze(qputx[:, dim * 3 : dim * 3 + 1]), qputx[:, dim * 3 + 1 :]
        if concat == 0:
            out = (q, p, u, t, x)
        elif concat == 1:
            out = (torch.cat((q, p), dim=1), u, t, x)
        elif concat == 2:
            out = (torch.cat((q, p, u), dim=1), t, x)
        else:
            raise ValueError("concat must be 0, 1, 2")

        return out


class imageProcessor:
    def __init__(self, shape) -> None:
        self.shape = shape

    def process(self, x_raw):
        # print(f"imageProcessor.process.raw: {x_raw.shape}")
        # shrink
        x = np.array(cv2.resize(x_raw, self.shape, interpolation=cv2.INTER_LINEAR), np.float32)
        # print(f"imageProcessor.process.shrink: {x.shape}")
        # gray(0~255 -> 0~1)
        x = np.array(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), np.float32) / 255
        # print(f"imageProcessor.process.gray: {x.shape}")
        # flatten
        x = np.reshape(x, -1)
        # print(f"imageProcessor.process.flatten: {x.shape}")

        return x

    def process_all(self, X_raw):
        X = []
        for x_raw in X_raw:
            X.append(self.process(x_raw))
        # print(f"imageProcessor.process_all: {len(X)}")
        return X

if __name__ == '__main__':
    dataset = datasetTool

    path = "d-1120_3-free.pkl"
    raw_dataset = dataset.load_pkl(path)
    qputx = raw_dataset["bef"] # qpux=dif

    theta, dtheta, anot, anot2, frames, = dataset.splitter(raw_dataset,qputx)
    theta, dtheta = np.squeeze(theta), np.squeeze(dtheta)
    theta = theta.to('cpu').detach().numpy().copy()
    dtheta = dtheta.to('cpu').detach().numpy().copy()
    frames = frames.to('cpu').detach().numpy().copy()
    frames = frames.reshape([-1,784])

    print(f"type:{type(theta)}") # [200,200]
    print(f"q:{theta.shape}") # [200,200]
    print(f"p:{dtheta.shape}") # [200,200]
    print(f"x:{frames.shape}")# [40000,784]
    print(f"pixeldata{dtheta}")

    # [trials,points,dimenstions] listなら直接入れてみて，squeezなどを使う必要がある．1次元ならsqueezすれば[200,200]にできる．





