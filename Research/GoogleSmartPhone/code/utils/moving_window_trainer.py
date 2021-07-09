import numpy as np
import pandas as pd
from glob import glob
import os
import sys
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings(action='ignore')


class TrainHandler():
    def __init__(self, model, optimizer, loss, score, device = 'cpu'):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.score = score
        pass

    def fit(self, train_loader, valid_loader, nepoches, checkpoint_path, ):

    def train(self, data_loader):
        self.model.train()  # 신경망을 학습 모드로 전환

        # 데이터로더에서 미니배치를 하나씩 꺼내 학습을 수행
        predict = []
        ground = []

        for phys, stat, label in tqdm(data_loader, desc = 'TRAIN', leave = False):

            phys = phys.to(self.device)
            stat = stat.to(self.device)
            label = label.to(self.device)

            self.optimizer.zero_grad()  # 경사를 0으로 초기화
            pred = self.model(phys, stat)  # 데이터를 입력하고 출력을 계산

            loss = self.loss(pred, label)  # 출력과 훈련 데이터 정답 간의 오차를 계산

            loss.backward()  # 오차를 역전파 계산
            self.optimizer.step()  # 역전파 계산한 값으로 가중치를 수정

            predict.append(pred)
            ground.append(label)

        loss = self.loss(predict, ground)
        score = self.score(predict, ground)

        return loss, score

    def valid(self, data_loader):
        self.model.eval()  # 신경망을 학습 모드로 전환

        # 데이터로더에서 미니배치를 하나씩 꺼내 학습을 수행
        predict = []
        ground = []

        for phys, stat, label in tqdm(data_loader, desc = 'TRAIN', leave = False):

            phys = phys.to(self.device)
            stat = stat.to(self.device)
            label = label.to(self.device)

            pred = self.model(phys, stat)  # 데이터를 입력하고 출력을 계산

            loss = self.loss(pred, label)  # 출력과 훈련 데이터 정답 간의 오차를 계산

            predict.append(pred)
            ground.append(label)

        loss = self.loss(predict, ground)
        score = self.score(predict, ground)

        return loss, score

    def test(self, data_loader):
        self.model.eval()  # 신경망을 학습 모드로 전환

        # 데이터로더에서 미니배치를 하나씩 꺼내 학습을 수행
        predict = []

        for phys, stat, _ in tqdm(data_loader, desc = 'TRAIN', leave = False):

            phys = phys.to(self.device)
            stat = stat.to(self.device)

            pred = self.model(phys, stat)  # 데이터를 입력하고 출력을 계산

            predict.append(pred)


        return predict

class DataHandler(Dataset):
    def __init__(self, train, test, valid_size, batch_size, window_size):
        train, valid = self.train_valid_split(train, valid_size)
        self.train = train
        self.valid = valid
        self.

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def train_valid_split(self, df: pd.DataFrame, valid_size):
        phones = df['phone'].unique()

        valid_num = int(len(phones) * valid_size)
        train_num = len(phones) - valid_num

        indexes = np.array(range(len(phones)))
        indexes = np.random.choice(indexes, len(indexes))

        df_train = []
        for phone in phones[indexes[:train_num]]:
            df_train.append(df[df['phone'] == phone])
        df_train = pd.concat(df_train)

        df_valid = []
        for phone in phones[indexes[train_num:-1]]:
            df_valid.append(df[df['phone'] == phone])
        df_valid = pd.concat(df_valid)

        return df_train.reset_index().drop(columns='index'), df_valid.reset_index().drop(columns='index')
