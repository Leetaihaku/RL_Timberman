import torch
import cv2
import numpy as np
import torch.nn.functional as F
import torchsummary as ts

from torch import nn
from torch import optim

# 상태 차원
STATE_DIM = 4
# 행동, 행동 차원
ACTION_OPTION = ['left', 'right']
ACTION_DIM = 4
# 노드 수
NODES = 12
# 학습률
LEARNING_RATE = 0.01

class Actor_network:
    '''액터-신경망'''
    def __init__(self, State_dim, Action_dim):
        self.State_dim = State_dim
        self.Action_dim = Action_dim
        self.Optimizer = optim.Adam
        self.Net = self.Build_ANet().cuda(device='cuda')
        ts.summary(self.Net, (1, STATE_DIM), device='cuda')

    def Build_ANet(self):
        model = nn.Sequential()
        model.add_module('fc1', nn.Linear(self.State_dim, NODES))
        model.add_module('relu1', nn.ReLU())
        model.add_module('fc2', nn.Linear(NODES, NODES))
        model.add_module('relu2', nn.ReLU())
        model.add_module('fc2', nn.Linear(NODES, self.Action_dim))
        return model


class Critic_network:
    '''크리틱-신경망'''
    def __init__(self, State_dim):
        self.State_dim = State_dim
        self.Optimizer = optim.Adam
        self.Net = self.Build_CNet().cuda(device='cuda')
        ts.summary(self.Net, (1, STATE_DIM), device='cuda')

    def Build_CNet(self):
        model = nn.Sequential()
        model.add_module('fc1', nn.Linear(self.State_dim, NODES))
        model.add_module('relu1', nn.ReLU())
        model.add_module('fc2', nn.Linear(NODES, NODES))
        model.add_module('relu2', nn.ReLU())
        model.add_module('fc2', nn.Linear(NODES, 1))
        return model

class Agent():
    '''강화학습 인공지능'''
    def __init__(self):
        self.Actor = Actor_network(State_dim=STATE_DIM, Action_dim=ACTION_DIM)
        self.Critic = Critic_network(State_dim=STATE_DIM)
        self.Epsilon = 1

    def Action(self, state):
        '''Agent 행동 추출'''
        if self.Epsilon > 0.5:
            return ACTION_OPTION[np.random.randint(2)]
        else:
            return ACTION_OPTION[self.Actor.Net(state)]

    def Value(self, state):
        '''Agent 가치함수 추출'''
        return self.Critic.Net(state)

