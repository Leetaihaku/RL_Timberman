import torch
import cv2
import numpy as np
import torch.nn.functional as F
import torchsummary as ts
import keyboard
import time

from torch import nn
from torch import optim

# 상태 차원
STATE_DIM = 4
# 행동, 행동 차원
ACTION_OPTION = ['left arrow', 'right arrow']
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


class Environment():
    def __init__(self):
        self.Next_state = ''

    def Step(self, extracted_arr):
        '''탐지화면 삼진화 -> 상태식(Domain) 생성 // x-axis :: 60 ++ 50, y-axis :: 0 ++ 320'''
        Branch = ''  # 나뭇가지 상태 -> 신경망 입력 형변환
        Player = ''  # 나무꾼 상태 -> 신경망 입력 형변환
        Revive_Y = ''  # 이어하기_Y 상태 -> 신경망 입력 형변환
        Revive_N = ''  # 이어하기_N 상태 -> 신경망 입력 형변환
        Episode_Start = '0'
        status = []  # 상태 임시저장 리스트

        # 이미지 추출 Raw 데이터 분해
        for data in extracted_arr:
            col_offset = data[1][0] // 320 + 1  # [y] +1 -> 상태 혼동 방지 bias
            row_offset = (data[1][1] - 60) // 50 + 1  # [x] -60 -> 모니터링 화면과 YOLO모델 픽셀 차이 상쇄 // +1 -> 상태혼동방지
            status.append([data[0], row_offset, col_offset])

        # 상태값 문자열 정리
        for i in range(len(status)):  # 신경망 입력 준비
            if status[i][0] == 'Branch':
                Branch += str(status[i][1]) + str(status[i][2])
            elif status[i][0] == 'Player':
                Player += str(status[i][1]) + str(status[i][2])
            elif status[i][0] == 'Revive_Y':
                Revive_Y += str(status[i][1]) + str(status[i][2])
            elif status[i][0] == 'Revive_N':
                Revive_N += str(status[i][1]) + str(status[i][2])
            elif status[i][0] == 'Episode_Start':
                Episode_Start = '1'
            else:
                print('상태 스택 쌓기 모듈에 알 수 없는 에러 발생')

        # 널 값 점검 조건부 -> 만일의 널 값 대비
        Branch = str(0) if Branch == '' else Branch
        Player = str(0) if Player == '' else Player
        Revive_Y = str(0) if Revive_Y == '' else Revive_Y
        Revive_N = str(0) if Revive_N == '' else Revive_N

        # 나뭇가지 데이터 정제(동일상태 상이인식 방지 => 근->원)
        Refined_branch = []
        for i in range(len(Branch)//2):
            Refined_branch.append(int(Branch[2*i:2*i+2]))
        Refined_branch = sorted(Refined_branch, reverse=True)
        Refined_branch = str(0) if Refined_branch == [] else ''.join(map(str, Refined_branch))

        # 다음상태 저장
        Next_state = torch.tensor([int(Refined_branch), int(Player), int(Revive_Y), int(Revive_N), int(Episode_Start)],
                                  device='cuda')
        return Next_state
