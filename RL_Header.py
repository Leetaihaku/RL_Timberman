import torch
import cv2
import numpy as np
import torch.nn.functional as F
import torchsummary as ts
import keyboard
import time
import keyboard

from torch import nn
from torch import optim
from collections import namedtuple

# 상태 차원
STATE_DIM = 5
# 행동, 행동 차원
ACTION_OPTION = ['left arrow', 'right arrow']
ACTION_DIM = 2
# 노드 수
NODES = 12
# 학습률
LEARNING_RATE = 0.01
# 할인률
GAMMA = 0.9
# 배치형식
BATCH = namedtuple('BATCH', ('state', 'action', 'q_value', 'advantage'))
# 배치사이즈
BATCH_SIZE = 10


def Actor_network():
    """액터-신경망"""
    model = nn.Sequential()
    model.add_module('fc1', nn.Linear(ACTION_DIM, NODES))
    model.add_module('relu1', nn.ReLU())
    model.add_module('fc2', nn.Linear(NODES, NODES))
    model.add_module('relu2', nn.ReLU())
    model.add_module('fc3', nn.Linear(NODES, ACTION_DIM))
    return model


def Critic_network():
    """크리틱-신경망"""
    model = nn.Sequential()
    model.add_module('fc1', nn.Linear(STATE_DIM, NODES))
    model.add_module('relu1', nn.ReLU())
    model.add_module('fc2', nn.Linear(NODES, NODES))
    model.add_module('relu2', nn.ReLU())
    model.add_module('fc3', nn.Linear(NODES, 1))
    return model


class Agent:
    """강화학습 인공지능"""
    def __init__(self):
        self.Actor = Actor_network().cuda(device='cuda')
        self.Critic = Critic_network().cuda(device='cuda')
        self.Optimizer = optim.Adam
        self.Epsilon = 1
        self.Batch = []
        self.Index = 0
        ts.summary(self.Actor, (1, ACTION_DIM), device='cuda')
        ts.summary(self.Critic, (1, STATE_DIM), device='cuda')

    def Start(self):
        keyboard.press_and_release('s')
        return

    def Action(self, state):
        """Agent 행동 추출"""
        action = torch.rand(2).cuda(device='cuda') if self.Epsilon > 0.5 else self.Actor.Net(state)
        action = action.argmax()
        keyboard.press_and_release(ACTION_OPTION[action.item()])
        return torch.tensor([1 if action > 0.5 else 0]).cuda(device='cuda')

    def Value(self, Batch):
        """Agent 가치함수 추출"""
        return self.Critic(Batch)

    def Save_batch(self, state, action, q_value, advantage):
        """BATCH 저장"""
        if len(self.Batch) < BATCH_SIZE:
            self.Batch.append(None)
        self.Batch[self.Index] = BATCH(state, action, q_value, advantage)
        self.Index = (self.Index + 1) % BATCH_SIZE

    def Advantage_and_Q_value(self, v_value, reward, next_v_value, done):
        """Advantage 및 행동가치함수 계산"""
        if done:
            q_value = reward
        else:
            q_value = reward + GAMMA * next_v_value
        advantage = q_value - v_value
        return advantage, q_value

    def Update_all_network(self):
        batch = self.Batch
        batch = BATCH(*zip(*batch))
        state_serial = torch.stack(batch.state)
        action_serial = torch.stack(batch.action)
        q_value_serial = torch.stack(batch.q_value)
        advantage_serial = torch.stack(batch.advantage)

        # 배치 비우기
        self.Batch = []

        # 액터 신경망 업데이트
        self.Actor.train()
        ######################
        # 디버깅 포인트
        # 모든 행동 softmax
        # 그 중 실제 행동 gather 추출
        # 크로스 엔트로피 입력으로 두 매개변수 전달
        actor_loss = F.cross_entropy() * advantage_serial # <- 곱하기 행동가치함수 추가 // cross_entropy가 로그-가우시안 연산을 내포 X -> 없다면 사용자정의!
        ######################
        self.Optimizer.zero_grad()
        actor_loss.backward()
        self.Optimizer.step()

        # 크리틱 신경망 업데이트
        self.Critic.train()
        critic_loss = F.mse_loss(state_serial, q_value_serial)
        self.Optimizer.zero_grad()
        critic_loss.backward()
        self.Optimizer.step()

class Environment:
    def Step(self, extracted_arr):
        """탐지화면 삼진화 -> 상태식(Domain) 생성 // x-axis :: 60 ++ 50, y-axis :: 0 ++ 320"""

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
        for i in range(len(Branch) // 2):
            Refined_branch.append(int(Branch[2 * i:2 * i + 2]))
        Refined_branch = sorted(Refined_branch, reverse=True)
        Refined_branch = str(0) if Refined_branch == [] else ''.join(map(str, Refined_branch))

        # 다음상태 저장
        Next_state = torch.tensor([float(Refined_branch), float(Player),
                                   float(Revive_Y), float(Revive_N), float(Episode_Start)], device='cuda')
        return Next_state
