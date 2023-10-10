import os # 폴더 생성이나 파일 경로 준비
import logging # 학습 과정 중에 정보를 기록
import abc # 추상 클래스를 정의
import collections 
import threading
import time # 학습 시간 측정
import numpy as np
from utils import sigmoid
from environment import Environment
from agent import Agent
from networks import Network, DNN, LSTMNetwork, CNN


class ReinforcementLearner:
    __metaclass__ = abc.ABCMeta
    lock = threading.Lock()

    def __init__(self, rl_method='rl', # 강화학습 방법, 하위 클래스에 따라 dqn, pg, ac, a2c, a3c 로 정해짐
                 stock_code=None, # 학습을 진행하는 주식 종목 코드
                 chart_data=None, # 환경에 해당하는 주식 일봉 차트 데이터
                 training_data=None, # 전처리된 학습 데이터
                 min_trading_unit=1, # 최소 투자 단위
                 max_trading_unit=2, # 최대 투자 단위
                 delayed_reward_threshold=.05, # 지연 보상 임계치 # 수익률이나 손실률이 이 임계값보다 클 경우 지연보상이 발생해 이전 행동들에 대한 학습이 진행됨.
                 net='dnn', # 신경망
                 num_steps=1, # LSTM, CNN에서 사용하는 샘플 묶음의 크기
                 lr=0.001, # 학습률
                 value_network=None, # 가치 신경망 
                 policy_network=None, # 정책 신경망
                 output_path='', # 학습 결과 저장 경로
                 reuse_models=True # 모델 재사용 여부
                 ):
        # 인자 확인 assert 뒤의 조건이 만족하지 않는다면 AssertionError 발생
        assert min_trading_unit > 0
        assert max_trading_unit > 0
        assert max_trading_unit >= min_trading_unit
        assert num_steps > 0
        assert lr > 0
        # 강화학습 설정
        self.rl_method = rl_method
        # 환경 설정
        self.stock_code = stock_code
        self.chart_data = chart_data
        self.environment = Environment(chart_data)
        # 에이전트 설정
        self.agent = Agent(self.environment, 
                           min_trading_unit=min_trading_unit,
                           max_trading_unit=max_trading_unit,
                           delayed_reward_threshold=delayed_reward_threshold)
        
        # 학습 데이터
        self.training_data = training_data
        self.sample = None
        self.training_data_idx = -1
        # 벡터 크기 = 학습 데이터 벡터 크기 + 에이전트 상태 크기 (학습데이터 특징: 26개, agent: 2개 )
        self.num_features = self.agent.STATE_DIM
        if self.training_data is not None:
            self.num_features += self.training_data.shape[1]
        # 신경망 설정
        self.net = net
        self.num_steps = num_steps
        self.lr = lr
        self.value_network = value_network
        self.policy_network = policy_network
        self.reuse_models = reuse_models
        # 가시화 모듈
        # self.visualizer = Visualizer()

        # 메모리, 기반으로 신경망 학습을 진행
        self.memory_sample = [] # 학습 데이터 샘플
        self.memory_action = [] # 수행한 행동
        self.memory_reward = [] # 획등한 보상
        self.memory_value = [] # 행동의 예측 가치
        self.memory_policy = [] # 행동의 예측 확률(정책)
        self.memory_pv = [] # 포트폴리오 가치,
        self.memory_num_stocks = [] # 보유 주식 수
        self.memory_exp_idx = [] # 탐험 위치
        self.memory_learning_idx = [] # 학습 위치
        # 에포크 관련 정보
        self.loss = 0.  # 학습에서 발생한 손실
        self.itr_cnt = 0 # 수익 발생 횟수
        self.exploration_cnt = 0 # 탐험 횟수 
        self.batch_size = 0 
        self.learning_cnt = 0 # 학습 횟수
        # 로그 등 출력 경로
        self.output_path = output_path