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
        