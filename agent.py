import numpy as np
import utils

# Agent 클래스는 투자 행동을 수행하고 투자금과 보유 주식을 관리하기 위한 클래스
class Agent:
    # 에이전트 상태가 구성하는 값 개수
    STATE_DIM = 2 # 주식 보유 비율, 포트폴리오 가치 비율 # 총 2차원

    ## 매매 수수료 및 세금
    TRADING_CHARGE = 0.00015 # 거래 수수료 (일반적으로 0.015%)
    TRADING_TAX = 0.0025 # 거래세 (실제 0.25%)수

    # 행동
    ACTION_BUY = 0 # 매수
    ACTION_SELL = 1 # 매도
    ACTION_HOLD = 2 # 홀딩
    # 인공신경망에서 확률을 구할 행동들
    ACTIONS = [ACTION_BUY, ACTION_SELL]
    NUM_ACTIONS = len(ACTIONS) # 인공 신경망에서 고려할 출력값의 개수

    def _init_(self, 
               environment, 
               min_trading_unit = 1, # 최소 매매 단위
               max_trading_unit = 2, # 최대 매매 단위
               delayed_reward_threshold = 0.05 # 지연보상 임계치
               ):
        # Environment 객체
        # 현재 주식 가격을 가져오기 위해 환경 참조
        self.environment = environment

        # 최소 매매 단위, 최대 매매 단위, 지연보상 임계치
        self.min_trading_unit = min_trading_unit
        self.max_trading_unit = max_trading_unit
        # 지연보상 임계치
        self.delayed_reward_threshold = delayed_reward_threshold

        # Agent 클래스의 속성
        self.initial_balance = 0 # 초기 자본금
        self.balance = 0 # 현재 현금 잔고
        self.num_stocks = 0 # 보유 주식 수
        # PV = balance + num_stocks * {현재 주식 가격}
        self.portfolio_value = 0
        self.base_portfolio_value = 0 # 직접 학습 시점의 PV
        self.num_buy = 0 # 매수 횟수
        self.num_sell = 0 # 매도 횟수
        self.num_hold = 0 # 홀딩 횟수
        self.immediate_reward = 0 # 즉시 보상
        self.profitloss = 0 # 현재 손익
        self.base_profitloss = 0 # 직전 지연 보상 이후 손익
        self.exploration_base = 0 # 탐험 행동 결과 기준

        # Agent 클래스의 상태
        self.ratio_hold = 0 # 주식 보유 비율
        self.ratio_portfolio_value = 0 # 포트폴리오 가치 비