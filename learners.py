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

    # 가치신경망, 손익률을 회귀분석하는 모델.
    def init_value_network(self, shared_network=None, 
                           activation='linear', 
                           loss='mse'):
        if self.net == 'dnn':
            self.value_network = DNN(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, 
                shared_network=shared_network,
                activation=activation, 
                loss=loss)
        elif self.net == 'lstm':
            self.value_network = LSTMNetwork(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, 
                num_steps=self.num_steps, 
                shared_network=shared_network,
                activation=activation, 
                loss=loss)
        elif self.net == 'cnn':
            self.value_network = CNN(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, 
                num_steps=self.num_steps, 
                shared_network=shared_network,
                activation=activation, 
                loss=loss)
        if self.reuse_models and os.path.exists(self.value_network_path):
            self.value_network.load_model(model_path=self.value_network_path)

    # 정책 신경망, PV를 높이기 위해 취하기 좋은 행동에 대한 분류 모델
    # sigmoid를 사용하여 확률로 사용할 수 있게함
    # 경우에 따라 손실함수를 mse가 아닌 cross entropy를 사용을 생각해 볼 수 있음
    def init_policy_network(self, shared_network=None, activation='sigmoid', 
                            loss='mse'):
        if self.net == 'dnn':
            self.policy_network = DNN(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, 
                shared_network=shared_network,
                activation=activation, 
                loss=loss)
        elif self.net == 'lstm':
            self.policy_network = LSTMNetwork(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, 
                num_steps=self.num_steps, 
                shared_network=shared_network,
                activation=activation, 
                loss=loss)
        elif self.net == 'cnn':
            self.policy_network = CNN(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, 
                num_steps=self.num_steps, 
                shared_network=shared_network,
                activation=activation, 
                loss=loss)
        if self.reuse_models and os.path.exists(self.policy_network_path):
            self.policy_network.load_model(model_path=self.policy_network_path)
    
    # 에포크 초기화 함수 
    def reset(self):
        self.sample = None
        self.training_data_idx = -1 # 학습 데이터를 처음부터 다시 읽기 위해
        # 환경 초기화
        self.environment.reset()
        # 에이전트 초기화
        self.agent.reset()
        # 가시화 초기화
        # self.visualizer.clear([0, len(self.chart_data)])
        # 메모리 초기화
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        self.memory_learning_idx = []
        # 에포크 관련 정보 초기화
        self.loss = 0.
        self.itr_cnt = 0 # 수행한 에포크 수
        self.exploration_cnt = 0 # 무작위 투자를 수행한 횟수를 저장
        self.batch_size = 0 # 미니배치 크기
        self.learning_cnt = 0 # 한 에포크동안 수행한 미니배치 수

    # 학습 데이터를 구성하는 샘플 하나를 생성.
    def build_sample(self):
        self.environment.observe()
        if len(self.training_data) > self.training_data_idx + 1:
            self.training_data_idx += 1
            self.sample = self.training_data.iloc[self.training_data_idx].tolist()
            self.sample.extend(self.agent.get_states())
            return self.sample
        return None
    

    # 배치 학습 데이터 생성
    @abc.abstractmethod
    def get_batch(self, batch_size, delayed_reward, discount_factor):
        pass

    # 신경망 학습
    def update_networks(self, batch_size, delayed_reward, discount_factor):
        # 배치 학습 데이터 생성
        x, y_value, y_policy = self.get_batch(batch_size, delayed_reward, discount_factor)
        
        if len(x) > 0:
            loss = 0
            if y_value is not None:
                # 가치 신경망 갱신
                loss += self.value_network.train_on_batch(x, y_value)
            if y_policy is not None:
                # 정책 신경망 갱신
                loss += self.policy_network.train_on_batch(x, y_policy)
            return loss
        return None

    def fit(self, delayed_reward, discount_factor):
        # 배치 학습 데이터 생성 및 신경망 갱신
        if self.batch_size > 0:
            _loss = self.update_networks(self.batch_size, delayed_reward, discount_factor)
            # 나중에 loss를 learning_cnt로 나누어 에포크의 학습 손실로 여김
            if _loss is not None:
                self.loss += abs(_loss)
                self.learning_cnt += 1
                self.memory_learning_idx.append(self.training_data_idx)
            self.batch_size = 0

    
    def run(self, 
            num_epoches=100, # 수행할 반복 학습 횟수
            balance = 10000000, # 초기 투자 자본금
            discount_factor = 0.9, # 상태-행동 가치를 구할 때 적용할 할인율
                                    # 보상이 발생했을 때 그 이전 보상이 발생한 시점과 현재 보상이 발생한 시점 사이에서
                                    # 수행한 행동 전체에 현재의 보상이 영향을 미침
                                    # 과거로 갈 수록 현재의 보상을 약하게 적용
            start_epsilon=0.5,  # 초기 탐험 비율. 전혀 학습되지 않은 초기에는 탐험 비율을 높여 무작위 투자를 수행. 
                                # 탐험을 통해 특정 상황에서 좋은 행동과 그렇지 않은 행동을 결정하기 위한 경험을 쌓음.
            learning=True   # 학습을 해서 신경망 모델을 만들고자 한다면 learning=True, 
                            # 학습된 모델을 가지고 투자 시뮬레이션만 하려한다면 learning=False
            ):
        info = "[{code}] RL:{rl} Net:{net} LR:{lr}" \
            "DF:{discount_factor} TU:[{min_trading_unit}," \
                "{max_trading_unit}] DRT:{delayed_reward_threshold}".format(
            code=self.stock_code, rl=self.rl_method, net=self.net,
            lr=self.lr, discount_factor=discount_factor,
            min_trading_unit=self.agent.min_trading_unit,
            max_trading_unit=self.agent.max_trading_unit,
            delayed_reward_threshold=self.agent.delayed_reward_threshold
        )

        with self.lock:
            logging.info(info)

        # 시작 시간
        time_start = time.time()


        '''가시화
        # 가시화 준비
        # 차트 데이터는 변하지 않으므로 미리 가시화
        self.visualizer.prepare(self.environment.chart_data, info)

        # 가시화 결과 저장할 폴더 준비
        self.epoch_summary_dir = os.path.join(self.output_path, f'epoch_summary_{self.stock_code}')
        if not os.path.isdir(self.epoch_summary_dir):
            os.makedirs(self.epoch_summary_dir)
        else:
            for f in os.listdir(self.epoch_summary_dir):
                os.remove(os.path.join(self.epoch_summary_dir, f))

        '''

        # 에이전트 초기 자본금 설정
        self.agent.set_balance(balance)
        
        # 학습에 대한 정보 초기화
        max_portfolio_value = 0 # 수행한 에포크 중에서 가장 높은 포트폴리오 가치
        epoch_win_cnt = 0   # 수행한 에포크 중에서 수익이 발생한 에포크 수 (포트폴리오 가치가 초기 자본금 보다 높아진 에포크 수)

        # 에포크 반복
        for epoch in range(num_epoches):
            time_start_epoch = time.time()  # 한 에포크를 수행하는데 필요한 시간 기록

            # step 샘플을 만들기 위한 큐
            q_sample = collections.deque(maxlen=self.num_steps)
            
            # 환경, 에이전트, 신경망, 가시화, 메모리 초기화
            self.reset()

            # 학습을 진행할 수록 탐험 비율 감소
            if learning:
                epsilon = self.start_epsilon * (1. - float(epoch) / (num_epoches - 1))
                self.agent.reset_exploration()
            else:
                epsilon = self.start_epsilon
            
            while True:
                # 샘플 생성
                next_sample = self.build_sample() # 환경 객체로부터 하나의 샘플을 읽어옴
                if next_sample is None: # 마지막까지 다 읽은것이므로 break
                    break

                # num_steps만큼 샘플 저장 # 다 준비돼야 행동을 결정할 수 있음.
                q_sample.append(next_sample)   
                if len(q_sample) < self.num_steps:  # 다 안찼으면 로직을 건너뜀
                    continue

                # 가치, 정책 신경망 예측
                pred_value = None
                pred_policy = None
                # 각 신경망의 predict함수를 호출해 예측 행동 가치와 예측행동 확률을 구함
                if self.value_network is not None:
                    pred_value = self.value_network.predict(list(q_sample)) 
                if self.policy_network is not None:
                    pred_policy = self.policy_network.predict(list(q_sample))
                
                # 신경망 또는 탐험에 의한 행동 결정
                # 무작위 투자 비율인 epsiolon 값의 확률로 무작위로 하거나, 
                # 신경망의 출력을 통해 결정. 
                # 정책 신경망의 출력은 매수를 했을 때와 매도를 했을 때의 포트폴리오 가치를 높일확률을 의미
                # 즉, 매수에 대한 정책 신경망 출력이 매도에 대한 출력보다 높으면 매수, 그 반대면 매도
                # 정책 신경망의 출력이 없으면 가치 신경망의 출력값이 높은 행동을 선택
                # 가치 신경망의 출력은 행동에 대한 예측가치(손익률)를 의미.

                # 결정한 행동인 action, 결정에 대한 확신도인 confidence, 무작위 투자 유무인 exploration
                action, confidence, exploration = self.agent.decide_action(pred_value, pred_policy, epsilon)

                # 결정한 행동을  수행하고 즉시 보상과 지연 보상 획득
                immediate_reward, delayed_reward = self.agent.act(action, confidence)

                # 행동 및 행동에 대한 결과를 기억
                self.memory_sample.append(list(q_sample))   # 학습 데이터의 샘플
                self.memory_action.append(action)   # 에이전트 행동
                self.memory_reward.append(immediate_reward) # 즉시보상
                if self.value_network is not None:
                    self.memory_value.append(pred_value) # 가치 신경망 출력
                if self.policy_network is not None:
                    self.memory_policy.append(pred_policy) # 정책 신경망 출력
                self.memory_pv.append(self.agent.portfolio_value) # 포트폴리오 가치
                self.memory_num_stocks.append(self.agent.num_stocks) # 보유 주식 수
                if exploration:
                    self.memory_exp_idx.append(self.training_data_idx) # 탐험 위치를 저장

                # 반복에 대한 정보 갱신
                self.batch_size += 1 # 배치크기
                self.itr_cnt += 1 # 반복 카운팅 횟수
                self.exploration_cnt += 1 if exploration else 0 # 탐험한 경우에만 1을 증가

                # 지연보상이 발생된 경우 미니 배치 학습
                # 지연보상은 지연보상임계치가 넘는 손익률이 발생했을 때 주어짐
                if learning and (delayed_reward != 0): 
                    self.fit(delayed_reward, discount_factor)

            # 에포크 종료 후 학습
            if learning:    # 남은 미니배치 학습
                self.fit(self.agent.profitloss, discount_factor)

            # 에포크 관련 정보 로그 기록
            num_epoches_digit = len(str(num_epoches)) # 1000번이면 0001 부터 에포크를 시작하기 위해
            epoch_str = str(epoch + 1).rjust(num_epoches_digit, '0')
            time_end_epoch = time.time()
            elapsed_time_epoch = time_end_epoch - time_start_epoch # 한 에포크를 수행하는데 걸린 시간
            
            # loss는 에포크 동안 수행한 미니배치들의 학습 손실을 모두 더해놓은 상태
            # loss를 학습 횟수만큼 나눠서 미니배치의 평균 학습손실로 갱신
            if self.learning_cnt > 0:
                self.loss /= self.learning_cnt 

            logging.info("[{}][Epoch {}/{}] Epsilon:{:.4f} "
                         "#Expl.:{}/{} #Buy:{} #Sell:{} #Hold:{} "
                         "#Stocks:{} PV:{:, .0f} "
                         "LC:{} Loss:{:.6f} ET:{:.4f}".format(
                             self.stock_code, # 주식 종목 코드
                             epoch_str, # 현재 에포크 번호
                             num_epoches, #
                             epsilon, # 해당 에포크에서의 탐험률
                             self.exploration_cnt, # 에포크 동안 수행한 탐험 횟수
                             self.itr_cnt,  # 에포크 동안 수행한 행동 횟수
                             self.agent.num_buy, # 에포크 동안 수행한 매수 횟수
                             self.agent.num_sell, # 에포크 동안 수행한 매도 횟수
                             self.agent.num_hold, # 에포크 동안 수행한 홀드 횟수
                             self.agent.num_stocks, # 에포크 종료 시점에 보유하고 있는 주식 수
                             self.agent.portfolio_value, # 에포크 종료 시점에 포트폴리오 가치
                             self.learning_cnt, # 에포크 동안 수행한 미니배치 학습 횟수
                             self.loss, # 에포크 동안 수행한 미니배치 학습 손실
                             elapsed_time_epoch # 에포크 수행 시간
                             ))


            # 에포크 관련 정보 가시화
            # self.visualize(epoch_str, num_epoches, epsilon)

            # 학습 관련 정보 갱신
            # 최대 포트폴리오 가치를 갱신하고 해당 에포크에서 포트폴리오 가치가 자본금보다 높으면 epoch_win_cnt를 1 증가
            max_portfolio_value = max(
                max_portfolio_value, self.agent.portfolio_value)
            if self.agent.portfolio_value > self.agent.initial_balance:
                epoch_win_cnt += 1

        # 종료 시간
        time_end = time.time()
        elapsed_time = time_end - time_start

        # 학습 관련 정보 로그 기록
        with self.lock:
            logging.info("[{code}] Elapsed Time:{elapsed_time:.4f} "
                         "Max PV:{max_pv:,.0f} #Win:{cnt_win}".format(
                             code=self.stock_code,  # 주식 종목 코드
                             elapsed_time=elapsed_time, # 학습에 소요된 시간
                             max_pv=max_portfolio_value, # 학습 동안 달성한 최대 포트폴리오 가치
                             cnt_win=epoch_win_cnt # 포트폴리오 가치가 자본금보다 높았던 에포크 수
                         ))
    
    # 신경망 모델 저장 함수
    def save_models(self):
        if self.value_network is not None and self.value_network_path is not None:
            self.value_network.save_model(self.value_network_path)
        if self.policy_network is not None and self.policy_network_path is not None:
            self.policy_network.save_model(self.policy_network_path)
