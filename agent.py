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
               environment,  # Environment 클래스의 객체
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
        self.portfolio_value = 0 # 보유 포트폴리오 가치 = 보유현금 + 보유 주식 수 * 현재 주가
        self.base_portfolio_value = 0 # 직접 학습 시점의 PV # 목표수익률 또는 기준 손실률을 달성하기 전의 과거 포트폴리오 가치
        self.num_buy = 0 # 매수 횟수
        self.num_sell = 0 # 매도 횟수
        self.num_hold = 0 # 홀딩 횟수
        self.immediate_reward = 0 # 에이전트가 가장 최근 행한 행동에 대한 즉시 보상 값
        self.profitloss = 0 # 현재 손익
        self.base_profitloss = 0 # 직전 지연 보상 이후 손익
        self.exploration_base = 0 # 탐험 행동 결과 기준 (매수 또는 매도 기준 정하기) # 탐험 기준 확률

        # Agent 클래스의 상태
        self.ratio_hold = 0 # 주식 보유 비율
        self.ratio_portfolio_value = 0 # 포트폴리오 가치 비

        
    # Agent 클래스의 속성 초기화
    def reset(self):
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.base_portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.immediate_reward = 0
        self.ratio_hold = 0
        self.ratio_portfolio_value = 0

    # 탐험의 기준이 되는 exporation_base 를 새로 정하는 함수. 매수 탐험을 선호하기 위해 50% 확률을 미리 부여
    def reset_exploration(self):
        self.exploration_base = 0.5 + np.random.rand() / 2
    
    # 에이전트의 초기 자본금을 설정
    def set_balance(self, balance):
        self.initial_balance = balance
    
    # 에이전트의 상태를 반환
    def get_states(self):
        """ self.ratio_hold 변수 설명
        주식 보유 비율 = 보유 주식 수 / (포트폴리오 가치 / 현재 주가)
        0 이면 주식을 하나도 보유하지 않은 것
        0.5 이면 최대 가질 수 있는 주식 대비 절반의 주식을 보유 하고 있는 것
        1 이면 최대로 주식을 보유하고 있는 것
        주식 수가 너무 적으면 매수의 관점에서 투자에 임하고 주식 수가 너무 많으면 매도의 관점에서 투자에 임하게 된다. 
        따라서 투자 행동 결정에 영향을 주기 위해 정책 신경망의 입력에 포함되어야 한다.
        """
        self.ratio_hold = self.num_stocks / int(self.portfolio_value / self.environment.get_price())

        """ self.ration_portfolio_value 변수 설명
        포트폴리오 가치 비율 = 포트폴리오 가치 / 기준 포트폴리오 가치
        포트폴리오 가치 비율은 기준 포트폴리오 가치 대비 현재 포트폴리오 가치의 비율
        기준 포트폴리오 가치는 직전에 목표 수익 또는 손익률을 달성했을 때의 포트폴리오 가치이다. 
        이 값은 현재 수익이 발생했는지 손실이 발생했는지를 판단할 수 있음
        포트폴리오 가치 비율이 0에 가까우면 손실이 큰 것이고 1보다 크면 수익이 발생했다는 뜻
        수익률이 목표 수익률에 가까우면 매도의 관점에서 투자하고는 한다.
        수익율이 투자행동에 영향을 줄 수 있기 때문에 이 값을 에이전트의 상태로 정하고 정책 신경망의 입력값으로 포함한다.
        """
        self.ratio_portfolio_value = self.portfolio_value / self.base_portfolio_value

        return (self.ratio_hold, self.ratio_portfolio_value)
    

    # 행동을 결정하는 함수, 행동은 [매도, 매수]  두가지
    def decide_action(self, pred_value, pred_policy, epsilon):
        confidence = 0.

        pred = pred_policy
        if pred is None:
            pred = pred_value
        
        if pred is None:
            # 예측 값이 없을 경우 탐험
            epsilon = 1
        else:
            # 값이 모두 같은 경우 탐험
            maxpred = np.max(pred)
            if (pred == maxpred).all():
                epsilon = 1

        # 탐험 결정
        if np.random.rand() < epsilon:  # 0~1 사이의 랜덤값을 생성하고, 이 값이 엡실론보다 작으면 무작위로 행동을 결정
            exploration = True
            if np.random.rand() < self.exploration_base:    # exploration_base 는 탐험의 기조로 작용, 에포크마다 새로 결정, 1에 가까울 수록 탐험할 때 매수를 더 많이 선택
                action = self.ACTION_BUY
            else:                                           # exploration_base 가 0에 가까울 수록 탐험할 때 매도를 더 많이 선택
                action = np.random.randint(self.NUM_ACTIONS - 1) + 1    # NUM_ACTIONS: 2 (매수, 매도 2가지)
        else:
            exploration = False
            action = np.argmax(pred)
        
        confidence = 0.5
        if pred_policy is not None:     # 정책 신경망의 출력값이 있으면, 정책 신경망의 출력값을 사용하여 탐험 결정
            confidence = pred[action]
        elif pred_value is not None:    # 정책 신경망의 출력값이 없으면, 가치 신경망의 출력값을 사용하여 탐험 결정
            confidence = utils.sigmoid(pred[action])

        return action, confidence, exploration  # 행동 (매수, 매도), 신뢰도, 탐험 여부 반환
            
    
    # 유효성 검사 함수
    def validation_action(self, action):
        if action == Agent.ACTION_BUY:
            # 적어도 1주를 살 수 있는지 확인
            if self.balance < self.environment.get_price() * (1 + self.TRADING_CHARGE) * self.min_trading_unit:
                return False
        elif action == Agent.ACTION_SELL:
            # 주식 잔고가 있는지 확인
            if self.num_stocks <= 0:
                return False
        return True
    
    # 매수 매도 단위 결정 함수
    def decide_trading_unit(self, confidence):
        if np.isnan(confidence):    # 신뢰도가 없으면 최소 단위만 매수
            return self.min_trading_unit
        # 높은 신뢰도 매수를 결정했으면 그에 맞게 더 많은 주식을 매수하고 높은 신뢰도 매도를 결정했으면 더 많은 보유 주식을 매도하는 것.
        added_traiding = max(min(int(confidence * (self.max_trading_unit - self.min_trading_unit)), self.max_trading_unit - self.min_trading_unit), 0)  # 신뢰도에 따라 매수 매도 단위 결정
        return self.min_trading_unit + added_traiding
    
    # 투자 행동 수행 함수
    # act 함수는 에이저늩가 결정한 행동을 수행
    def act(self, action, confidence):  # action 은 탐험 또는 정책 신경망을 통해 결정한 행동, 매수와 매도 를 의미하는 0 또는 1의 값을 가짐
                                        # confidence 는 정책 신경망을 통해 결정한 경우 결정한 행동에 대한 소프트맥스 확률 값
        if not self.validation_action(action):  # 유효성 검사
            action = Agent.ACTION_HOLD  # 유효하지 않은 행동이면 홀딩

        # 환경에서 현재 가격 얻기
        curr_price = self.environment.get_price()   # 현재 주가를 가져옴

        # 즉시 보상 초기화
        self.immediate_reward = 0   # 즉시 보상은 에이전트가 행동할 때마다 결정되기 때문에 초기화

        # 매수
        if action == Agent.ACTION_BUY:
            # 매수할 단위를 판단 (살 주식 수)
            trading_unit = self.decide_trading_unit(confidence)  # 매수할 단위를 판단하는 함수
            balance = (self.balance - curr_price * (1 + self.TRADING_CHARGE) * trading_unit) # 매수 후의 잔금
            # 보유 현금이 모자랄 경우 보유 현금으로 가능한 만큼 최대한 매수
            if balance < 0: 
                trading_unit = max(min(int(self.balance / (curr_price * (1 + self.TRADING_CHARGE))), self.max_trading_unit), self.min_trading_unit) # 결정한 매수 단위가 최대 단일 거래 단위를 넘어가면 최대 단일 거래 단위로 제한하고 최소 거래 단위보다 최소한 1주를 매수
            # 수수료를 적용하여 총 매수 금액 산정
            invest_amount = curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            if invest_amount > 0:
                self.balance -= invest_amount # 보유 현금을 갱신
                self.num_stocks += trading_unit # 보유 주식 수를 갱신
                self.num_buy += 1 # 매수 횟수 증가 # 통계 정보

        # 매도
        elif action == Agent.ACTION_SELL:
            # 매도할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            # 보유 주식이 모자랄 경우 가능한 만큼 최대한 매도
            trading_unit = min(trading_unit, self.num_stocks)
            # 매도
            invest_amount = curr_price * (1 - (self.TRADING_TAX + self.TRADING_CHARGE)) * trading_unit
            if invest_amount > 0:
                self.num_stocks -= trading_unit # 보유 주식 수를 갱신
                self.balance += invest_amount # 보유 현금을 갱신
                self.num_sell += 1 # 매도 횟수 증가 # 통계 정보

        # 홀딩
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1 # 홀딩 횟수 증가 # 통계 정보
        

        # 포트폴리오 가치 갱신 # 포트폴리오 가치는 잔고, 주식 보유 수, 현재 주식 가격에 의해 결정됩니다.
        self.portfolio_value = self.balance + curr_price * self.num_stocks
        self.profitloss = (self.portfolio_value - self.initial_balance) / self.initial_balance

        # 즉시 보상 - 수익률    # 즉시보상은 기준 포트폴리오 가치 대비 현재 포트폴리오 가치의 비율
        self.immediate_reward = self.profitloss

        # 지연 보상 - 익절, 손절 기준
        delayed_reward = 0
        self.base_profitloss = (self.portfolio_value - self.base_portfolio_value) / self.base_portfolio_value
        # threshold를 초과하는 경우 즉시 보상값으로 정하고 그 외의 경우 0으로 설정
        # 즉, 임계치를 초과하는 수익이 났으면 긍정, 임계치를 초과하는 손실이 났다면 부정적인 보상을 받게 됨.
        if self.base_profitloss > self.delayed_reward_threshold or self.base_profitloss < -self.delayed_reward_threshold:
            # 목표 수익률 달성하여 기준 포트폴리오 가치 갱신율
            # 또는 손실 기준치를 초과하여 기준 포트폴리오 가치 갱신
            self.base_portfolio_value = self.portfolio_value
            delayed_reward = self.immediate_reward
        else:
            delayed_reward = 0
        
        return self.immediate_reward, delayed_reward