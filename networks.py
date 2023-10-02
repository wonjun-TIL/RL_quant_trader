import os
import threading
import numpy as np

class DummyGraph:
    def as_default(self): return self
    def __enter__(self): pass
    def __exit__(self, type, vale, traceback): pass

def set_session(sess): pass

graph = DummyGraph()
sess = None

if os.environ['KERAS_BACKEND'] == 'tensorflow':
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, LSTM, Conv2D, BatchNormalization, Dropout, MaxPooling2D, Flatten
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras.backend import set_session
    import tensorflow as tf
    graph = tf.get_default_graph()
    sess = tf.compat.v1.Session()
elif os.environ['KERAS_BACKEND'] == 'plaidml.keras.backend':
    from keras.models import Model
    from keras.layers import Input, Dense, LSTM, Conv2D, BatchNormalization, Dropout, MaxPooling2D, Flatten
    from keras.optimizers import SGD


class Network:
    lock = threading.Lock() # 스레드를 이용해 병렬로 신경망을 사용하기 때문에 스레드 간의 충돌 방지를 위해 lock 클래스의 객체를 생성

    def __init__(self, input_dim=0, output_dim=0, lr=0.001, shared_network=None, activation='sigmoid', loss='mse'):
        self.input_dim = input_dim  # 입력 데이터의 크기
        self.output_dim = output_dim # 출력 데이터의 크기
        self.lr = lr    # 학습률
        self.shared_network = shared_network # 공유 신경망
        self.activation = activation # 활성화 함수
        self.loss = loss # 손실 함수
        self.model = None # 신경망 모델

    # 샘픙레 대한 행동의 가치 또는 확률 예측
    def predict(self, sample):
        with self.lock:
            with graph.as_default():
                if sess is not None:
                    set_session(sess)
                return self.model.predict(sample).flatten() # 신경망의 출력값을 반환
    
    # 학습 데이터와 레이블 x, y를 입력으로 받아서 모델을 학습시킴.
    def train_on_batch(self, x, y):
        loss = 0
        with self.lock: # lock을 사용하여 스레드 간의 충돌 방지
            with graph.as_default(): # 그래프를 사용하여 스레드 간의 충돌 방지
                if sess is not None: # 세션을 사용하여 스레드 간의 충돌 방지
                    set_session(sess) 
                loss = self.model.train_on_batch(x, y)
            return loss
        
    # 모델을 파일로 저장하는 함수
    def save_model(self, model_path):
        if model_path is not None and self.model is not None:
            self.model.save_weights(model_path, overwrite=True) # 모델의 가중치를 파일로 저장
    
    # 파일로부터 모델을 읽어오는 함수
    def load_model(self, model_path):
        if model_path is not None:
            self.model.load_weights(model_path)

    # DNN, LSTM, CNN 신경망의 공유 신경망을 생성하는 클래스 함수
    # Network 클래스의 하위 클래스들은 각각 get_network_head 함수를 가지고 있음.
    # 신경망 유형에 따라 DNN, LSTMNetwrok, CNN 클래스의 클래스함수인 get_network_head를 호출하여 공유 신경망을 생성
    @classmethod
    def get_shared_network(cls, net='dnn', num_steps=1, input_dim=0):
        with graph.as_default():
            if sess is not None:
                set_session(sess)
            if net == 'dnn':
                return DNN.get_network_head(Input((input_dim, )))
            elif net == 'lstm':
                return LSTMNetwork.get_network_head(Input((num_steps, input_dim)))
            elif net == 'cnn':
                return CNN.get_network_head(Input((1, num_steps, input_dim)))
            

