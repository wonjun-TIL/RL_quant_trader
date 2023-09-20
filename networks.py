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

    