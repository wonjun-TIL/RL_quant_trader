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


