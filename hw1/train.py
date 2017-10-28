
# coding: utf-8

# In[ ]:


import numpy as np
np.random.seed(666)
from utils import *

# # GPU usage
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.6
# set_session(tf.Session(config=config))


# In[ ]:


sentence2frames = dict()
sentence2labels = dict()

def build_data(data_path):
    read_ark(data_path + 'fbank/train.ark', sentence2frames)
    read_ark(data_path + 'mfcc/train.ark', sentence2frames)
    read_label(data_path + 'label/train.lab', sentence2labels)
    
    NUM_SENTENCES = len(sentence2frames)
    sentences = sorted(sentence2labels)
    X = np.zeros((NUM_SENTENCES, MAX_SENTENCE_LEN, FEATURE_LEN))
    y = np.zeros((NUM_SENTENCES, MAX_SENTENCE_LEN, NUM_CLASSES))

    for i, s in enumerate(sentences):
        frames = sentence2frames[s]
        X[i, :len(frames)] = frames
        y[i, :len(frames)] = np.stack([one_hot(NUM_CLASSES, l) for l in sentence2labels[s]])


# In[ ]:


def train_rnn(model_path):
    model = build_rnn_model(summary=True)
    monitor = 'loss'
    model_filepath = model_path + 'model_{monitor}'.format(monitor=monitor) + '_{epoch:03d}_{loss:.4f}_mapafter_lstm4_dp0.2_hidden500_adam0.0005.h5'
    checkpoint = ModelCheckpoint(model_filepath, monitor=monitor, verbose=0, save_best_only=True)
    model.fit(X, y, batch_size=4, epochs=15, verbose=1, callbacks=[checkpoint])
    
def train_cnn_rnn(model_path):
    model = build_cnn_rnn_model(summary=True)
    monitor = 'loss'
    model_filepath = model_path + 'model_{monitor}'.format(monitor=monitor) + '_{epoch:03d}_{loss:.4f}_mapafter_conv1d_lstm2_dp0.2_hidden300_adam0.0005.h5'
    checkpoint = ModelCheckpoint(model_filepath, monitor=monitor, verbose=0, save_best_only=True)
    model.fit(X, y, batch_size=32, epochs=20, verbose=1, callbacks=[checkpoint])

