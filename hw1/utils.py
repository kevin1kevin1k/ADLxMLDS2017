
# coding: utf-8

# In[ ]:


import numpy as np
np.random.seed(666)
from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, Bidirectional, Dropout, Conv1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import itertools


# In[ ]:


DROPOUT = 0.2
FEATURE_LEN = 108
OPTIMIZER = Adam(lr=0.0005)
ACTIVATION = 'softmax'
NUM_CLASSES = 48
MAX_SENTENCE_LEN = 777


PREDICTION_PATH = '../predictions/'


# In[ ]:


phone48_phone39 = dict()
phone48_char48 = dict()
phone48_int48 = dict()
int48_phone48 = dict()

def read_maps(data_path):
    with open(data_path + 'phones/48_39.map') as phone48_phone39_f:
        for line in phone48_phone39_f:
            phone48, phone39 = line.strip().split('\t')
            phone48_phone39[phone48] = phone39
            if phone48 not in phone48_int48:
                int48 = len(phone48_int48)
                phone48_int48[phone48] = int48
                int48_phone48[int48] = phone48

    with open(data_path + '48phone_char.map') as phone48_char48_f:
        for line in phone48_char48_f:
            phone48, int48, char48 = line.strip().split('\t')
            phone48_char48[phone48] = char48


# In[ ]:


def read_ark(filepath, sentence2frames):
    with open(filepath) as train_ark_f:
        last_sentence = ''
        frames = list()
        is_first_line = True
        for line in train_ark_f:
            line = line.strip().split()
            instance_id = line[0]
            speaker_id, sentence_id, frame_id = instance_id.split('_')
            sentence = '_'.join((speaker_id, sentence_id))
            features = list(map(float, line[1:]))

            if sentence != last_sentence and not is_first_line:
                if last_sentence not in sentence2frames:
                    sentence2frames[last_sentence] = frames
                else:
                    sentence2frames[last_sentence] = np.concatenate(
                        (sentence2frames[last_sentence], frames),
                        axis=1
                    )
                frames = list()

            last_sentence = sentence
            frames.append(features)
            is_first_line = False

        # process last sentence
        if last_sentence not in sentence2frames:
            sentence2frames[last_sentence] = frames
        else:
            sentence2frames[last_sentence] = np.concatenate(
                (sentence2frames[last_sentence], frames),
                axis=1
            )
        frames = list()


# In[ ]:


def read_label(filepath, sentence2labels):
    with open(filepath) as train_lab_f:
        last_sentence = ''
        labels = list()
        is_first_line = True
        
        for line in train_lab_f:
            instance_id, label = line.strip().split(',')
            speaker_id, sentence_id, frame_id = instance_id.split('_')
            sentence = '_'.join((speaker_id, sentence_id))
            index = phone48_int48[label]


            if sentence != last_sentence and not is_first_line:
                sentence2labels[last_sentence] = labels
                labels = list()

            last_sentence = sentence
            labels.append(index)
            is_first_line = False

        # process last sentence
        sentence2labels[last_sentence] = labels
        labels = list()


# In[ ]:


def one_hot(n, i):
    ans = np.zeros(n)
    ans[i] = 1
    return ans


# In[ ]:


def build_rnn_model(summary=True):
    NUM_UNITS = 500
    NUM_LSTM_LAYERS = 4
    
    model = Sequential()
    for i in range(NUM_LSTM_LAYERS):
        model.add(
            Bidirectional(
                LSTM(units=NUM_UNITS, dropout=DROPOUT, return_sequences=True),
                input_shape=(MAX_SENTENCE_LEN, FEATURE_LEN)
            )
        )
    model.add(TimeDistributed(Dense(NUM_CLASSES, activation=ACTIVATION)))
    model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])
    if summary:
        model.summary()
    return model


# In[ ]:


def build_cnn_rnn_model(summary=True):
    NUM_UNITS = 300
    NUM_LSTM_LAYERS = 2
    KERNEL_SIZE = 5
    FILTERS = 32

    model = Sequential()
    model.add(Conv1D(FILTERS,
                     KERNEL_SIZE,
                     padding='same',
                     activation='relu',
                     strides=1,
                     input_shape=(MAX_SENTENCE_LEN, FEATURE_LEN)))
    for i in range(NUM_LSTM_LAYERS):
        model.add(
            Bidirectional(
                LSTM(units=NUM_UNITS, dropout=DROPOUT, return_sequences=True)
            )
        )
    model.add(TimeDistributed(Dense(NUM_CLASSES, activation=ACTIVATION)))
    model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])

    if summary:
        model.summary()
    return model

