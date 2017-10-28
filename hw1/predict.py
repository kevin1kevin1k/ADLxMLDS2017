
# coding: utf-8

# In[ ]:


import numpy as np
np.random.seed(666)
import keras
from utils import *


# In[ ]:


sentence2frames_test = dict()

def build_data_test(data_path):
    read_ark(data_path + 'fbank/test.ark', sentence2frames_test)
    read_ark(data_path + 'mfcc/test.ark', sentence2frames_test)

    # number of testing sentences
    NUM_SENTENCES_TEST = len(sentence2frames_test)

    sentences_test = sorted(sentence2frames_test)
    X_test = np.zeros((NUM_SENTENCES_TEST, MAX_SENTENCE_LEN, FEATURE_LEN))

    for i, s in enumerate(sentences_test):
        frames = sentence2frames_test[s]
        X_test[i, :len(frames)] = frames


# In[ ]:


def predict_single_model(model_path):
    model = keras.models.load_model(model_path)
    y_predict = model.predict(X_test)
    y_predict = np.argmax(y_predict, axis=-1)
    return y_predict

def predict_ensemble(model_paths):
    y_predict = np.zeros([NUM_SENTENCES_TEST, MAX_SENTENCE_LEN, NUM_CLASSES])
    for model_path in model_paths:
        model = keras.models.load_model('../models/' + model_path)
        y_predict += model.predict(X_test)
        print(model_path)

    y_predict = np.argmax(y_predict, axis=-1)
    return y_predict


# In[ ]:


def decode(seq):
    L = len(seq)
    sil = phone48_int48['sil']
    
    # trim leading sil
    for i in range(L):
        if seq[i] != sil:
            break
    start = i

    # trim trailing sil
    is_sil = False
    for i in range(L - 1, -1, -1):
        if is_sil:
            if seq[i] != sil:
                break
        else:
            if seq[i] == sil:
                is_sil = True
    end = i + 1
    
    trimmed = [phone48_char48[phone48_phone39[int48_phone48[i]]] for i in seq[start:end]]
    return''.join([k for k, v in itertools.groupby(trimmed)])


# In[ ]:


def mode(seq):
    count = {}
    lst = []
    for i in seq:
        if i not in count:
            count[i] = 1
            lst.append(i)
        else:
            count[i] += 1
    mode = max(count.values())
#     return [i for i in count if count[i] == mode]
    for i in lst:
        if count[i] == mode:
            return i

def decode_by_mode(seq, context=3):
    L = len(seq)
    sil = phone48_int48['sil']
    
    # trim leading sil
    for i in range(L):
        if seq[i] != sil:
            break
    start = i

    # trim trailing sil
    is_sil = False
    for i in range(L - 1, -1, -1):
        if is_sil:
            if seq[i] != sil:
                break
        else:
            if seq[i] == sil:
                is_sil = True
    end = i + 1
    
    trimmed = [phone48_char48[phone48_phone39[int48_phone48[i]]] for i in seq[start:end]]
    
    T = len(trimmed)
    modes = ''
    for i in range(T):
        l, r = max(0, i - context), min(T - 1, i + context)
        modes += mode(trimmed[l:r+1])[0]

    return ''.join([k for k, v in itertools.groupby(modes)])


# In[ ]:


hist = {}

def mode_history(seq):
    count = {}
    for i in seq:
        if i not in count:
            count[i] = 1
        else:
            count[i] += 1
    mode = max(count.values())
    candidates = [i for i in count if count[i] == mode]
    winner = None
    winner_tuple = None
    for c in candidates:
        if winner is None or (c in hist and hist[c] > winner_tuple):
            winner = c
            if c not in hist:
                hist[c] = (0, 0)
            winner_tuple = hist[c]
    if winner not in hist:
        hist[winner] = (1, mode)
    else:
        t = hist[winner]
        hist[winner] = (t[0] + 1, t[1] + mode)
    
    return winner


def decode_by_mode_history(seq):
    L = len(seq)
    sil = phone48_int48['sil']
    
    # trim leading sil
    for i in range(L):
        if seq[i] != sil:
            break
    start = i

    # trim trailing sil
    is_sil = False
    for i in range(L - 1, -1, -1):
        if is_sil:
            if seq[i] != sil:
                break
        else:
            if seq[i] == sil:
                is_sil = True
    end = i + 1
    
    trimmed = [phone48_char48[phone48_phone39[int48_phone48[i]]] for i in seq[start:end]]
    
    T = len(trimmed)
    modes = ''
    left = 3
    right = 3
    for i in range(T):
        l, r = max(0, i - left), min(T - 1, i + right)
        modes += mode_history(trimmed[l:r+1])[0]
    return ''.join([k for k, v in itertools.groupby(modes)])


# In[ ]:


def write_csv(csv_filename):
    y_predict_decoded = list(map(decode_by_mode_history, y_predict))

    with open(csv_filename, 'w') as prediction_f:
        prediction_f.write('id,phone_sequence\n')
        for i, sentence in enumerate(sentences_test):
            prediction_f.write('{id},{y}\n'.format(id=sentence, y=y_predict_decoded[i]))

