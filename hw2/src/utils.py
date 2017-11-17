
# coding: utf-8

# In[ ]:


import os
import numpy as np
import json
import re
from models import *


# In[ ]:


PAD_token = 0
SOS_token = 1
EOS_token = 2

class Lang:
    def __init__(self, name='caption'):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_words = 3 # Count default tokens

    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed: return
        self.trimmed = True
        
        keep_words = []
        
        for i in range(self.n_words):
            w = self.index2word[i]
            if w in ['PAD', 'SOS', 'EOS']:
                continue
            if self.word2count[w] >= min_count:
                keep_words.append(w)
            
        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_words = 3 # Count default tokens

        for word in keep_words:
            self.index_word(word)


# In[ ]:


# Lowercase, trim, and remove non-letter characters
def normalize(s):
    s = s.lower().strip()
    s = re.sub(r"([,.!?])", r" \1 ", s)
    s = re.sub(r"[^a-zA-Z,.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


# In[ ]:


# Return a list of indexes, one for each word in the sentence
def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def variable_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    var = Variable(torch.LongTensor(indexes).view(-1, 1))
    if USE_CUDA: var = var.cuda()
    return var


# In[ ]:


DATA_DIR = '../MLDS_hw2_data/'


# In[ ]:


filenames_train = os.listdir(DATA_DIR + 'training_data/feat/')
num_videos = len(filenames_train)
filepath = os.path.join(DATA_DIR + 'training_data/feat/', filenames_train[0])
X_0 = np.load(filepath)
if __name__ == '__main__':
    print(X_0)
num_frames, num_features = X_0.shape
if __name__ == '__main__':
    print(num_videos, num_frames, num_features)

X_train = np.zeros([num_videos, num_frames, num_features])
for i, filename in enumerate(filenames_train):
    filepath = os.path.join(DATA_DIR + 'training_data/feat/', filename)
    X_i = np.load(filepath)
    X_train[i, :, :] = X_i


# In[ ]:


filenames_test = os.listdir(DATA_DIR + 'testing_data/feat/')
num_videos_test = len(filenames_test)
filepath = os.path.join(DATA_DIR + 'testing_data/feat/', filenames_test[0])
X_0 = np.load(filepath)
if __name__ == '__main__':
    print(X_0)
num_frames, num_features = X_0.shape
if __name__ == '__main__':
    print(num_videos_test, num_frames, num_features)

X_test = np.zeros([num_videos_test, num_frames, num_features])
for i, filename in enumerate(filenames_test):
    filepath = os.path.join(DATA_DIR + 'testing_data/feat/', filename)
    X_i = np.load(filepath)
    X_test[i, :, :] = X_i


# In[ ]:


with open(DATA_DIR + 'training_label.json') as label_file:
    list_of_id_captions_dicts = json.load(label_file)

if __name__ == '__main__':
    print(list_of_id_captions_dicts[0])

captions_train = [None] * num_videos
for id_captions_dict in list_of_id_captions_dicts:
    id_ = id_captions_dict['id']
    captions = id_captions_dict['caption']
    
    index = filenames_train.index(id_ + '.npy')
    captions_train[index] = captions


# In[ ]:


captions_train_normalized = [[normalize(caption) for caption in captions] for captions in captions_train]


# In[ ]:


def filter_captions(captions_list):
    filtered_captions_list = []
    for captions in captions_list:
        filtered_captions = []
        for caption in captions:
            L = len(caption.split(' '))
            if MIN_LENGTH <= L <= MAX_LENGTH:
                filtered_captions.append(caption)
        filtered_captions_list.append(filtered_captions)
    return filtered_captions_list

captions_train_filtered = filter_captions(captions_train_normalized)


# In[ ]:


output_lang = Lang()

print("Indexing words...")
for captions in captions_train_filtered:
    for caption in captions:
        output_lang.index_words(caption)

print('Indexed {} words in output'.format(output_lang.n_words))

output_lang.trim(2)


# In[ ]:


def show_length_count(captions_list):
    length_count = list()
    for captions in captions_list:
        if len(captions) == 0:
            print('NO CAPTION!!')
        for caption in captions:
            L = len(caption.split(' '))
            length_count.append(L)

    print(min(length_count), max(length_count), len(length_count))

    import matplotlib.pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')

    plt.hist(length_count, max(length_count) - min(length_count) + 1)
    plt.show()

