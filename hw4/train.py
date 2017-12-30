
# coding: utf-8

# In[1]:


index2tags = dict()
tag2indices = dict()

with open('./data/tags_clean.csv') as f:
    for line in f:
        index, tags = line.strip().split(',')
        index = index
        tag_list = tags.split('\t')
        index2tags[index] = []
        for t in tag_list:
            tag = t.split(':')[0].strip()
            index2tags[index].append(tag)
            if tag not in tag2indices:
                tag2indices[tag] = []
            tag2indices[tag].append(index)
len(tag2indices)


# In[5]:


HAIR_REMOVE = (
    'pubic hair',
    'damage hair',
    'short hair',
    'long hair',
)

EYES_REMOVE = (
    '11 eyes',
    'bicolored eyes',
)

has_hair = set()
has_eyes = set()
for tag in tag2indices:
    if tag.endswith(' hair') and tag not in HAIR_REMOVE:
        print(tag, len(tag2indices[tag]))
        has_hair.update(tag2indices[tag])
for tag in tag2indices:
    if tag.endswith(' eyes') and tag not in EYES_REMOVE:
        print(tag, len(tag2indices[tag]))
        has_eyes.update(tag2indices[tag])

has_both = sorted(list(has_hair & has_eyes))
len(has_both)


# In[3]:


import os
import skimage
import skimage.io
import skimage.transform
import numpy as np

FACES_PATH = './data/faces/'

X = np.zeros((len(has_both), 64, 64, 3))
for i, face_index in enumerate(has_both):
    # (96, 96, 3)
    image = skimage.io.imread(os.path.join(FACES_PATH, face_index + '.jpg'))
    
    # (64, 64, 3)
    image_resized = skimage.transform.resize(image, (64, 64), mode='constant')
    
    X[i] = image_resized
X.shape

