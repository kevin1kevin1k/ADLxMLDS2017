
# coding: utf-8

# In[ ]:


import os
import skimage
import skimage.io
import skimage.transform
import numpy as np
import random
import scipy.misc
from utils import *


# In[ ]:


SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.manual_seed(SEED)


# In[ ]:


IMAGE_SIZE = 64
NOISE_DIM = 100
BATCH_SIZE = 64
N_EPOCHS = 30000
LEARNING_RATE = 1e-4
TAGS_PATH = './data/tags_clean.csv'
FACES_PATH = './data/faces/'


# In[ ]:


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

index2tags = dict()
tag2indices = dict()

is_hair = lambda tag: tag.endswith(' hair') and tag not in HAIR_REMOVE
is_eyes = lambda tag: tag.endswith(' eyes') and tag not in EYES_REMOVE

with open(TAGS_PATH) as f:
    for line in f:
        index, tags = line.strip().split(',')
        index = index
        tag_list = tags.split('\t')
        tags = []
        for t in tag_list:
            tag = t.split(':')[0].strip()
            if is_hair(tag) or is_eyes(tag):
                tags.append(tag)
                if tag not in tag2indices:
                    tag2indices[tag] = []
                tag2indices[tag].append(index)
        index2tags[index] = tags

print(len(index2tags))

has_hair = set()
has_eyes = set()
for tag in tag2indices:
    if is_hair(tag):
        print(tag, len(tag2indices[tag]))
        has_hair.update(tag2indices[tag])
for tag in tag2indices:
    if is_eyes(tag):
        print(tag, len(tag2indices[tag]))
        has_eyes.update(tag2indices[tag])

has_both = sorted(list(has_hair & has_eyes))
print(len(has_both))


# In[ ]:


X = np.zeros((len(has_both), IMAGE_SIZE, IMAGE_SIZE, 3))
for i, face_index in enumerate(has_both):
    # (96, 96, 3)
    image = skimage.io.imread(os.path.join(FACES_PATH, face_index + '.jpg'))
    
    # (64, 64, 3)
    image_resized = skimage.transform.resize(image, (IMAGE_SIZE, IMAGE_SIZE), mode='constant')
    
    X[i] = image_resized

X = X.transpose(0, 3, 1, 2)
print(X.shape)


# In[ ]:


# https://github.com/shariqiqbal2810/WGAN-GP-PyTorch/blob/master/code/train_SVHN.py

def train_model(noise_dim=100, dim_factor=64,
                K=5, lmbda=10., batch_size=64, n_epochs=300,
                learning_rate=1e-4, verbose=False,
                betas=(0.5, 0.9)):
    gen_net = Generator(image_size=dim_factor, z_dim=noise_dim, verbose=verbose)
    disc_net = Discriminator(image_size=dim_factor, verbose=verbose)

    # initialize optimizers
    gen_optimizer = optim.Adam(gen_net.parameters(), lr=learning_rate,
                               betas=betas)
    disc_optimizer = optim.Adam(disc_net.parameters(), lr=learning_rate,
                                betas=betas)
    # create tensors for input to algorithm
    gen_noise_tensor = torch.FloatTensor(batch_size, noise_dim)
    gp_alpha_tensor = torch.FloatTensor(batch_size, 1, 1, 1)
    # convert tensors and parameters to cuda
    if use_cuda:
        gen_net = gen_net.cuda()
        disc_net = disc_net.cuda()
        gen_noise_tensor = gen_noise_tensor.cuda()
        gp_alpha_tensor = gp_alpha_tensor.cuda()
    # wrap noise as variable so we can backprop through the graph
    gen_noise_var = Variable(gen_noise_tensor, requires_grad=False)

    gen_loss = []
    disc_loss = []
    for ep in range(n_epochs):
        indices = random.sample(list(range(len(has_both))), batch_size)
        X_data = X[indices]
        # wrap data in torch Tensor
        X_tensor = torch.Tensor(X_data)
        if use_cuda:
            X_tensor = X_tensor.cuda()
        X_var = Variable(X_tensor, requires_grad=False)
        # calculate total iterations

        for i in range(K):
            # train discriminator
            enable_gradients(disc_net)  # enable gradients for disc net
            disable_gradients(gen_net)  # saves computation on backprop
            disc_net.zero_grad()
            loss = wgan_gp_discriminator_loss(gen_noise_var, X_var, gen_net,
                                              disc_net, lmbda, gp_alpha_tensor)
            loss.backward()
            disc_optimizer.step()
            # append loss to list
            disc_loss.append(loss.data[0])

        for i in range(1):
            # train generator
            enable_gradients(gen_net)  # enable gradients for gen net
            disable_gradients(disc_net)  # saves computation on backprop
            gen_net.zero_grad()
            loss, gen_data = wgan_generator_loss(gen_noise_var, gen_net, disc_net)
            loss.backward()
            gen_optimizer.step()
            # append loss to list
            gen_loss.append(loss.data[0])

            if ep % 100 == 0:
                img = gen_data.data.cpu().numpy().transpose(0, 2, 3, 1)[0]
                scipy.misc.imsave('./images/test{}.jpg'.format(ep), img)

        print(ep, 'done')
    return gen_loss, disc_loss


# In[ ]:


gen_loss, disc_loss = train_model(
    noise_dim=NOISE_DIM,
    dim_factor=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    n_epochs=N_EPOCHS,
    learning_rate=LEARNING_RATE,
    verbose=False,
)

