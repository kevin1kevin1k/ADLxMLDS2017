
# coding: utf-8

# In[ ]:


import os
import skimage
import skimage.io
import skimage.transform
import numpy as np
import random
import scipy.misc
import argparse
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
N_EPOCHS = 100000
LEARNING_RATE = 1e-4
SAVE_IMAGE_EVERY = 100
SAVE_MODEL_EVERY = 100
IMAGE_DIR = '/mnt/disk0/kevin1kevin1k/images/'
MODEL_DIR = '/mnt/disk0/kevin1kevin1k/models/'
TAGS_PATH = './data/tags_clean.csv'
FACES_DIR = './data/faces/'


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
        tag_list = tags.split('\t')
        tags = []
        for t in tag_list:
            tag = t.split(':')[0].strip()
            if is_hair(tag) or is_eyes(tag):
                tags.append(tag)
                if tag not in tag2indices:
                    tag2indices[tag] = []
                tag2indices[tag].append(index)
        if len(tags) >= 1:
            index2tags[index] = tags

num_indices = len(index2tags)
# print(num_indices)

# has_hair = set()
# has_eyes = set()
# for tag in tag2indices:
#     if is_hair(tag):
#         print(tag, len(tag2indices[tag]))
#         has_hair.update(tag2indices[tag])
# for tag in tag2indices:
#     if is_eyes(tag):
#         print(tag, len(tag2indices[tag]))
#         has_eyes.update(tag2indices[tag])

# has_both = sorted(list(has_hair & has_eyes))
# print(len(has_both))


# In[ ]:


tags_list = sorted(list(tag2indices.keys()))
num_tags = len(tags_list)
def tags_to_vector(tags):
    vec = np.zeros((num_tags))
    for tag in tags:
        index = tags_list.index(tag)
        vec[index] = 1.0
    return vec


# In[ ]:


X = np.zeros((num_indices, IMAGE_SIZE, IMAGE_SIZE, 3))
tag_matrix = np.zeros((num_indices, num_tags))

def fill_X_and_tag_matrix():
    global X
    global tag_matrix

    for i, face_index in enumerate(index2tags):
        # (96, 96, 3)
        image = skimage.io.imread(os.path.join(FACES_DIR, face_index + '.jpg'))

        # (64, 64, 3)
        image_resized = skimage.transform.resize(image, (IMAGE_SIZE, IMAGE_SIZE), mode='constant')

        X[i] = image_resized
        tag_matrix[i] = tags_to_vector(index2tags[face_index])

    X = X.transpose(0, 3, 1, 2)
#     print(X.shape)


# In[ ]:


# https://github.com/shariqiqbal2810/WGAN-GP-PyTorch/blob/master/code/train_SVHN.py

def train_model(noise_dim=100, dim_factor=64,
                K=5, lmbda=10., batch_size=64, n_epochs=300,
                learning_rate=1e-4, verbose=False,
                betas=(0.5, 0.9)):
    z_dim = noise_dim + num_tags
    gen_net = Generator(image_size=dim_factor, z_dim=z_dim, verbose=verbose)
    disc_net = Discriminator(image_size=dim_factor, num_tags=num_tags, verbose=verbose)

    # initialize optimizers
    gen_optimizer = optim.Adam(gen_net.parameters(), lr=learning_rate,
                               betas=betas)
    disc_optimizer = optim.Adam(disc_net.parameters(), lr=learning_rate,
                                betas=betas)
    # create tensors for input to algorithm
    gen_noise_tensor = torch.FloatTensor(batch_size, noise_dim)
    fixed_noise_tensor = torch.FloatTensor(1, noise_dim)
    gp_alpha_tensor = torch.FloatTensor(batch_size, 1, 1, 1)
    # convert tensors and parameters to cuda
    if use_cuda:
        gen_net = gen_net.cuda()
        disc_net = disc_net.cuda()
        gen_noise_tensor = gen_noise_tensor.cuda()
        fixed_noise_tensor = fixed_noise_tensor.cuda()
        gp_alpha_tensor = gp_alpha_tensor.cuda()
    # wrap noise as variable so we can backprop through the graph
    gen_noise_var = Variable(gen_noise_tensor, requires_grad=False)

    fixed_noise_var = Variable(fixed_noise_tensor, requires_grad=False)
    fixed_noise_var.data.normal_()

    fixed_tags = ('blue hair', 'red eyes')
    fixed_tags_tensor = torch.FloatTensor(tags_to_vector(fixed_tags)).unsqueeze(0)
    if use_cuda:
        fixed_tags_tensor = fixed_tags_tensor.cuda()
    fixed_tags_var = Variable(fixed_tags_tensor, requires_grad=False)

    gen_loss = []
    disc_loss = []
    for ep in range(n_epochs):
        indices = random.sample(list(range(num_indices)), batch_size)
        X_data = X[indices]
        # wrap data in torch Tensor
        X_tensor = torch.FloatTensor(X_data)
        if use_cuda:
            X_tensor = X_tensor.cuda()
        X_var = Variable(X_tensor, requires_grad=False)

        tags_data = tag_matrix[indices]
        # wrap data in torch Tensor
        tags_tensor = torch.FloatTensor(tags_data)
        if use_cuda:
            tags_tensor = tags_tensor.cuda()
        tags_var = Variable(tags_tensor, requires_grad=False)

        wrong_indices = [num_indices - 1 - index for index in indices]
        wrong_tags_data = tag_matrix[wrong_indices]
        # wrap data in torch Tensor
        wrong_tags_tensor = torch.FloatTensor(wrong_tags_data)
        if use_cuda:
            wrong_tags_tensor = wrong_tags_tensor.cuda()
        wrong_tags_var = Variable(wrong_tags_tensor, requires_grad=False)

        for i in range(K):
            # train discriminator
            enable_gradients(disc_net)  # enable gradients for disc net
            disable_gradients(gen_net)  # saves computation on backprop
            disc_net.zero_grad()
            loss = wgan_gp_discriminator_loss(gen_noise_var, tags_var, wrong_tags_var, X_var, gen_net,
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
            loss = wgan_generator_loss(gen_noise_var, tags_var, gen_net, disc_net)
            loss.backward()
            gen_optimizer.step()
            # append loss to list
            gen_loss.append(loss.data[0])

        if (ep + 1) % SAVE_IMAGE_EVERY == 0:
            gen_data = gen_net(fixed_noise_var, fixed_tags_var)
            img = gen_data.data.cpu().numpy().transpose(0, 2, 3, 1)[0]
            image_path = os.path.join(IMAGE_DIR, 'bs{}'.format(batch_size), 'test{}.jpg'.format(ep + 1))
            scipy.misc.imsave(image_path, img)

        if (ep + 1) % SAVE_MODEL_EVERY == 0:
            gen_path = os.path.join(MODEL_DIR, 'bs{}'.format(batch_size), 'gen_net_{}.pt'.format(ep + 1))
            disc_path = os.path.join(MODEL_DIR, 'bs{}'.format(batch_size), 'disc_net_{}.pt'.format(ep + 1))
            torch.save(gen_net, gen_path)
            torch.save(disc_net, disc_path)

        print('Epoch {} done, G loss: {}, D loss: {}'.format(ep + 1, gen_loss[-1], disc_loss[-1]))
    return gen_loss, disc_loss


# In[ ]:


def parse():
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--batch_size', default=64, help='batch size')
    args = parser.parse_args()
    return args


# In[ ]:


def main():
    fill_X_and_tag_matrix()
    args = parse()
    gen_loss, disc_loss = train_model(
        noise_dim=NOISE_DIM,
        dim_factor=IMAGE_SIZE,
        batch_size=int(args.batch_size),
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        verbose=False,
    )


# In[ ]:


if __name__ == '__main__':
    main()

