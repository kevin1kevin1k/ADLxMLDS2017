
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
from collections import namedtuple
from itertools import count
from time import time


# In[ ]:


TOTAL_STEPS = 10**7
MEMORY_SIZE = 100
TARGET_UPDATE_FREQ = 1000
EVAL_UPDATE_FREQ = 4
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 0.99
EPSILON_END = 0.05
SAVE_EVERY = 10000
MODEL_PATH = '/mnt/disk0/kevin1kevin1k/models/'


# In[ ]:


USE_CUDA = False # torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


# In[ ]:


Transition = namedtuple('Transition', ('s', 'a', 's_', 'r'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = []
        self.index = 0
        self.capacity = capacity
    
    def get_batch(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return batch
    
    def add_transition(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.index] = Transition(*args)
        self.index = (self.index + 1) % self.capacity
    
    def __len__(self):
        return len(self.memory)


# In[ ]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, 4)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# In[ ]:


def to_var(state):
    state_ = state.transpose(2, 0, 1).astype(float)
    var = Variable(FloatTensor(state_))
    var = var.unsqueeze(0)
    return var


# In[ ]:


class DQN(object):
    def __init__(self, target_update_freq):
        self.eval_net = Net()
        self.target_net = Net()
        if USE_CUDA:
            self.eval_net.cuda()
            self.target_net.cuda()

        self.memory = ReplayMemory(MEMORY_SIZE)
        self.optimizer = torch.optim.RMSprop(self.eval_net.parameters(), lr=1e-4)
        self.loss = nn.SmoothL1Loss()
        self.target_update_freq = target_update_freq
    
    def get_action(self, state, progress):
        threshold = EPSILON_START + (EPSILON_END - EPSILON_START) * progress
        threshold = max(threshold, EPSILON_END)
        use_model = random.random() > threshold
        if progress == -1: # testing mode
            use_model = True
        if use_model:
            var = to_var(state)
            actions = self.eval_net(var)
            action = actions.data.max(1)[1][0]
        else:
            action = random.randrange(4)
        return action
    
    def can_learn(self):
        return len(self.memory) >= MEMORY_SIZE
    
    def learn(self, update_target):
        transitions = self.memory.get_batch(BATCH_SIZE)
        batch = Transition(*zip(*transitions)) # transpose
        
        b_s  = Variable(torch.cat(batch.s))
        b_a  = Variable(torch.cat(batch.a))
        b_s_ = Variable(
            torch.cat([s for s in batch.s_ if s is not None]),
            volatile=True
        )
        b_r  = Variable(torch.cat(batch.r))

        q_eval = self.eval_net(b_s).gather(1, b_a)
        a_next = Variable(torch.zeros(BATCH_SIZE).type(LongTensor))
        q_next = Variable(torch.zeros(BATCH_SIZE).type(FloatTensor))
        non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, batch.s_)))
        a_next[non_final_mask] = self.target_net(b_s_).max(1)[1]
        a_next[non_final_mask] = self.target_net(b_s_).max(1)[1]
        q_next.volatile = False
        q_target = b_r + GAMMA * q_next
        loss = self.loss(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if update_target:
            self.target_net.load_state_dict(self.eval_net.state_dict())
