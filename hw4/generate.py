
# coding: utf-8

# In[ ]:


import os
from train import *
try:
    import matplotlib.pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    PLOT = True
except:
    PLOT = False
print('plot:', PLOT)


# In[ ]:


TEXT_PATH = './data/testing_text.txt'


# In[ ]:


def gen(tags, plot=False, save=False, save_path='', ep=30000, mult=50):
    G = torch.load('/mnt/disk0/kevin1kevin1k/models/bs64/gen_net_{}.pt'.format(ep))
    D = torch.load('/mnt/disk0/kevin1kevin1k/models/bs64/disc_net_{}.pt'.format(ep))
    G.cpu()
    D.cpu()
    z = Variable(torch.FloatTensor(1, NOISE_DIM), requires_grad=False)
    z.data.normal_()
    tags_array = tags_to_vector(tags) * mult
    tags_tensor = torch.FloatTensor(tags_array).unsqueeze(0)
    t = Variable(tags_tensor, requires_grad=False)
    x = G(z, t)
    img = x.data.add_(1.0).mul_(0.5).cpu().numpy().transpose(0, 2, 3, 1)[0]
    if plot:
        plt.figure()
        plt.imshow(img)
    if save:
        scipy.misc.imsave(save_path, img)

def gen_5(index, tags, plot=False, save=False):
    for i, ep in enumerate((10000, 12000, 15000, 20000, 30000), start=1):
        save_path = './samples/sample_{}_{}.jpg'.format(index, i) if save else ''
        gen(tags, plot, save, save_path, ep)


# In[ ]:


def main():
    with open(TEXT_PATH) as f:
        for line in f:
            index, tags = line.strip().split(',')
            tag_list = tags.split(' ')
            tags = [' '.join(tag_list[i:i+2]) for i in range(0, len(tag_list), 2)]
            print(tags)
            
            gen_5(index, tags, plot=PLOT, save=True)
#             for m in (1, 5, 10, 50, 100):
#                 print(m)
#                 gen(tags, PLOT, save=False, mult=m)


# In[ ]:


if __name__ == '__main__':
    main()

