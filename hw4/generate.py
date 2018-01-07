
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


def gen(tags, plot=False, save=False, save_path='', ep=30000, mult=50):
    model_dir = './models/'
    G = torch.load(os.path.join(model_dir, 'gen_net_{}.pt'.format(ep)))
    D = torch.load(os.path.join(model_dir, 'disc_net_{}.pt'.format(ep)))
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

def gen_5(index, tags, plot=False, save=False, extra=False):
    for i, ep in enumerate(range(13000, 25000+1, 3000), start=1):
        if extra:
            save_path_format = './samples/extra_sample_{}_{}.jpg'
        else:
            save_path_format = './samples/sample_{}_{}.jpg'

        save_path = save_path_format.format(index, i) if save else ''
        gen(tags, plot, save, save_path, ep)


# In[ ]:


def main():
    args = parse()
    with open(args.text_path) as f:
        for line in f:
            index, tags = line.strip().split(',')
            tag_list = tags.split(' ')
            tags = [' '.join(tag_list[i:i+2]) for i in range(0, len(tag_list), 2)]
            print(tags)
            
            gen_5(index, tags, plot=PLOT, save=True, extra=args.extra)


# In[ ]:


if __name__ == '__main__':
    main()

