
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, grad


# In[ ]:


# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/deep_convolutional_gan/model.py`

def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom deconvolutional layer for simplicity."""
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


class Generator(nn.Module):
    """Generator containing 7 deconvolutional layers."""
    def __init__(self, z_dim=256, image_size=64, conv_dim=64, verbose=False):
        super(Generator, self).__init__()
        self.fc = deconv(z_dim, conv_dim*8, int(image_size/16), 1, 0, bn=False)
        self.deconv1 = deconv(conv_dim*8, conv_dim*4, 4)
        self.deconv2 = deconv(conv_dim*4, conv_dim*2, 4)
        self.deconv3 = deconv(conv_dim*2, conv_dim, 4)
        self.deconv4 = deconv(conv_dim, 3, 4, bn=False)
        self.verbose = verbose

    def forward(self, z):
        if self.verbose: print(z.size())
        z = z.view(z.size(0), z.size(1), 1, 1)      # If image_size is 64, output shape is as below.
        if self.verbose: print(z.size())
        out = self.fc(z)                            # (?, 512, 4, 4)
        if self.verbose: print(out.size())
        out = F.leaky_relu(self.deconv1(out), 0.05)  # (?, 256, 8, 8)
        if self.verbose: print(out.size())
        out = F.leaky_relu(self.deconv2(out), 0.05)  # (?, 128, 16, 16)
        if self.verbose: print(out.size())
        out = F.leaky_relu(self.deconv3(out), 0.05)  # (?, 64, 32, 32)
        if self.verbose: print(out.size())
        out = F.tanh(self.deconv4(out))             # (?, 3, 64, 64)
        if self.verbose: print(out.size())
        return out

# bn set to False for WGAN-GP
def conv(c_in, c_out, k_size, stride=2, pad=1, bn=False):
    """Custom convolutional layer for simplicity."""
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


class Discriminator(nn.Module):
    """Discriminator containing 4 convolutional layers."""
    def __init__(self, image_size=64, conv_dim=64, verbose=False):
        super(Discriminator, self).__init__()
        self.conv1 = conv(3, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        self.conv4 = conv(conv_dim*4, conv_dim*8, 4)
        self.fc = conv(conv_dim*8, 1, int(image_size/16), 1, 0, False)
        self.verbose = verbose

    def forward(self, x):                         # If image_size is 64, output shape is as below.
        if self.verbose: print(x.size())
        out = F.leaky_relu(self.conv1(x), 0.05)    # (?, 64, 32, 32)
        if self.verbose: print(out.size())
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 16, 16)
        if self.verbose: print(out.size())
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 8, 8)
        if self.verbose: print(out.size())
        out = F.leaky_relu(self.conv4(out), 0.05)  # (?, 512, 4, 4)
        if self.verbose: print(out.size())
        out = self.fc(out).squeeze()
        if self.verbose: print(out.size())
        return out


# In[ ]:


# https://github.com/shariqiqbal2810/WGAN-GP-PyTorch/blob/master/code/utils.py

def wgan_generator_loss(gen_noise, gen_net, disc_net):
    """
    Generator loss for Wasserstein GAN (same for WGAN-GP)
    Inputs:
        gen_noise (PyTorch Tensor): Noise to feed through generator
        gen_net (PyTorch Module): Network to generate images from noise
        disc_net (PyTorch Module): Network to determine whether images are real
                                   or fake
    Outputs:
        loss (PyTorch scalar): Generator Loss
    """
    # draw noise
    gen_noise.data.normal_()
    # get generated data
    gen_data = gen_net(gen_noise)
    # feed data through discriminator
    disc_out = disc_net(gen_data)
    # get loss
    loss = -disc_out.mean()
    return loss, gen_data


def wgan_gp_discriminator_loss(gen_noise, real_data, gen_net, disc_net, lmbda,
                               gp_alpha):
    """
    Discriminator loss with gradient penalty for Wasserstein GAN (WGAN-GP)
    Inputs:
        gen_noise (PyTorch Tensor): Noise to feed through generator
        real_data (PyTorch Tensor): Noise to feed through generator
        gen_net (PyTorch Module): Network to generate images from noise
        disc_net (PyTorch Module): Network to determine whether images are real
                                   or fake
        lmbda (float): Hyperparameter for weighting gradient penalty
        gp_alpha (PyTorch Tensor): Values to use to randomly interpolate
                                   between real and fake data for GP penalty
    Outputs:
        loss (PyTorch scalar): Discriminator Loss
    """
    # draw noise
    gen_noise.data.normal_()
    # get generated data
    gen_data = gen_net(gen_noise)
    # feed data through discriminator
    disc_out_gen = disc_net(gen_data)
    disc_out_real = disc_net(real_data)
    # get loss (w/o GP)
    loss = disc_out_gen.mean() - disc_out_real.mean()
    # draw interpolation values
    gp_alpha.uniform_()
    # interpolate between real and generated data
    interpolates = gp_alpha * real_data.data + (1 - gp_alpha) * gen_data.data
    interpolates = Variable(interpolates, requires_grad=True)
    # feed interpolates through discriminator
    disc_out_interp = disc_net(interpolates)
    # get gradients of discriminator output with respect to input
    gradients = grad(outputs=disc_out_interp.sum(), inputs=interpolates,
                     create_graph=True)[0]
    # calculate gradient penalty
    grad_pen = ((gradients.view(gradients.size(0), -1).norm(
        2, dim=1) - 1)**2).mean()
    # add gradient penalty to loss
    loss += lmbda * grad_pen
    return loss

def enable_gradients(net):
    for p in net.parameters():
        p.requires_grad = True


def disable_gradients(net):
    for p in net.parameters():
        p.requires_grad = False


# In[ ]:


if __name__ == '__main__':
    # testing

    z_dim = 100
    G = Generator(z_dim=z_dim, image_size=64, verbose=True)
    D = Discriminator(image_size=64, verbose=True)
    z = Variable(torch.FloatTensor(10, z_dim))
    x = G(z)
    y = D(x)

    wgan_generator_loss(z, G, D)
    alpha = torch.FloatTensor(10, 1, 1, 1)
    wgan_gp_discriminator_loss(z, G(z), G, D, 0.5, alpha)

