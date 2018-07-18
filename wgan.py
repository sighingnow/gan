#! /usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Wasserstein GAN.

Reference:
[1]: `Wasserstein Generative Adversarial Networks, ICML 2017,
     <http://proceedings.mlr.press/v70/arjovsky17a.html>`_
'''

import argparse
import logging
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import torchvision.datasets as vdatasets
import torchvision.transforms as vtransforms
import torchvision.utils as vutils

from mlutils import find_gpu, structuralize

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        self.model = nn.Sequential(                  # input: config.in_channels x 1 x 1
            nn.ConvTranspose2d(config.in_channels, 64 * 8,
                               4, 1, 0, bias=False), # (64 * 8) x 4 x 4
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64 * 8, 64 * 4,
                               4, 2, 1, bias=False), # (64 * 4) x 8 x 8
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64 * 4, 64 * 2,
                               4, 2, 1, bias=False), # (64 * 2) x 16 x 16
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64 * 2, 64 * 1,
                               4, 2, 1, bias=False), # (64 * 1) x 32 x 32
            nn.BatchNorm2d(64 * 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64 * 1, 1,
                               4, 2, 1, bias=False), # 1 x 64 x 64
            nn.Tanh()
        )

    def forward(self, *x):
        return self.model(*x)

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        self.model = nn.Sequential(         # input: 1 x 64 x 64
            nn.Conv2d(1, 64 * 1,
                      4, 2, 1, bias=False), # (64 * 1) x 32 x 32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 1, 64 * 2,
                      4, 2, 1, bias=False), # (64 * 2) x 16 x 16
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 2, 64 * 4,
                      4, 2, 1, bias=False), # (64 * 4) x 8 x 8
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 4, 64 * 8,
                      4, 2, 1, bias=False), # (64 * 8) x 4 x 4
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 8, 1,
                      4, 1, 0, bias=False), # 1 x 1 x 1
            # Don't add an extra Sigmoid layer at the end of discriminator.
        )

    def forward(self, *x):
        return self.model(*x).view(-1, 1)

class WGAN(object):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        self.G = Generator(config.g).to(config.device)
        self.D = Discriminator(config.d).to(config.device)

        self.G.apply(self.weights_init)
        self.D.apply(self.weights_init)

        if config.verbose:
            self.logger.info(self.G)
            self.logger.info(self.D)

        self.optim_g = optim.RMSprop(self.G.parameters(),
                                     lr=config.g.learning_rate)
        self.optim_d = optim.RMSprop(self.D.parameters(),
                                     lr=config.d.learning_rate)

        self.dataset = self.load_data()

    def weights_init(self, m):
        ''' Initialization for Conv and BatchNorm, see §4 of [1]
        '''
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(0.0, 0.02)
            m.bias.data.zero_()

    def load_data(self):
        dataset = vdatasets.MNIST(
            root=self.config.datadir,
            transform=vtransforms.Compose([
                vtransforms.Resize(64),
                vtransforms.CenterCrop(64),
                vtransforms.ToTensor(),
                vtransforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]))
        return utils.data.DataLoader(dataset,
                                     batch_size=self.config.batch_size,
                                     shuffle=True,
                                     num_workers=2)

    def train(self, num_epoch=None, start_epoch=1):
        if num_epoch is None:
            num_epoch = self.config.num_epoch

        # used to inspect the intermedia result of the models.
        fixed_noise = torch.randn(self.config.batch_size, self.config.g.in_channels, 1, 1,
                                  device=self.config.device)

        for epoch in range(start_epoch, num_epoch + start_epoch):
            for i, data in enumerate(self.dataset, 0):
                self.D.zero_grad()

                real_data = data[0].to(self.config.device)
                batch_size = real_data.size()[0]
                noise = torch.randn(batch_size, self.config.g.in_channels, 1, 1,
                                    device=self.config.device)
                fake_data = self.G(noise)

                d_error = - (torch.mean(self.D(real_data)) - torch.mean(self.D(fake_data.detach())))
                d_error.backward()
                self.optim_d.step()

                for param in self.D.parameters():
                    param.data.clamp_(-0.01, 0.01) # IMPORTANT !

                if (i + 1) % self.config.ncritic == 0:
                    self.G.zero_grad()
                    g_error = - torch.mean(self.D(fake_data))
                    g_error.backward()
                    self.optim_g.step()

            self.logger.info('Finish epoch %d' % epoch)

            if epoch % self.config.print_interval == 0:
                self.logger.info('epoch[%3d]: d_error: %f, g_error: %f',
                                 epoch, d_error.mean(), g_error.mean())
                self.G.eval()
                fixed_fake_data = self.G(fixed_noise)
                self.G.train()
                vutils.save_image(fixed_fake_data.detach(), 'wgan_fake_samples_epoch_%03d.png' % epoch, normalize=True)


parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--verbose', action='store_true')

if __name__ == '__main__':
    logger = logging.getLogger('WGAN')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter('%(asctime)s[%(name)s]%(levelname)s - %(message)s', '%m-%d %H:%M:%S'))
    logger.handlers = [handler]

    opt = parser.parse_args()

    if opt.cuda:
        try:
            device = 'cuda:%d' % (find_gpu(1)[0])
        except:
            logger.warning('CUDA device is unavailable, fall back to CPU')
            device = 'cpu'
    else:
        device = 'cpu'

    basic_config = {
        # NB: the original paper use 0.00005 as learning rate, however in the experiment I found
        # that 0.0005 is ok to training the mnist dataset.
        'g': {
            'in_channels': 100,
            'learning_rate': 0.0005,
        },
        'd': {
            'learning_rate': 0.0005,
        },
        'ncritic': 5,
        'device': device,
        'batch_size': 64,
        'num_epoch': 20,
        'print_interval': 5,
        'verbose': opt.verbose,
        'datadir': './dataset/data/mnist',
    }

    gan = WGAN(structuralize('config', **basic_config), logger)
    gan.train()
