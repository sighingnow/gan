#! /usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Deep convolutional GAN.

Reference:
[1]: `Unsupervised Representation Learning with Deep Convolutional Generative
     Adversarial Networks, ICLR 2016 <http://arxiv.org/abs/1511.06434>`_
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
                               4, 2, 1, bias=False), # 3 x 64 x 64
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
            nn.Sigmoid(),
        )

    def forward(self, *x):
        return self.model(*x).view(-1, 1).squeeze(1)

class DCGAN(object):
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

        self.loss = nn.BCELoss()

        self.optim_g = optim.Adam(self.G.parameters(),
                                  lr=config.g.learning_rate,
                                  betas=config.g.betas)
        self.optim_d = optim.Adam(self.D.parameters(),
                                  lr=config.d.learning_rate,
                                  betas=config.d.betas)

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
        if self.config.dataset[0] == 'mnist':
            dataset = vdatasets.MNIST(
                root=self.config.dataset[1],
                transform=vtransforms.Compose([
                    vtransforms.Resize(64),
                    vtransforms.CenterCrop(64),
                    vtransforms.ToTensor(),
                    vtransforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))
        elif self.config.dataset[0] == 'lfw':
            dataset = vdatasets.ImageFolder(
                root=self.config.dataset[1],
                transform=vtransforms.Compose([
                    vtransforms.Resize(64),
                    vtransforms.CenterCrop(64),
                    vtransforms.ToTensor(),
                    vtransforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))
        else:
            self.logger.critical('%s is not a valid dataset' % self.config.dataset[0])
        return utils.data.DataLoader(dataset,
                                     batch_size=self.config.batch_size,
                                     shuffle=True,
                                     num_workers=2)

    def train(self):
        # used to inspect the intermedia result of the models.
        fixed_noise = torch.randn(self.config.batch_size, self.config.g.in_channels, 1, 1,
                                  device=self.config.device)

        for epoch in range(1, self.config.num_epoch + 1):
            for _i, data in enumerate(self.dataset, 0):
                self.D.zero_grad()

                real_data = data[0].to(self.config.device)
                batch_size = real_data.size()[0]
                noise = torch.randn(batch_size, self.config.g.in_channels, 1, 1,
                                    device=self.config.device)
                fake_data = self.G(noise)

                real_label = torch.ones((batch_size, 1), device=self.config.device)
                fake_label = torch.zeros((batch_size, 1), device=self.config.device)

                d_real_error = self.loss(self.D(real_data), real_label)
                d_fake_error = self.loss(self.D(fake_data.detach()), fake_label)

                d_error = (d_real_error + d_fake_error) / 2
                d_error.backward()
                self.optim_d.step()

                self.G.zero_grad()

                g_fake_error = self.loss(self.D(fake_data), real_label)
                g_fake_error.backward()
                self.optim_g.step()

            self.logger.info('Finish epoch %d' % epoch)

            if epoch % self.config.print_interval == 0:
                self.logger.info('epoch[%3d]: d_error: %f, g_fake_error: %f',
                                 epoch, d_error.mean(), g_fake_error.mean())
                self.G.eval()
                fixed_fake_data = self.G(fixed_noise)
                self.G.train()
                vutils.save_image(fixed_fake_data.detach(), 'fake_samples_epoch_%03d.png' % epoch, normalize=True)

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--verbose', action='store_true')

if __name__ == '__main__':
    logger = logging.getLogger('DCGAN')
    logger.setLevel(logging.INFO)
    logger.handlers = [logging.StreamHandler(sys.stderr)]

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
        'g': {
            'in_channels': 100,
            'learning_rate': 2e-4,
            'betas': (0.5, 0.999),
        },
        'd': {
            'learning_rate': 2e-4,
            'betas': (0.5, 0.999),
        },
        'device': device,
        'batch_size': 64,
        'num_epoch': 5,
        'print_interval': 5,
        'verbose': opt.verbose,
        'dataset': ('mnist', 'dataset/data/mnist'), # ('lfw', './dataset/data/lfw-deepfunneled')
    }

    gan = DCGAN(structuralize('config', **basic_config), logger)
    gan.train()
