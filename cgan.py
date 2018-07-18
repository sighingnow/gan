#! /usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Conditional Generative Adversarial Nets.

Reference:
[1]: `Conditional Generative Adversarial Nets <https://arxiv.org/pdf/1411.1784>`_
'''

import argparse
import logging
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import torchvision.datasets as vdatasets
import torchvision.transforms as vtransforms
import torchvision.utils as vutils

from mlutils import find_gpu, structuralize
from slim import one_hot, Flatten

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config

        def linear_block(in_size, out_size, normalize=True):
            layers = [nn.Linear(in_size, out_size)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_size, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model_noise = nn.Sequential(       # input: 100 * 1
            *linear_block(100, 100, False),     # 100 * 1
        )

        self.model_label = nn.Sequential(       # input: 10 * 1
            *linear_block(10, 100, False),      # 100 * 1
        )

        self.model = nn.Sequential(             # input: 200 * 1
            *linear_block(200, 256, True),
            *linear_block(256, 512, True),
            *linear_block(512, 1024, True),
            nn.Linear(1024, np.prod(self.config.img_shape)),
            nn.Tanh(),
        )

    # pylint: disable=arguments-differ
    def forward(self, noise, labels):
        a = self.model_noise(noise)
        b = self.model_label(labels)
        return self.model(torch.cat((a, b), dim=1)).view(-1, *self.config.img_shape)

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config

        self.model_image = nn.Sequential(
            Flatten(),
        )

        self.model_label = nn.Sequential(
            nn.Linear(10, 128, False),
        )

        self.model = nn.Sequential(         # input: 1 x 64 x 64
            nn.Linear(128 + np.prod(self.config.img_shape), 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    # pylint: disable=arguments-differ
    def forward(self, image, labels):
        a = self.model_image(image)
        b = self.model_label(labels)
        return self.model(torch.cat((a, b), dim=1))

class CGAN(object):
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
        dataset = vdatasets.MNIST(
            root=self.config.datadir,
            transform=vtransforms.Compose([
                vtransforms.Resize(64),
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
        fixed_noise = torch.rand(self.config.batch_size, self.config.g.in_channels,
                                 device=self.config.device)
        fixed_tags = torch.randint(low=0, high=self.config.nlabels,
                                   size=(self.config.batch_size, 1),
                                   dtype=torch.long)
        fixed_side_info = one_hot(fixed_tags, self.config.nlabels).to(self.config.device)

        for epoch in range(start_epoch, num_epoch + start_epoch):
            for _i, data in enumerate(self.dataset, 0):
                self.D.zero_grad()

                batch_size = data[0].size(0)

                noise = torch.rand(batch_size, self.config.g.in_channels,
                                   device=self.config.device)

                real_side_info = one_hot(data[1].view(-1, 1), self.config.nlabels).to(self.config.device)
                fake_tags = torch.randint(0, self.config.nlabels, size=(data[1].size(0), 1), dtype=torch.long)
                fake_side_info = one_hot(fake_tags, self.config.nlabels).to(self.config.device)

                real_data = data[0].to(self.config.device)
                fake_data = self.G(noise, fake_side_info)

                real_label = torch.ones((batch_size, 1), device=self.config.device)
                fake_label = torch.zeros((batch_size, 1), device=self.config.device)

                d_real_error = self.loss(self.D(real_data, real_side_info), real_label)
                d_fake_error = self.loss(self.D(fake_data.detach(), fake_side_info), fake_label)

                d_error = (d_real_error + d_fake_error) / 2
                d_error.backward()
                self.optim_d.step()

                self.G.zero_grad()

                g_fake_error = self.loss(self.D(fake_data, fake_side_info), real_label)
                g_fake_error.backward()
                self.optim_g.step()

            self.logger.info('Finish epoch %d' % epoch)

            if epoch % self.config.print_interval == 0:
                self.logger.info('epoch[%3d]: d_error: %f, g_fake_error: %f',
                                 epoch, d_error.mean(), g_fake_error.mean())
                self.G.eval()
                fixed_fake_data = self.G(fixed_noise, fixed_side_info)
                self.G.train()
                vutils.save_image(fixed_fake_data.detach(), 'cgan_fake_samples_epoch_%03d.png' % epoch, normalize=True)

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--verbose', action='store_true')

if __name__ == '__main__':
    logger = logging.getLogger('CGAN')
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
            'img_shape': (1, 64, 64),
            'learning_rate': 2e-4,
            'betas': (0.5, 0.999),
        },
        'd': {
            'img_shape': (1, 64, 64),
            'learning_rate': 2e-4,
            'betas': (0.5, 0.999),
        },
        'device': device,
        'batch_size': 64,
        'num_epoch': 20,
        'print_interval': 5,
        'verbose': opt.verbose,
        'nlabels': 10,
        'datadir': 'dataset/data/mnist',
    }

    gan = CGAN(structuralize('config', **basic_config), logger)
    gan.train()
