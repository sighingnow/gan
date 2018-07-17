#! /usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Least Squares GAN.

Reference:
[1]: `Least Squares Generative Adversarial Networks, ICCV 2017
     <http://openaccess.thecvf.com/content_ICCV_2017/papers/Mao_Least_Squares_Generative_ICCV_2017_paper.pdf>`_
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

class Flatten(nn.Module):
    # pylint: disable=arguments-differ
    def forward(self, x):
        return x.view(x.size(0), -1)

class Reshape(nn.Module):
    def __init__(self, *size):
        super(Reshape, self).__init__()
        self.shape = size
    # pylint: disable=arguments-differ
    def forward(self, x):
        return x.view(x.size(0), *self.shape)

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        self.model = nn.Sequential(                         # input: config.in_channels
            nn.Linear(config.in_channels, 256 * 7 * 7),     # 256 x 7 x 7
            Reshape(256, 7, 7),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 3, 2, bias=False), # 256 x 15 x 15
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 3, 1, bias=False), # 256 x 17 x 17
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 3, 2, bias=False), # 256 x 35 x 35
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 3, 1, bias=False), # 256 x 37 x 37
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, 2, bias=False), # 128 x 75 x 75
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, bias=False),  # 64 x 151 x 151
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 3, 1, bias=False),    # 1 x 153 x 153
            nn.Tanh(),
        )

    def forward(self, *x):
        return self.model(*x)

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        self.model = nn.Sequential(         # input: 1 x 153 x 153
            nn.Conv2d(1, 64,
                      5, 2, 0, bias=False), # 64 x 75 x 75
            nn.Conv2d(64, 128,
                      5, 2, 0, bias=False), # 128 x 36 x 36
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256,
                      5, 2, 0, bias=False), # 256 x 16 x 16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512,
                      5, 2, 0, bias=False), # 512 x 6 x 6
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            Flatten(),                      # (512 x 6 x 6)
            nn.Linear(512 * 6 * 6, 1),      # 1
        )

    def forward(self, *x):
        return self.model(*x)

class LSGAN(object):
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

        self.loss = nn.MSELoss()

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
                vtransforms.Resize(153),
                vtransforms.CenterCrop(153),
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
        fixed_noise = torch.randn(self.config.batch_size, self.config.g.in_channels,
                                  device=self.config.device)

        for epoch in range(start_epoch, num_epoch + start_epoch):
            for _i, data in enumerate(self.dataset, 0):
                self.D.zero_grad()

                real_data = data[0].to(self.config.device)
                batch_size = real_data.size(0)
                noise = torch.randn(batch_size, self.config.g.in_channels,
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

                g_fake_error = self.loss(self.D(fake_data), real_label) / 2
                g_fake_error.backward()
                self.optim_g.step()

            self.logger.info('Finish epoch %d' % epoch)

            if epoch % self.config.print_interval == 0:
                self.logger.info('epoch[%3d]: d_error: %f, g_fake_error: %f',
                                 epoch, d_error.mean(), g_fake_error.mean())
                self.G.eval()
                fixed_fake_data = self.G(fixed_noise)
                self.G.train()
                vutils.save_image(fixed_fake_data.detach(), 'lsgan_fake_samples_epoch_%03d.png' % epoch, normalize=True)

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--verbose', action='store_true')

if __name__ == '__main__':
    logger = logging.getLogger('LSGAN')
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
        'num_epoch': 10,
        'print_interval': 5,
        'verbose': opt.verbose,
        'datadir': 'dataset/data/mnist',
    }

    gan = LSGAN(structuralize('config', **basic_config), logger)
    gan.train()
