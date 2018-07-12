#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

import torch
import torch.nn as nn
import torch.optim as optim

from mlutils import structuralize

class Generator(nn.Module):
    def __init__(self, config):
        self.config = config
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(config.in_size, config.hidden_size),
            nn.ELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(config.hidden_size, config.out_size)
        )

    def forward(self, *x):
        return self.model(*x)

class Discriminator(nn.Module):
    def __init__(self, config):
        self.config = config
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(config.in_size, config.hidden_size),
            nn.ELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ELU(),
            nn.Linear(config.hidden_size, config.out_size),
            nn.Sigmoid()
        )

    def forward(self, *x):
        return self.model(*x)

def get_distr_sampler(mu, sigma):
    return lambda batch_size, dim: torch.randn((batch_size, dim)).mul(sigma).add(mu)

def get_generator_sampler():
    return lambda batch_size, dim: torch.rand((batch_size, dim))

class GANTrivial(object):
    def __init__(self, config):
        self.config = config
        self.G = Generator(config.g)
        self.D = Discriminator(config.d)
        self.loss = nn.BCELoss()
        self.optim_g = optim.Adam(self.G.parameters(),
                                  lr=config.g.learning_rate,
                                  betas=config.g.betas)
        self.optim_d = optim.Adam(self.D.parameters(),
                                  lr=config.d.learning_rate,
                                  betas=config.d.betas)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

    def train(self):
        d_sampler = lambda batch_size, dim: torch.randn((batch_size, dim)).mul(1.2).add(4.6)
        g_sampler = lambda batch_size, dim: torch.rand((batch_size, dim))

        for epoch in range(self.config.num_epoch):
            for _d_step in range(self.config.d_steps):
                self.D.zero_grad()

                d_real_data = d_sampler(self.config.batch_size, self.config.d.in_size)
                d_real_error = self.loss(self.D(d_real_data),
                                         torch.ones((self.config.batch_size, self.config.d.out_size)))

                d_fake_data = self.G(g_sampler(self.config.batch_size, self.config.g.in_size)).detach()
                d_fake_error = self.loss(self.D(d_fake_data),
                                         torch.zeros(self.config.batch_size, self.config.g.out_size))

                d_error = (d_real_error + d_fake_error) / 2
                d_error.backward()

                self.optim_d.step()

            for _g_step in range(self.config.g_steps):
                self.G.zero_grad()

                g_fake_data = self.G(g_sampler(self.config.batch_size, self.config.g.in_size))
                g_fake_error = self.loss(self.D(g_fake_data),
                                         torch.ones(self.config.batch_size, self.config.g.out_size))

                g_fake_error.backward()

                self.optim_g.step()

            if epoch % self.config.print_interval == 0:
                fake_data = gan.G(torch.rand((10000, 1)))
                self.logger.info('epoch[%d]: d_error: %f, g_fake_error: %f, (%f, %f)',
                                 epoch,
                                 d_error.mean(),
                                 g_fake_error.mean(),
                                 fake_data.mean(),
                                 fake_data.std())

if __name__ == '__main__':
    basic_config = {
        'g': {
            'in_size': 1,
            'hidden_size': 50,
            'out_size': 1,
            'learning_rate': 2e-3,
            'betas': (0.9, 0.999),
        },
        'd': {
            'in_size': 1,
            'hidden_size': 50,
            'out_size': 1,
            'learning_rate': 2e-3,
            'betas': (0.9, 0.999),
        },
        'd_steps': 1,
        'g_steps': 1,
        'batch_size': 100,
        'num_epoch': 1000,
        'print_interval': 100,
    }

    gan = GANTrivial(structuralize('config', **basic_config))
    gan.train()
