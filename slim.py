#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class Flatten(nn.Module):
    '''Flatten layer'''
    # pylint: disable=arguments-differ
    def forward(self, x):
        return x.view(x.size(0), -1)

class Reshape(nn.Module):
    '''Reshape layer'''
    def __init__(self, *size):
        super(Reshape, self).__init__()
        self.shape = size
    # pylint: disable=arguments-differ
    def forward(self, x):
        return x.view(x.size(0), *self.shape)

class OneHot(nn.Module):
    '''One-hot encode layer'''
    def __init__(self, nlabels):
        super(OneHot, self).__init__()
        self.nlabels = nlabels
    # pylint: disable=arguments-differ
    def forward(self, labels):
        return one_hot(labels, self.nlabels)

def one_hot(labels, nlabels):
    '''One-hot encoding, accepts a 2-D tensor as input, the first dimension is mini-batch.

       For example:

            >>> one_hot(3, torch.tensor([2,1]).view(-1, 1))
            tensor([[ 0.,  0.,  1.],
                    [ 0.,  1.,  0.]])
    '''
    assert len(labels.size()) == 2 and labels.size(1) == 1, "The one_hot exepcts a 2-D tensor as input"
    mask = torch.zeros(labels.size(0), nlabels)
    return mask.scatter_(1, labels, torch.ones(labels.size()))
