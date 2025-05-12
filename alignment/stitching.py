# -*- coding: utf-8 -*-
"""
@author: LorenzoPannacci
"""

import torch
import torch.nn as nn
import pickle
from channel import Channel
from model import DeepJSCC, _Encoder, _Decoder

class StitchedDeepJSCC(nn.Module):
    def __init__(self, model1, model2):
        super(StitchedDeepJSCC, self).__init__()

        # get encoder from model1
        self.encoder = model1.encoder
        self.snr = model1.snr

        if self.snr is not None:
            self.channel = model1.channel

        # get decoder from model2
        self.decoder = model2.decoder

    def forward(self, x):
        z = self.encoder(x)
        if hasattr(self, 'channel') and self.channel is not None:
            z = self.channel(z)
        x_hat = self.decoder(z)
        return x_hat

    def change_channel(self, channel_type='AWGN', snr=None):
        if snr is None:
            self.channel = None
        else:
            self.channel = Channel(channel_type, snr)

    def get_channel(self):
        if hasattr(self, 'channel') and self.channel is not None:
            return self.channel.get_channel()
        return None

    def loss(self, prd, gt):
        criterion = nn.MSELoss(reduction='mean')
        loss = criterion(prd, gt)
        return loss

class AlignedDeepJSCC(nn.Module):
    def __init__(self, model1, model2, align_matrix):
        super(AlignedDeepJSCC, self).__init__()

        # get encoder from model1
        self.encoder = model1.encoder
        self.snr = model1.snr

        if self.snr is not None:
            self.channel = model1.channel

        # get aligner
        self.align_matrix = align_matrix

        # get decoder from model2
        self.decoder = model2.decoder

    def forward(self, x):
        z = self.encoder(x)
        if hasattr(self, 'channel') and self.channel is not None:
            z = self.channel(z)

        shape = z.shape
        z = z.flatten()
        z = z @ self.align_matrix
        z = z.reshape(shape)
        
        x_hat = self.decoder(z)
        return x_hat

    def change_channel(self, channel_type='AWGN', snr=None):
        if snr is None:
            self.channel = None
        else:
            self.channel = Channel(channel_type, snr)

    def get_channel(self):
        if hasattr(self, 'channel') and self.channel is not None:
            return self.channel.get_channel()
        return None

    def loss(self, prd, gt):
        criterion = nn.MSELoss(reduction='mean')
        loss = criterion(prd, gt)
        return loss