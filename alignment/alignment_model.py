# -*- coding: utf-8 -*-
"""
@author: LorenzoPannacci
"""

import torch.nn as nn
from channel import Channel

class _LinearAlignment(nn.Module):
    def __init__(self, align_matrix):
        super(_LinearAlignment, self).__init__()

        self.align_matrix = nn.Parameter(align_matrix)

    def forward(self, x):
        # get shape of input
        shape = x.shape

        # flatten input
        x = x.flatten(start_dim=1)

        # apply alignment
        x = x @ self.align_matrix

        # return to original shape
        return x.reshape(shape)

class AlignedDeepJSCC(nn.Module):
    def __init__(self, model1, model2, aligner):
        super(AlignedDeepJSCC, self).__init__()

        # get encoder from model1
        self.encoder = model1.encoder
        
        self.snr = model1.snr

        if self.snr is not None:
            self.channel = model1.channel

        # get aligner
        self.aligner = aligner

        # get decoder from model2
        self.decoder = model2.decoder

    def forward(self, x):
        z = self.encoder(x)

        if self.channel is not None:
            z = self.channel(z)

        if self.aligner is not None:
            z = self.aligner(z)
        
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