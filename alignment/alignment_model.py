# -*- coding: utf-8 -*-
"""
@author: LorenzoPannacci
"""

import torch.nn as nn
from channel import Channel
import torch

class _LinearAlignment(nn.Module):
    def __init__(self, size=None, align_matrix=None):
        super(_LinearAlignment, self).__init__()

        if align_matrix is not None:
            self.align_matrix = nn.Parameter(align_matrix)

        else:
            self.align_matrix = nn.Parameter(torch.empty(size, size))
            nn.init.xavier_uniform_(self.align_matrix)

    def forward(self, x):
        # get shape of input
        shape = x.shape

        # flatten input
        x = x.flatten(start_dim=1)

        # apply alignment
        x = x @ self.align_matrix

        # return to original shape
        return x.reshape(shape)


class _ConvolutionalAlignment(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(_ConvolutionalAlignment, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)

    def forward(self, x):
        return self.conv(x)


class _ZeroShotAlignment(nn.Module):
    def __init__(self, F_tilde, G_tilde):
        super(_ZeroShotAlignment, self).__init__()

        self.F_tilde = nn.Parameter(F_tilde.T)
        self.G_tilde = nn.Parameter(G_tilde.T)

    def compression(self, input):
        return input @ self.F_tilde
    
    def decompression(self, input):
        return input @ self.G_tilde


class AlignedDeepJSCC(nn.Module):
    def __init__(self, encoder, decoder, aligner, snr, channel_type):
        super(AlignedDeepJSCC, self).__init__()

        # get encoder from model1
        self.encoder = encoder
        
        self.snr = snr

        if self.snr is not None:
            self.channel = Channel(channel_type, snr)
        else:
            self.channel = None

        # get aligner
        self.aligner = aligner

        # get decoder from model2
        self.decoder = decoder

        # get forward function
        if type(self.aligner) == _ZeroShotAlignment:
            self.forward = self._forward_zeroshot
        
        else:
            self.forward = self._forward_default

    def _forward_default(self, x):
        z = self.encoder(x)

        if self.channel is not None:
            z = self.channel(z)

        if self.aligner is not None:
            z = self.aligner(z)
        
        x_hat = self.decoder(z)
        return x_hat

    def _forward_zeroshot(self, x):
        z = self.encoder(x)

        # get shape of input
        shape = z.shape

        # flatten input
        if z.dim() == 4:

            z = z.flatten(start_dim=1)
        
        elif z.dim() == 3:
            z = z.flatten(start_dim=0)
            z = z.reshape(1, -1)

        # zeroshot compression
        z = self.aligner.compression(z)

        if self.channel is not None:
            z = self.channel(z)

        # zeroshot decompression
        z = self.aligner.decompression(z)

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