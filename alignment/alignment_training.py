# -*- coding: utf-8 -*-
"""
@author: LorenzoPannacci
"""

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import random

from torch.utils.data import Dataset, DataLoader, Subset
from alignment.alignment_model import _ConvolutionalAlignment, _LinearAlignment, _ZeroShotAlignment
from alignment.linear_models_gpu import Baseline

from model import DeepJSCC
from tqdm import tqdm
from channel import Channel

class AlignmentDataset(Dataset):
    def __init__(self, dataloader, model1, model2, device, flat=False):
        self.outputs = []

        model1.eval()
        model1.to(device)

        model2.eval()
        model2.to(device)

        with torch.no_grad():
            for inputs, _ in tqdm(dataloader, desc="Computing model outputs"):
                inputs = inputs.to(device)

                out1 = model1(inputs)
                out2 = model2(inputs)

                for o1, o2 in zip(out1, out2):
                    if flat:
                        o1 = o1.flatten()
                        o2 = o2.flatten()

                    self.outputs.append((o1.cpu(), o2.cpu()))

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, idx):
        return self.outputs[idx]  


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_from_checkpoint(path, snr, c, device):
    state_dict = torch.load(path, map_location=device)
    from collections import OrderedDict
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k.replace('module.','') # remove `module.`
        new_state_dict[name] = v

    model = DeepJSCC(c=c, channel_type="AWGN", snr=snr)

    model.load_state_dict(new_state_dict)
    model.change_channel("AWGN", snr)

    return model


def dataset_to_matrices(dataset, batch_size=128):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    data_1 = []
    data_2 = []
    
    for batch in loader:
        data_1.append(batch[0])
        data_2.append(batch[1])

    return torch.cat(data_1, dim=0), torch.cat(data_2, dim=0)


def aligner_least_squares(matrix_1, matrix_2):
    Y = matrix_1.T
    Z = matrix_2.T

    Q = Y @ Z.T @ torch.inverse(Z @ Z.T)

    return _LinearAlignment(align_matrix=Q)


def aligner_least_squares(matrix_1, matrix_2, n_samples):
    Y = matrix_1.T  # [d, n]
    Z = matrix_2.T  # [d, n]

    ZZ_T = Z @ Z.T
    YZ_T = Y @ Z.T

    reg_matrix = (10000) * torch.eye(ZZ_T.size(0), device=ZZ_T.device, dtype=ZZ_T.dtype)
    Q = YZ_T @ torch.linalg.inv(ZZ_T + reg_matrix)

    return _LinearAlignment(align_matrix=Q)


def load_alignment_dataset(model1_fp, model2_fp, train_snr, train_loader, device, flat=True):
    model1 = load_from_checkpoint(model1_fp, train_snr).encoder
    model2 = load_from_checkpoint(model2_fp, train_snr).encoder

    return AlignmentDataset(train_loader, model1, model2, device, flat)


def train_linear_aligner(data, permutation, n_samples):

    indices = permutation[:n_samples]
    subset = Subset(data, indices)

    matrix_1, matrix_2 = dataset_to_matrices(subset)

    return aligner_least_squares(matrix_1, matrix_2, n_samples)

def train_neural_aligner(data, permutation, n_samples, batch_size, resolution, ratio, train_snr, device):
    # train settings
    epochs_max = 10000
    patience = 10
    min_delta = 1e-5
    lambda_reg = 0.001

    # prepare data
    indices = permutation[:n_samples]
    subset = Subset(data, indices)
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True)

    # prepare model and optimizer
    aligner = _LinearAlignment(size=resolution * resolution * 3 * 2 // ratio).to(device)
    channel = Channel("AWGN", train_snr)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(aligner.parameters(), lr=1e-3, weight_decay=lambda_reg)

    # init train state
    best_loss = float('inf')
    best_model_state = None
    checks_without_improvement = 0
    epoch = 0

    # train loop
    while True:
        epoch_loss = 0.0

        for inputs, targets in dataloader:
            
            if train_snr is not None:
                inputs = channel(inputs)

            optimizer.zero_grad()
            outputs = aligner(inputs.to(device))

            mse_loss = criterion(outputs, targets.to(device))
            loss = inputs.shape[0] * mse_loss

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch += 1

        # check if improvement
        avg_loss = epoch_loss / len(dataloader)
        if best_loss - avg_loss > min_delta:
            best_loss = avg_loss
            best_model_state = copy.deepcopy(aligner.state_dict())
            checks_without_improvement = 0
        else:
            checks_without_improvement += 1

        # break if patience exceeded
        if checks_without_improvement >= patience:
            break

        # break if max epochs exceeded
        if epoch > epochs_max:
            break

    # restore best model
    if best_model_state is not None:
        aligner.load_state_dict(best_model_state)

    return aligner, epoch

def train_conv_aligner(data, permutation, n_samples, c, batch_size, train_snr, device):
    # train settings
    epochs_max=10000
    patience=10
    min_delta=1e-5
    reg_val = 0.001

    # prepare data
    indices = permutation[:n_samples]
    subset = Subset(data, indices)
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True)

    # prepare model and optimizer
    aligner = _ConvolutionalAlignment(in_channels=2*c, out_channels=2*c, kernel_size=5).to(device)
    channel = Channel("AWGN", train_snr)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(aligner.parameters(), lr=1e-3, weight_decay=reg_val)

    # init train state
    best_loss = float('inf')
    best_model_state = None
    checks_without_improvement = 0
    epoch = 0

    # train loop
    while True:
        epoch_loss = 0.0

        for inputs, targets in dataloader:

            if train_snr is not None:
                inputs = channel(inputs)

            optimizer.zero_grad()
            outputs = aligner(inputs.to(device))
            loss = criterion(outputs, targets.to(device))
            loss = loss * inputs.shape[0] # scale by batch size
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch += 1

        # check if improvement
        avg_loss = epoch_loss / len(dataloader)
        if best_loss - avg_loss > min_delta:
            best_loss = avg_loss
            best_model_state = copy.deepcopy(aligner.state_dict())
            checks_without_improvement = 0
        else:
            checks_without_improvement += 1

        # break if patience exceeded
        if checks_without_improvement >= patience:
            break

        # break if max epochs exceeded
        if epoch > epochs_max:
            break

    # restore best model
    if best_model_state is not None:
        aligner.load_state_dict(best_model_state)
    
    return aligner, epoch

def train_zeroshot_aligner(data, permutation, n_samples, resolution, train_snr, seed):
    # prepare data
    indices = permutation[:n_samples]
    subset = Subset(data, indices)
    dataloader = DataLoader(subset, batch_size=len(subset))
    input, output = next(iter(dataloader))

    flattened_image_size = resolution**2

    # init
    baseline = Baseline(
        input_dim=flattened_image_size,
        output_dim=flattened_image_size,
        channel_matrix=torch.eye(1, dtype=torch.complex64),
        snr=train_snr,
        channel_usage=None,
        typology='pre',
        strategy='PFE',
        use_channel=True if train_snr is not None else False,
        seed=seed,
    )

    # fit
    baseline.fit(input, output)

    # convert
    return _ZeroShotAlignment(baseline.F_tilde, baseline.G_tilde)