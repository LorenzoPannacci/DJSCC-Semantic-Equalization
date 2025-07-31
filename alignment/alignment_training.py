# -*- coding: utf-8 -*-
"""
@author: LorenzoPannacci

In this file are defined functions and classes for training of aligners.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import random

from torch.utils.data import Dataset, DataLoader, Subset
from alignment.alignment_model import _ConvolutionalAlignment, _LinearAlignment, _ZeroShotAlignment

from model import DeepJSCC
from tqdm import tqdm
from channel import Channel

class AlignmentDataset(Dataset):
    """
    Dataset class for alignment. Samples are representations of the same image
    in the latent space of two different DeepJSCC models.
    """

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


def load_alignment_dataset(model1_fp, model2_fp, train_snr, train_loader, c, device, flat=True):
    model1 = load_from_checkpoint(model1_fp, train_snr, c, device).encoder
    model2 = load_from_checkpoint(model2_fp, train_snr, c, device).encoder

    return AlignmentDataset(train_loader, model1, model2, device, flat)


def set_seed(seed):
    """
    Sets all seeds for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_from_checkpoint(path, snr, c, device):
    """
    Load DeepJSCC from .pkl checkpoint.
    """

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
    """
    Convert dataset to matrices for Least Squares optimization.
    """

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    data_1 = []
    data_2 = []
    
    for batch in loader:
        data_1.append(batch[0])
        data_2.append(batch[1])

    return torch.cat(data_1, dim=0), torch.cat(data_2, dim=0)

def aligner_least_squares(matrix_1, matrix_2, regularization):
    """
    Solve least squares problem with regularization.
    """

    Y = matrix_1.T
    Z = matrix_2.T

    ZZ_T = Z @ Z.T
    YZ_T = Y @ Z.T

    reg_matrix = regularization * torch.eye(ZZ_T.size(0), device=ZZ_T.device, dtype=ZZ_T.dtype)
    Q = YZ_T @ torch.linalg.inv(ZZ_T + reg_matrix)

    return _LinearAlignment(align_matrix=Q)


def train_linear_aligner(data, permutation, n_samples):
    """
    Train linear aligner with least squares.
    """

    # train settings
    regularization = 10000

    # prepare data
    indices = permutation[:n_samples]
    subset = Subset(data, indices)
    matrix_1, matrix_2 = dataset_to_matrices(subset)

    return aligner_least_squares(matrix_1, matrix_2, regularization)

def train_neural_aligner(data, permutation, n_samples, batch_size, resolution, ratio, train_snr, device):
    """
    Train linear aligner with Adam optimization.
    """

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
    """
    Train convolutional aligner with Adam optimization.
    """

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

def train_zeroshot_aligner(data, permutation, n_samples, train_snr, channel_usage, device):
    """
    Initializes zeroshot aligner.
    """
    
    # prepare data
    indices = permutation[:n_samples]
    subset = Subset(data, indices)
    dataloader = DataLoader(subset, batch_size=len(subset))
    input, output = next(iter(dataloader))
    input = input.to(device)
    output = output.to(device)

    # gets F_tilde and G_tilde

    idx = torch.randperm(input.size(0), device=device)[:channel_usage]
    input_subset = input[idx]
    output_subset = output[idx]

    U, _, Vt = torch.linalg.svd(input_subset, full_matrices=False)
    F_tilde = (U @ Vt).to(device)

    U, _, Vt = torch.linalg.svd(output_subset, full_matrices=False)
    G_tilde = (U @ Vt).H.to(device)

    # gets L and mean

    input = F_tilde @ input.T
    C = torch.cov(input)

    try:
        L = torch.linalg.cholesky(C)

    except RuntimeError as e:
        if 'cholesky' in str(e).lower() or 'positive definite' in str(e).lower():

            for eps in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0, 1e+1, 1e+2, 1e+3]:
                try:
                    reg = eps * torch.eye(C.shape[0] if len(C.shape) > 0 else 1, device=device, dtype=C.dtype)
                    L = torch.linalg.cholesky(C + reg)
                    break
                except RuntimeError as e_inner:
                    if 'cholesky' in str(e_inner).lower() or 'positive definite' in str(e_inner).lower():
                        continue
                    else:
                        raise e_inner
            else:
                raise RuntimeError("Cholesky failed even after increasing regularization.")

        elif 'The input tensor A must have at least 2 dimensions' in str(e):
            L = torch.sqrt(C).unsqueeze(0).unsqueeze(1)

        else:
            raise e

    mean = input.mean(axis=1, keepdim=True)

    # gets G (and F but its constant)

    if train_snr is not None:
        reg = (1.0 / (10 ** (train_snr / 10)))
        G = torch.linalg.inv(torch.Tensor([1 + reg]).unsqueeze(0))

    else:
        G = torch.Tensor([1]).unsqueeze(0)

    # build and returns aligner
    return _ZeroShotAlignment(F_tilde, G_tilde, G, L, mean)