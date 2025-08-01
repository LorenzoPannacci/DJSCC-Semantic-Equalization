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

from torch.utils.data import Dataset, DataLoader
from alignment.alignment_model import _ConvolutionalAlignment, _LinearAlignment, _ZeroShotAlignment, _MLPAlignment

from model import DeepJSCC
from tqdm import tqdm
from channel import Channel

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from dataset import Vanilla

def get_data_loaders(dataset, resolution, batch_size, num_workers):
    if dataset == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((resolution, resolution))])

        train_dataset = datasets.CIFAR10(root='../dataset/', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)

        test_dataset = datasets.CIFAR10(root='../dataset/', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    elif dataset == 'imagenet':
        # the size of paper is 128
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((resolution, resolution))])

        print("loading data of imagenet")

        train_dataset = datasets.ImageFolder(root='./dataset/ImageNet/train', transform=transform)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)

        test_dataset = Vanilla(root='./dataset/ImageNet/val', transform=transform)
        test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    elif dataset == 'imagenette':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((resolution, resolution))])

        train_dataset = datasets.Imagenette(root='../dataset/', split="train", download=True, transform=transform)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)

        test_dataset = datasets.Imagenette(root='../dataset/', split="val", download=True, transform=transform)
        test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    else:
        raise Exception('Unknown dataset')
    
    return train_loader, test_loader


class AlignmentDataset(Dataset):
    def __init__(self, dataloader, model1, model2, device, flat=False):
        self.inputs = []
        self.model1 = model1.to(device).eval()
        self.model2 = model2.to(device).eval()
        self.device = device
        self.flat = flat

        for batch, _ in tqdm(dataloader, desc="Caching inputs"):
            self.inputs.extend(batch)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx].unsqueeze(0).to(self.device)  # add batch dim
        with torch.no_grad():
            out1 = self.model1(x)
            out2 = self.model2(x)
        if self.flat:
            out1 = out1.flatten()
            out2 = out2.flatten()
        return out1, out2


class AlignmentSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

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


def train_linear_aligner(data, permutation, n_samples, train_snr, regularization=10000):
    """
    Solve least squares problem with regularization.
    """

    # prepare data
    indices = permutation[:n_samples]
    subset = AlignmentSubset(data, indices)
    matrix_1, matrix_2 = dataset_to_matrices(subset)

    X = matrix_1.H
    Y = matrix_2.H

    # noise handling
    snr_linear = 10 ** (train_snr / 10)
    sigma2 = 1.0 / snr_linear # noise variance
    noise_cov = sigma2 * torch.eye(X.shape[0], device=X.device, dtype=X.dtype)

    # regularization
    reg_matrix = regularization * torch.eye(X.shape[0], device=X.device, dtype=X.dtype)

    F = Y @ X.H @ torch.linalg.inv(X @ X.H + noise_cov + reg_matrix)

    return _LinearAlignment(align_matrix=F.T).cpu()


def train_neural_aligner(data, permutation, n_samples, batch_size, resolution, ratio, train_snr, device):
    """
    Train linear aligner with Adam optimization.
    """

    # train settings
    epochs_max = 1000
    patience = 10
    min_delta = 1e-5
    lambda_reg = 0.001

    # prepare data
    indices = permutation[:n_samples]
    subset = AlignmentSubset(data, indices)
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

    return aligner.cpu(), epoch


def train_mlp_aligner(data, permutation, n_samples, batch_size, resolution, ratio, train_snr, device):
    """
    Train MLP aligner with Adam optimization.
    """

    # train settings
    epochs_max = 1000
    patience = 10
    min_delta = 1e-5
    lambda_reg = 0.01

    # prepare data
    indices = permutation[:n_samples]
    subset = AlignmentSubset(data, indices)
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True)

    # prepare model and optimizer
    size = resolution * resolution * 3 * 2 // ratio
    aligner = _MLPAlignment(size, [size]).to(device)
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

    return aligner.cpu(), epoch


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
    subset = AlignmentSubset(data, indices)
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
    
    return aligner.cpu(), epoch


def train_zeroshot_aligner(data, permutation, n_samples, train_snr, channel_usage, device):
    """
    Initializes zeroshot aligner.
    """
    
    # prepare data
    indices = permutation[:n_samples]
    subset = AlignmentSubset(data, indices)
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
    return _ZeroShotAlignment(F_tilde, G_tilde, G, L, mean).cpu()