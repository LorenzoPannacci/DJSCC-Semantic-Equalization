"""This module defines the functions/classes needed for the linear optimization with full GPU support."""

import math
import torch
import numpy as np
from tqdm.auto import tqdm
from sklearn.cluster import KMeans
from scipy.linalg import solve_sylvester

from alignment.zeroshot_utils_gpu import (
        complex_compressed_tensor,
        decompress_complex_tensor,
        prewhiten,
        sigma_given_snr,
        awgn,
        a_inv_times_b,
        mmse_svd_equalizer,
        complex_gaussian_matrix,
    )


# ============================================================
#
#                     CLASSES DEFINITION
#
# ============================================================
class Baseline:
    """A linear version of the baseline in which we're not taking into account the advantages of semantic communication.

    Args:
        input_dim : int
            The input dimension.
        output_dim : int
            The output dimension.
        channel_matrix : torch.Tensor
            The channel matrix H in torch.Tensor format.
        snr : float
            The snr in dB of the communication channel. Set to None if unaware.
        proto : int
            The number of observation per class used to compute the means in PPFE. Default 10.
        channel_usage : int
            Number of packets to consider, if None then all. Default None.
        typology : str
            The typology of baseline, possible values 'pre' or 'post'. Default 'pre'.
        strategy : str
            The strategy to adopt in sending the features, possible values 'First-K', 'Top-K', 'Eigen-K', 'UPE', 'PFE' or 'PPFE'. Default 'First-K'.
        seed : int
            The seed for the kmeans. Default 42.
        device : str
            The device to use ('cpu' or 'cuda'). Default 'cuda' if available, else 'cpu'.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        channel_matrix: torch.Tensor,
        snr: float,
        proto: int = 10,
        channel_usage: int = None,
        typology: str = 'post',
        strategy: str = 'First-K',
        use_channel: bool = True,
        seed: int = 42,
        device: str = None,
    ):
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        assert len(channel_matrix.shape) == 2, (
            'The matrix must be 2 dimesional.'
        )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.channel_matrix = channel_matrix.to(self.device)
        self.snr = snr
        self.proto = proto
        self.channel_usage = channel_usage
        self.typology = typology
        self.strategy = strategy
        self.use_channel = use_channel
        self.seed = seed

        # Check values
        assert self.typology in {'pre', 'post'}, (
            f'The passed typology {self.typology} is not supported.'
        )
        assert self.strategy in {
            'First-K',
            'Top-K',
            'Eigen-K',
            'UPE',
            'PFE',
            'PPFE',
        }, f'The passed strategy {self.strategy} is not supported.'

        # Get the receiver and transmitter antennas
        self.antennas_receiver, self.antennas_transmitter = (
            self.channel_matrix.shape
        )

        # Instantiate the alignment matrix A
        self.A = None

        # Define self.channel_usage if it set to None
        if not self.channel_usage:
            self.channel_usage = math.ceil(
                (self.input_dim // 2) / self.antennas_transmitter
            )

        self.G, self.F = mmse_svd_equalizer(self.channel_matrix, self.snr)
        # Ensure G and F are on the correct device
        if self.G is not None:
            self.G = self.G.to(self.device)
        if self.F is not None:
            self.F = self.F.to(self.device)

        return None

    def __compression(self, input: torch.Tensor) -> torch.Tensor:
        """Compress the input.

        Args:
            input : torch.Tensor
                The input as real d x n.

        Return:
            input : torch.Tensor
                The compressed input as complex channel_usage * N_t x n.
        """
        # Ensure input is on correct device
        input = input.to(self.device)
        
        # Get the number of features of the input
        self.size, n = input.shape

        # Features to transmit
        self.sent_features = 2 * self.channel_usage * self.antennas_transmitter

        match self.strategy:
            case 'First-K':
                input = input[: self.sent_features, :]

            case 'Top-K':
                # Get the indexes based on the selected strategy
                _, self.indexes = torch.topk(
                    input.abs(), self.sent_features, dim=0
                )
                # Ensure indexes are on the same device
                self.indexes = self.indexes.to(self.device)

                # Retrieve the values based on the indexes
                input = input[self.indexes, torch.arange(n, device=self.device)]

            case 'Eigen-K':
                input = self.F_tilde @ input

            case 'PFE':
                input = self.F_tilde @ input

            case 'PPFE':
                input = self.F_tilde @ input

            case 'UPE':
                input = self.F_tilde @ input

            case _:
                raise Exception('The passed strategy is not supported.')

        # Complex Compression
        input = complex_compressed_tensor(input, device=self.device)

        return input

    def __decompression(self, received: torch.Tensor) -> torch.Tensor:
        """Decompression of the received message.

        Args:
            received : torch.Tensor
                The received message

        Return:
            output : torch.Tensor
                The output.
        """
        received = received.to(self.device)
        _, n = received.shape

        # Decompress the transmitted signal
        received = decompress_complex_tensor(received)

        output = torch.zeros(self.size, n, device=self.device)

        match self.strategy:
            case 'First-K':
                output[: self.sent_features, :] = received

            case 'Top-K':
                output[self.indexes, torch.arange(n, device=self.device)] = received

            case 'Eigen-K':
                output = self.G_tilde @ received

            case 'PFE':
                output = self.G_tilde @ received

            case 'PPFE':
                output = self.G_tilde @ received

            case 'UPE':
                output = self.G_tilde @ received

            case _:
                raise Exception('The passed strategy is not supported.')

        return output

    def __packets_precoding(self, input: torch.Tensor) -> list[torch.Tensor]:
        """Precoding of packets given an input.

        Args:
            input : torch.Tensor
                The input to transmit, expected as features x number of observations.

        Returns
            list[torch.Tensor]
                The list of precoded packets, each of them of dimension self.antennas_transmitter.
        """
        input = input.to(self.device)
        
        # Compress the input
        input = self.__compression(input)

        # Perform the prewhitening step
        input = a_inv_times_b(self.L, input - self.mean)

        # Create the packets of size self.antennas_transmitter
        packets = torch.split(input, self.antennas_transmitter, dim=0)

        # Return the precoded packets
        return [self.F @ p for p in packets]

    def __packets_decoding(
        self, packets: list[torch.Tensor]
    ) -> torch.Tensor:
        """Decoding the transmitted packets.

        Args:
            packets : list[torch.Tensor]
                The list of the received packets.

        Returns
            received : torch.Tensor
                The output seen as a single torch.Tensor of dimension self.output_dim x num. of observation.
        """
        # Ensure all packets are on the correct device
        packets = [p.to(self.device) for p in packets]
        
        # Decode the packets
        packets = [self.G @ p for p in packets]

        # Concat the packets
        received = torch.cat(packets, dim=0)

        # Remove whitening
        received = self.L @ received + self.mean

        # Decompress the transmitted signal
        received = self.__decompression(received)

        return received

    def __transmit_message(self, input: torch.Tensor) -> torch.Tensor:
        """Function that implements the transmission of a message.

        Args:
            input : torch.Tensor
                The input to transmit.

        Returns:
            output : torch.Tensor
                The output transmitted.
        """
        input = input.to(self.device)
        
        with torch.no_grad():
            # Performing the precoding packets
            packets = self.__packets_precoding(input)

            # Pass through the channel
            if self.use_channel:
                # Transmit and add the AWGN if needed
                if self.snr is not None:
                    # Get the sigma
                    sigma = sigma_given_snr(
                        snr=self.snr,
                        signal=torch.ones(1, device=self.device)
                        / math.sqrt(self.antennas_transmitter),
                    )
                    packets = [
                        self.channel_matrix @ p
                        + awgn(sigma=sigma, size=p.shape, device=self.device)
                        for p in packets
                    ]
                else:
                    # Transmit the packets
                    packets = [self.channel_matrix @ p for p in packets]

            # Decode the packets
            output = self.__packets_decoding(packets)

        return output.T

    def fit(self, input: torch.Tensor, output: torch.Tensor) -> None:
        """Fitting method of the linear baseline.
        This function performs the semantic alignment between input and output.

        Args:
            input : torch.Tensor
                The input to transmit.
            output : torch.Tensor
                The output to allign to.

        Returns:
            None
        """
        # Move tensors to device
        input = input.to(self.device)
        output = output.to(self.device)

        with torch.no_grad():
            # Alignment of the input to the output
            self.A = torch.linalg.lstsq(input, output).solution.T

            if self.strategy == 'Eigen-K':
                sent_size = 2 * self.channel_usage * self.antennas_transmitter
                U, S, Vt = torch.linalg.svd(self.A, full_matrices=False)
                S[sent_size:] = 0
                S = torch.sqrt(torch.diag(S))

                self.F_tilde = (S @ Vt)[:sent_size, :].to(self.device)
                self.G_tilde = (U @ S)[:, :sent_size].to(self.device)

            elif self.strategy == 'UPE':
                sent_size = 2 * self.channel_usage * self.antennas_transmitter
                U, _, Vt = torch.linalg.svd(
                    output.H @ input, full_matrices=False
                )
                self.F_tilde = Vt[:sent_size, :].to(self.device)
                self.G_tilde = U[:, :sent_size].to(self.device)

            elif self.strategy == 'PFE':
                sent_size = 2 * self.channel_usage * self.antennas_transmitter
                idx = torch.randperm(input.size(0), device=self.device)[:sent_size]

                input_subset = input[idx]
                output_subset = output[idx]

                # Encoder Frame
                U, S, Vt = torch.linalg.svd(input_subset, full_matrices=False)
                self.F_tilde = (U @ Vt).to(self.device)

                # Decoder Frame
                U, S, Vt = torch.linalg.svd(output_subset, full_matrices=False)
                self.G_tilde = (U @ Vt).H.to(self.device)

            elif self.strategy == 'PPFE':
                sent_size = 2 * self.channel_usage * self.antennas_transmitter
                # Move to CPU for sklearn, then back to GPU
                data = output.detach().cpu().numpy()
                kmeans = KMeans(n_clusters=sent_size, random_state=self.seed)
                labels = torch.tensor(kmeans.fit_predict(data), device=self.device)

                n_input = []
                n_output = []
                for label in labels.unique():
                    mask = labels == label

                    # Retrieve the obs for each label
                    inps = input[mask]
                    outs = output[mask]

                    # Get only some obs to calculate the proto
                    iidx = torch.randperm(inps.size(0), device=self.device)[: self.proto]

                    # Calculate the mean
                    n_input.append(inps[iidx].mean(dim=0))
                    n_output.append(outs[iidx].mean(dim=0))

                # Stack all of them in a tensor
                input_means = torch.stack(n_input).to(self.device)
                output_means = torch.stack(n_output).to(self.device)

                # Shuffle
                idx = torch.randperm(input_means.size(0), device=self.device)
                input_means = input_means[idx]
                output_means = output_means[idx]

                # Encoder Frame
                U, S, Vt = torch.linalg.svd(input_means, full_matrices=False)
                self.F_tilde = (U @ Vt).to(self.device)

                # Decoder Frame
                U, S, Vt = torch.linalg.svd(output_means, full_matrices=False)
                self.G_tilde = (U @ Vt).H.to(self.device)

            match self.typology:
                case 'pre':
                    if self.strategy not in {'Eigen-K', 'PFE', 'PPFE', 'UPE'}:
                        # Align the input
                        input = self.A @ input.T

                    # Compress the input
                    input = self.__compression(input.T)

                    # Learn L and the mean
                    _, self.L, self.mean = prewhiten(input, device=self.device)

                case 'post':
                    # Compress the input
                    input = self.__compression(input.T)

                    # Learn L and the mean
                    _, self.L, self.mean = prewhiten(input, device=self.device)

                case _:
                    raise Exception(
                        f'The passed typology {self.typology} is not supported.'
                    )

        return None

    def transform(self, input: torch.Tensor) -> torch.Tensor:
        """Transform the passed input.

        Args:
            input : torch.Tensor
                The input tensor.

        Returns:
            output : torch.Tensor
                The transformed version of the input.
        """
        input = input.to(self.device)

        # Transpose
        input = input.T

        match self.typology:
            case 'pre':
                if self.strategy not in {'Eigen-K', 'PFE', 'PPFE', 'UPE'}:
                    # Align the input
                    input = self.A @ input

                # Transmit the input
                output = self.__transmit_message(input)

            case 'post':
                # Transmit the input
                output = self.__transmit_message(input)

                if self.strategy not in {'Eigen-K', 'PFE', 'PPFE', 'UPE'}:
                    # Align the output
                    output = (self.A @ output.T).T

            case _:
                raise Exception(
                    f"Unrecognised case of self.typology parameter, set to '{self.typology}'."
                )

        return output

    def eval(self, input: torch.Tensor, output: torch.Tensor) -> float:
        """Eval an input given an expected output.

        Returns:
            float
                The mse loss.
        """
        input = input.to(self.device)
        output = output.to(self.device)

        # Check if self.A is fitted
        assert self.A is not None, (
            "You have to fit the solver first by calling the '.fit()' method."
        )

        # Get the predictions
        preds = self.transform(input)

        return torch.nn.functional.mse_loss(
            preds, output, reduction='mean'
        ).item()

# ============================================================
#
#                     MAIN DEFINITION
#
# ============================================================


def main() -> None:
    """Some sanity tests..."""
    print('Start performing sanity tests...')
    print()
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    print()

    # Variables definition
    cost: int = 1
    snr: float = 20
    channel_usage: int = 1
    iterations: int = 10
    input_dim: int = 384
    output_dim: int = 768
    antennas_transmitter: int = 4
    antennas_receiver: int = 4
    channel_matrix: torch.Tensor = complex_gaussian_matrix(
        mean=0, std=1, size=(antennas_receiver, antennas_transmitter)
    ).to(device)

    input = torch.randn(100, input_dim, device=device)
    output = torch.randn(100, output_dim, device=device)

    print('Test for the Baseline...', end='\t')
    baseline = Baseline(
        input_dim=input_dim,
        output_dim=output_dim,
        channel_matrix=channel_matrix,
        snr=snr,
        channel_usage=channel_usage,
        strategy='First-K',
    )
    baseline.fit(input, output)
    baseline.transform(input)
    print(baseline.eval(input, output))
    print('[Passed]')

    return None


if __name__ == '__main__':
    main()