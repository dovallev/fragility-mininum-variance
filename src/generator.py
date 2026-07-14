from abc import ABC, abstractmethod

import numpy as np


class InstanceGenerator(ABC):
    @abstractmethod
    def generate(self) -> tuple[np.ndarray, np.ndarray, list]:
        """Returns (covariance_matrix, return_array, asset_list)."""
        ...


class BlockCovarianceGenerator(InstanceGenerator):
    def __init__(self, block_sizes, block_ranges, off_block_range, shuffle=True, seed=None,
                 block_sigmas=None):
        self.block_sizes = block_sizes
        self.block_ranges = block_ranges
        self.off_block_range = off_block_range
        self.shuffle = shuffle
        if block_sigmas is not None:
            assert len(block_sigmas) == len(block_sizes), \
                "block_sigmas must have one entry per block"
            for s, size in zip(block_sigmas, block_sizes):
                assert len(s) == size, \
                    "each block_sigmas entry must match the corresponding block_size"
        self.block_sigmas = block_sigmas
        if seed is not None:
            np.random.seed(seed)

    def _generate_psd_block(self, size, value_range, variability=0.2):
        base_value = np.random.uniform(value_range[0], value_range[1])
        block = base_value + np.random.uniform(-variability, variability, (size, size))
        block = (block + block.T) / 2
        eigvals, eigvecs = np.linalg.eigh(block)
        eigvals = np.maximum(eigvals, 1e-3)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    def generate(self, eps=1e-3) -> tuple[np.ndarray, np.ndarray, list]:
        n = sum(self.block_sizes)
        matrix = np.random.uniform(self.off_block_range[0], self.off_block_range[1], (n, n))
        matrix = (matrix + matrix.T) / 2

        start = 0
        for size, value_range in zip(self.block_sizes, self.block_ranges):
            end = start + size
            matrix[start:end, start:end] = self._generate_psd_block(size, value_range)
            start = end

        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals = np.maximum(eigvals, eps)
        matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T

        if self.block_sigmas is not None:
            sigma = np.sqrt(np.diag(matrix))
            omega = matrix / np.outer(sigma, sigma)
            sigmas_all = np.concatenate([np.asarray(s, dtype=float) for s in self.block_sigmas])
            matrix = np.diag(sigmas_all) @ omega @ np.diag(sigmas_all)

        if self.shuffle:
            idx = np.random.permutation(n)
            matrix = matrix[np.ix_(idx, idx)]

        assert np.allclose(matrix, matrix.T, rtol=1e-5, atol=1e-8), 'Matrix is not symmetric'
        assert np.all(np.linalg.eigh(matrix)[0] > 0), 'Matrix is not PSD'

        return_array = np.random.uniform(0, 1, (n, 1))
        return matrix, return_array, None


class BlockDiagonalStructureGenerator(InstanceGenerator):
    """Covariance matrix with the paper's exact block-diagonal correlation structure.

    Builds Omega = (1-global_rho)*blockdiag(Omega_1,...,Omega_K) + global_rho*ee^T
    where each Omega_i = (1-rho_i)*I + rho_i*ee^T (constant-correlation block).

    Equivalently, entry (j,k) of Omega is:
        1                                    if j == k
        global_rho + (1-global_rho)*rho_i    if j,k belong to block i  (j != k)
        global_rho                           if j,k belong to different blocks

    After calling generate(), self.true_clusters holds 1-indexed cluster labels
    aligned with the returned covariance matrix.

    Parameters
    ----------
    block_sigmas : list of array-like
        Volatility vectors, one per block.  len(block_sigmas) == K.
    block_rhos : list of float
        Within-block constant correlation rho_i per block.
        For singleton blocks (len == 1) the value is ignored (rho_i = 0).
    global_rho : float
        Common cross-block correlation rho.
    """

    def __init__(self, block_sigmas, block_rhos, global_rho, asset_list=None):
        assert len(block_sigmas) == len(block_rhos), \
            "block_sigmas and block_rhos must have the same length"
        self.block_sigmas = [np.asarray(s, dtype=float) for s in block_sigmas]
        self.block_rhos   = list(block_rhos)
        self.global_rho   = float(global_rho)
        self._asset_list  = asset_list
        self.true_clusters = None   # set by generate()

    @property
    def n_blocks(self):
        return len(self.block_sigmas)

    @property
    def n_assets(self):
        return sum(len(s) for s in self.block_sigmas)

    def generate(self) -> tuple[np.ndarray, np.ndarray, list]:
        n      = self.n_assets
        rho    = self.global_rho
        Omega  = np.full((n, n), rho)
        np.fill_diagonal(Omega, 1.0)
        clusters = np.zeros(n, dtype=int)

        start = 0
        for block_id, (sigmas, rho_i) in enumerate(
                zip(self.block_sigmas, self.block_rhos), start=1):
            k   = len(sigmas)
            end = start + k
            if k > 1:
                within_rho = rho + (1.0 - rho) * rho_i
                Omega[start:end, start:end] = within_rho
                np.fill_diagonal(Omega[start:end, start:end], 1.0)
            clusters[start:end] = block_id
            start = end

        sigmas_all = np.concatenate(self.block_sigmas)
        V = np.diag(sigmas_all) @ Omega @ np.diag(sigmas_all)

        self.true_clusters = clusters
        return_array = sigmas_all.reshape(-1, 1)
        asset_list   = self._asset_list if self._asset_list is not None else list(range(n))
        return V, return_array, asset_list


class ConstantCorrelationGenerator(InstanceGenerator):
    """Covariance from Sigma @ Omega @ Sigma where Omega has constant off-diagonal rho."""
    def __init__(self, sigmas: list, rho: float, asset_list: list = None):
        self.sigmas = sigmas
        self.rho = rho
        self.asset_list = asset_list

    def generate(self) -> tuple[np.ndarray, np.ndarray, list]:
        n = len(self.sigmas)
        Omega = np.full((n, n), self.rho)
        np.fill_diagonal(Omega, 1.0)
        Sigma = np.diag(self.sigmas)
        covariance_matrix = Sigma @ Omega @ Sigma
        return_array = np.array(self.sigmas).reshape(-1, 1)
        return covariance_matrix, return_array, self.asset_list
