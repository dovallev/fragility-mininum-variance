from abc import ABC, abstractmethod

import numpy as np


class CovarianceEstimator(ABC):
    @abstractmethod
    def fit(self, returns: np.ndarray) -> np.ndarray:
        """Estimate covariance matrix from a (T x n) returns array."""
        ...


class SampleCovarianceEstimator(CovarianceEstimator):
    """Sample (empirical) covariance estimator.

    Computes the standard unbiased sample covariance matrix:

        S = (1 / (T-1)) * X_c' X_c

    where X_c is the mean-centred returns matrix. This is the maximum-likelihood
    estimator under a Gaussian assumption and is unbiased regardless of the
    distribution.
    """

    def fit(self, returns: np.ndarray) -> np.ndarray:
        return np.cov(returns, rowvar=False)
