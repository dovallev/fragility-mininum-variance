import numpy as np


class PortfolioValidator:

    @staticmethod
    def validate_covariance_matrix(matrix, check_psd=True):
        assert isinstance(matrix, np.ndarray), 'Covariance matrix should be a numpy array'
        assert len(matrix.shape) == 2, 'Covariance matrix should be a 2-dimensional numpy array'
        assert matrix.shape[0] == matrix.shape[1], 'Covariance matrix should be square'
        if check_psd:
            assert np.allclose(matrix, matrix.T, rtol=1e-5, atol=1e-8), 'Covariance matrix is not symmetric'
            assert np.all(np.linalg.eigh(matrix)[0] >= -1e-8), 'Covariance matrix is not PSD'

    @staticmethod
    def validate_return_array(return_array, num_assets):
        assert return_array is None or isinstance(return_array, np.ndarray), 'Return array must be an array'
        if return_array is None:
            return np.ones((num_assets, 1))
        return return_array.reshape(-1, 1)

    @staticmethod
    def validate_asset_list(asset_list, num_assets):
        assert asset_list is None or isinstance(asset_list, list), 'Asset list must be a list'
        if asset_list is None:
            return list(range(num_assets))
        assert len(asset_list) == num_assets, 'Length of asset list must match the number of assets'
        return asset_list
