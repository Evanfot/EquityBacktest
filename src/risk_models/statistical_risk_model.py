from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

#  PCA Risk Model
# Steps:
# -Run PCA on returns series
# -Create series for factors
# -Find covariance matrix of these factors (sigmai,j will be 0)
# -Find betas of returns to factors
# -Calculate idiosyncratic returns for each ticker = variance of residuals after
#   (residual = return - return predicted by betas to factors)


def fit_pca(returns, num_PC_exposures, svd_solver):
    """
    Fit PCA model with returns.

    Parameters
    ----------
    returns : DataFrame
        Returns for each ticker and date
    num_PC_exposures : int
        Number of PCs for PCA
    svd_solver: str
        The solver to use for the PCA model

    Returns
    -------
    pca : PCA
        Model fit to returns
    """
    pca = PCA(n_components=num_PC_exposures, svd_solver=svd_solver)
    pca.fit(returns)
    return pca


def factor_betas(pca, factor_beta_indices, factor_beta_columns):
    """
    Get the factor betas from the PCA model.

    Parameters
    ----------
    pca : PCA
        Model fit to returns
    factor_beta_indices : 1 dimensional Ndarray
        factor beta indices
    factor_beta_columns : 1 dimensional Ndarray
        factor beta columns

    Returns
    -------
    factor_betas : DataFrame
        factor betas
    """
    assert len(factor_beta_indices.shape) == 1
    assert len(factor_beta_columns.shape) == 1
    return pd.DataFrame(pca.components_.T, factor_beta_indices, factor_beta_columns)


def factor_returns(pca, returns, factor_return_indices, factor_return_columns):
    """
    Get the factor returns from the PCA model.

    Parameters
    ----------
    pca : PCA
        Model fit to returns
    returns : DataFrame
        Returns for each ticker and date
    factor_return_indices : 1 dimensional Ndarray
        factor return indices
    factor_return_columns : 1 dimensional Ndarray
        factor return columns

    Returns
    -------
    factor_returns : DataFrame
        factor returns
    """
    assert len(factor_return_indices.shape) == 1
    assert len(factor_return_columns.shape) == 1
    return pd.DataFrame(
        pca.transform(returns.values), factor_return_indices, factor_return_columns
    )


def factor_cov_matrix(factor_returns, ann_factor):
    """
    Get the factor covariance matrix

    Parameters
    ----------
    factor_returns : DataFrame
        factor returns
    ann_factor : int
        Annualization factor

    Returns
    -------
    factor_cov_matrix : DataFrame
        factor covariance matrix
    """
    return np.diag(factor_returns.var(axis=0, ddof=1) * ann_factor)


def idiosyncratic_var_matrix(returns, factor_returns, factor_betas, ann_factor):
    """
    Get the idiosyncratic variance matrix

    Parameters
    ----------
    returns : DataFrame
        Returns for each ticker and date
    factor_returns : DataFrame
        factor returns
    factor_betas : DataFrame
        factor betas
    ann_factor : int
        Annualization factor

    Returns
    -------
    idiosyncratic_var_matrix : DataFrame
        Idiosyncratic variance matrix
    """
    common_returns_ = pd.DataFrame(
        np.dot(factor_returns, factor_betas.T), returns.index, returns.columns
    )

    residuals_ = returns - common_returns_
    return pd.DataFrame(
        np.diag(np.var(residuals_, axis=0)) * ann_factor,
        returns.columns,
        returns.columns,
    )


def idiosyncratic_var_vector(returns, idiosyncratic_var_matrix):
    """
    Get the idiosyncratic variance vector

    Parameters
    ----------
    returns : DataFrame
        Returns for each ticker and date
    idiosyncratic_var_matrix : DataFrame
        Idiosyncratic variance matrix

    Returns
    -------
    idiosyncratic_var_vector : DataFrame
        Idiosyncratic variance Vector
    """
    return pd.DataFrame(np.diagonal(idiosyncratic_var_matrix), returns.columns)
