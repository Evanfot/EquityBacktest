import numpy as np


def predict_portfolio_risk(
    factor_betas, factor_cov_matrix, idiosyncratic_var_matrix, weights
):
    """
    Get the predicted portfolio risk

    Formula for predicted portfolio risk is sqrt(X.T(BFB.T + S)X) where:
      X is the portfolio weights
      B is the factor betas
      F is the factor covariance matrix
      S is the idiosyncratic variance matrix

    Parameters
    ----------
    factor_betas : DataFrame
        factor betas
    factor_cov_matrix : 2 dimensional Ndarray
        factor covariance matrix
    idiosyncratic_var_matrix : DataFrame
        Idiosyncratic variance matrix
    weights : DataFrame
        Portfolio weights

    Returns
    -------
    predicted_portfolio_risk : float
        Predicted portfolio risk
    """
    assert len(factor_cov_matrix.shape) == 2
    return np.sqrt(
        weights.T.dot(
            factor_betas.dot(factor_cov_matrix.dot(factor_betas.T))
            + idiosyncratic_var_matrix
        ).dot(weights)
    ).iloc[0][0]


# %%
from scipy.stats import pearsonr


def transfer_coefficient(alpha_vector, optimized_weights):

    transfer_coefficient, pvalue = pearsonr(alpha_vector, optimized_weights)
    return transfer_coefficient


def sharpe_ratio(factor_returns, annualization_factor):
    """
    Get the sharpe ratio for each factor for the entire period

    Parameters
    ----------
    factor_returns : DataFrame
        Factor returns for each factor and date
    annualization_factor: float
        Annualization Factor

    Returns
    -------
    sharpe_ratio : Pandas Series of floats
        Sharpe ratio
    """
    return annualization_factor * factor_returns.mean() / factor_returns.std()
