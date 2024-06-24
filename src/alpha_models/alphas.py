# ###
# %% Momentum 1 year

from zipline.pipeline.factors import Returns, CustomFactor
import numpy as np


def momentum_sector_neutral(window_length, universe, sector):

    return (
        Returns(window_length=window_length, mask=universe)
        .demean(groupby=sector)
        .rank()
        .zscore()
    )


# %% ### Mean Reversion 5 Day Sector Neutral


def mean_reversion_sector_neutral(window_length, universe, sector):
    """
    Generate a mean reversion sector neutral factor

    Parameters
    ----------
    window_length : int
        Returns window length
    universe : Zipline Filter
        Universe of stocks filter
    sector : Zipline Classifier
        Sector classifier

    Returns
    -------
    factor : Zipline Factor
        Mean reversion sector neutral factor
    """
    return momentum_sector_neutral(window_length, universe, sector) * -1


# ## Mean Reversion Sector Neutral Smoothed Factor

# %%
from zipline.pipeline.factors import SimpleMovingAverage


def mean_reversion_sector_neutral_smoothed(window_length, universe, sector):
    """
    Generate the mean reversion sector neutral smoothed factor

    Parameters
    ----------
    window_length : int
        Returns window length
    universe : Zipline Filter
        Universe of stocks filter
    sector : Zipline Classifier
        Sector classifier

    Returns
    -------
    factor : Zipline Factor
        Mean reversion sector neutral smoothed factor
    """

    return (
        SimpleMovingAverage(
            inputs=[mean_reversion_sector_neutral(window_length, universe, sector)],
            window_length=window_length,
        )
        .rank()
        .zscore()
    )


class AnnualisedVolatility(CustomFactor):
    """
    Volatility. The degree of variation of a series over time as measured by
    the standard deviation of daily returns.
    https://en.wikipedia.org/wiki/Volatility_(finance)

    **Default Inputs:** [Returns(window_length=2)]

    Parameters
    ----------
    annualization_factor : float, optional
        The number of time units per year. Defaults is 252, the number of NYSE
        trading days in a normal year.
    """

    inputs = [Returns(window_length=2)]
    params = {"annualization_factor": 252.0}
    window_length = 252

    def compute(self, today, assets, out, returns, annualization_factor):
        out[:] = np.nanstd(returns, axis=0) * (annualization_factor**0.5)


def annualised_volatility_sector_neutral(window_length, universe, sector):
    """
    Generate a volatility sector neutral factor

    Parameters
    ----------
    window_length : int
        Returns window length
    universe : Zipline Filter
        Universe of stocks filter
    sector : Zipline Classifier
        Sector classifier

    Returns
    -------
    factor : Zipline Factor
        volatility sector neutral factor
    """
    return -1 * (
        AnnualisedVolatility(mask=universe).demean(groupby=sector).rank().zscore()
    )


# %%
