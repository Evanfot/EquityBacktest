# ###
# %% Momentum 1 year

from zipline.pipeline.factors import Returns


def momentum_sector_neutral(window_length, universe, sector):
    return (
        Returns(window_length=window_length, mask=universe)
        .demean(groupby=sector)
        .rank()
        .zscore()
    )


# %% ### Mean Reversion 5 Day Sector Neutral


def mean_reversion_5day_sector_neutral(window_length, universe, sector):
    """
    Generate the mean reversion 5 day sector neutral factor

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
        Mean reversion 5 day sector neutral factor
    """
    return momentum_sector_neutral(window_length, universe, sector) * -1


# ## Mean Reversion 5 Day Sector Neutral Smoothed Factor

# %%
from zipline.pipeline.factors import SimpleMovingAverage


def mean_reversion_5day_sector_neutral_smoothed(window_length, universe, sector):
    """
    Generate the mean reversion 5 day sector neutral smoothed factor

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
        Mean reversion 5 day sector neutral smoothed factor
    """

    return (
        SimpleMovingAverage(
            inputs=[
                mean_reversion_5day_sector_neutral(window_length, universe, sector)
            ],
            window_length=window_length,
        )
        .rank()
        .zscore()
    )
