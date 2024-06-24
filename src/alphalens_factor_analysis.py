# %%
#!/Users/evanfotopoulos/Projects/EquityBacktest/.venv/bin/python

# %%
import sys

sys.path.append("..")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from zipline.pipeline.classifiers import Classifier
from zipline.utils.numpy_utils import int64_dtype
from zipline.data import bundles
from zipline.pipeline.factors import AverageDollarVolume
from zipline.pipeline import Pipeline

# data
from zipline.utils.calendar_utils import get_calendar
from data.pricing import (
    ingest_function,
    build_pipeline_engine,
    get_data_portal,
    get_pricing,
)
from zipline.pipeline.domain import ZA_EQUITIES
from data.sector import dict_asset_sector, dict_sector_int, create_sector

# alphas
from alpha_models.alphas import (
    momentum_sector_neutral,
    mean_reversion_sector_neutral,
    mean_reversion_sector_neutral_smoothed,
    annualised_volatility_sector_neutral,
)

# analysis
import alphalens as al
from analytics.stats import sharpe_ratio

import yaml

# %% data

with open("../config.yml", "r") as file:
    settings = yaml.safe_load(file)
config_heading = "insample"
EOD_BUNDLE_NAME = settings[config_heading]["bundles"]["bundle_name"]
factor_start_date = pd.Timestamp(settings[config_heading]["backtest"]["start_date"])
factor_end_date = pd.Timestamp(settings[config_heading]["backtest"]["end_date"])
calendar_code = settings[config_heading]["bundles"]["calendar_code"]
domain = ZA_EQUITIES
ingest_func = ingest_function(EOD_BUNDLE_NAME)

bundles.register(EOD_BUNDLE_NAME, ingest_func)
bundle_data = bundles.load(EOD_BUNDLE_NAME)
engine = build_pipeline_engine(bundle_data, domain)

universe = AverageDollarVolume(window_length=120).top(100)

trading_calendar = get_calendar(calendar_code)

data_portal = get_data_portal(bundle_data, trading_calendar)

# Create sector for bundle
sector_tickers = bundle_data.asset_finder.retrieve_all(bundle_data.asset_finder.sids)
sector, ticker_sector = create_sector(
    sector_tickers, dict_asset_sector, dict_sector_int
)

# %%

# ## Create alpha factors
pipeline = Pipeline(screen=universe)
pipeline.add(
    momentum_sector_neutral(251, universe, sector), "momentum_sector_neutral_1YR"
)
pipeline.add(
    mean_reversion_sector_neutral(19, universe, sector),
    "Mean_Reversion_1Mo_Sector_Neutral",
)
pipeline.add(annualised_volatility_sector_neutral(252, universe, sector), "Volatility")
pipeline.add(
    mean_reversion_sector_neutral(5, universe, sector),
    "Mean_Reversion_5Day_Sector_Neutral",
)
pipeline.add(
    mean_reversion_sector_neutral_smoothed(5, universe, sector),
    "Mean_Reversion_5Day_Sector_Neutral_Smoothed",
)
all_factors = engine.run_pipeline(pipeline, factor_start_date, factor_end_date)
all_factors["alpha"] = (
    (all_factors["momentum_sector_neutral_1YR"])
    .add(all_factors["Mean_Reversion_1Mo_Sector_Neutral"])
    .add(all_factors["Mean_Reversion_5Day_Sector_Neutral"] * 0.5)
)

all_factors.index = all_factors.index.set_levels(
    [pd.to_datetime(all_factors.index.levels[0]), all_factors.index.levels[1]]
)

# %%

assets = all_factors.index.levels[1].values.tolist()
pricing = get_pricing(
    data_portal, trading_calendar, assets, factor_start_date, factor_end_date
)

# %%
sector_names = dict((v, k) for k, v in dict_sector_int.items())

clean_factor_data = {
    factor: al.utils.get_clean_factor_and_forward_returns(
        factor=factor_data,
        prices=pricing,
        periods=[1, 5],
        groupby=ticker_sector,
        groupby_labels=sector_names,
    )
    for factor, factor_data in all_factors.items()
}

unixt_factor_data = {
    factor: factor_data.set_index(
        pd.MultiIndex.from_tuples(
            [(x.timestamp(), y) for x, y in factor_data.index.values],
            names=["date", "asset"],
        )
    )
    for factor, factor_data in clean_factor_data.items()
}

# %% Quantile Analysis

ls_factor_returns = pd.DataFrame()

for factor, factor_data in clean_factor_data.items():
    ls_factor_returns[factor] = al.performance.factor_returns(factor_data).iloc[:, 0]

(1 + ls_factor_returns).cumprod().plot()

# %%  Basis points per day per quantile
qr_factor_returns = pd.DataFrame()

for factor, factor_data in unixt_factor_data.items():
    qr_factor_returns[factor] = al.performance.mean_return_by_quantile(factor_data)[
        0
    ].iloc[:, 0]

(10000 * qr_factor_returns).plot.bar(
    subplots=True, sharey=True, layout=(4, 2), figsize=(14, 14), legend=False
)

# %% Turnover Analysis
ls_FRA = pd.DataFrame()

for factor, factor_data in clean_factor_data.items():
    ls_FRA[factor] = al.performance.factor_rank_autocorrelation(factor_data)

ls_FRA.plot(title="Factor Rank Autocorrelation")
plt.show()

# %% Sharpe ratio

daily_annualization_factor = np.sqrt(252)
sharpe = sharpe_ratio(ls_factor_returns, daily_annualization_factor).round(2)
print("Sharpe Ratio: {}".format(sharpe))

# %%
