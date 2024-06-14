#!/Users/evanfotopoulos/Projects/EquityBacktest/.venv/bin/python
# %% Imports
import sys

sys.path.append("..")
import pandas as pd
import cvxpy as cp
import numpy as np
import datetime as dt
from matplotlib import pyplot as plt
import yaml

from zipline import run_algorithm
from zipline.api import (
    attach_pipeline,
    record,
    order_target_percent,
    order_target,
    symbol,
    schedule_function,
    date_rules,
    time_rules,
    pipeline_output,
)
from zipline.pipeline.domain import ZA_EQUITIES
from six import viewkeys

from data.sector import dict_asset_sector, dict_sector_int, create_sector
from data.pricing import (
    ingest_function,
)
from optimisers.strict_factor_optimiser import OptimalHoldingsStrictFactor
from optimisers.markowitz_optimiser import OptimalHoldings
from alpha_models.alphas import (
    momentum_sector_neutral,
    mean_reversion_5day_sector_neutral,
)
from risk_models.statistical_risk_model import (
    fit_pca,
    factor_betas,
    factor_returns,
    factor_cov_matrix,
    idiosyncratic_var_matrix,
    idiosyncratic_var_vector,
)

from optimisers.strict_factor_optimiser import OptimalHoldingsStrictFactor
from zipline.pipeline import Pipeline
from zipline.finance import commission, slippage
from zipline.pipeline.factors import AverageDollarVolume


from zipline.data import bundles
from zipline.utils.calendar_utils import get_calendar

import pyfolio as pf

# %% Read config
with open("../config.yml", "r") as file:
    settings = yaml.safe_load(file)

EOD_BUNDLE_NAME = settings["bundles"]["bundle_name"]
path = settings["paths"]["output_path"]
calendar_code = settings["bundles"]["calendar_code"]
start = pd.Timestamp(settings["backtest"]["start_date"])
end = pd.Timestamp(settings["backtest"]["end_date"])
starting_cap = settings["backtest"]["starting_capital"]
run_time = dt.datetime.now().strftime("%Y%m%d_%H%M")


def create_output_path(
    file_name: str, path: str = path, run_time: str = run_time
) -> str:
    return "".join([path, run_time, "_", file_name])


domain_za = ZA_EQUITIES
calendar = get_calendar(calendar_code)


ingest_func = ingest_function(EOD_BUNDLE_NAME)
bundles.register(EOD_BUNDLE_NAME, ingest_func)
bundle_data = bundles.load(EOD_BUNDLE_NAME)


sector_tickers = bundle_data.asset_finder.retrieve_all(bundle_data.asset_finder.sids)
sector, ticker_sector = create_sector(
    sector_tickers, dict_asset_sector, dict_sector_int
)


# %%
def make_pipeline():
    universe = AverageDollarVolume(window_length=120).top(100)
    momentum_sector_neutral_1yr = momentum_sector_neutral(251, universe, sector)
    momentum_sector_neutral_1m = momentum_sector_neutral(19, universe, sector)
    mean_reversion_5_day_sector_neutral = mean_reversion_5day_sector_neutral(
        5, universe, sector
    )

    return Pipeline(
        columns={
            "universe": universe,
            "momentum_sector_neutral_1y": momentum_sector_neutral_1yr,
            "momentum_sector_neutral_1m": momentum_sector_neutral_1m,
            "Mean_Reversion_5Day_Sector_Neutral": mean_reversion_5_day_sector_neutral,
        },
        domain=domain_za,
    )


def initialize(context):
    attach_pipeline(make_pipeline(), "my_pipeline")
    # Rebalance each day.  In daily mode, this is equivalent to putting
    # `rebalance` in our handle_data, but in minute mode, it's equivalent to
    # running at the start of the day each day.
    schedule_function(
        rebalance, date_rule=date_rules.every_day(), time_rule=time_rules.market_close()
    )

    # Explicitly set the commission/slippage to the "old" value until we can
    # rebuild example data.
    context.set_commission(commission.PerShare(cost=0.0075, min_trade_cost=1))
    context.set_slippage(slippage.VolumeShareSlippage())


def before_trading_start(context, data):
    """
    Called every day before market open.
    """


def calculate_risk(Close):
    five_year_returns = Close.pct_change()[1:].fillna(0)
    num_factor_exposures = 4
    pca = fit_pca(five_year_returns, num_factor_exposures, "full")
    risk_model = {}

    risk_model["factor_betas"] = factor_betas(
        pca, five_year_returns.columns.values, np.arange(num_factor_exposures)
    )

    risk_model["factor_returns"] = factor_returns(
        pca, five_year_returns, five_year_returns.index, np.arange(num_factor_exposures)
    )

    ann_factor = 252
    risk_model["factor_cov_matrix"] = pd.DataFrame(
        factor_cov_matrix(risk_model["factor_returns"], ann_factor)
    )
    risk_model["idiosyncratic_var_matrix"] = idiosyncratic_var_matrix(
        five_year_returns,
        risk_model["factor_returns"],
        risk_model["factor_betas"],
        ann_factor,
    )
    risk_model["idiosyncratic_var_vector"] = idiosyncratic_var_vector(
        five_year_returns, risk_model["idiosyncratic_var_matrix"]
    )
    return risk_model


def calculate_optimal(pipeline, risk_model: dict):
    alpha_vector = pipeline

    optimal_weights = OptimalHoldingsStrictFactor(
        weights_max=0.1,
        weights_min=-0.1,
        risk_cap=0.9,
        factor_max=0.2,
        factor_min=-0.2,
    ).find(
        alpha_vector.fillna(0),
        risk_model["factor_betas"],
        risk_model["factor_cov_matrix"],
        risk_model["idiosyncratic_var_vector"],
    )
    return optimal_weights


def rebalance(context, data):
    print(context.datetime)
    if len(pipeline_output("my_pipeline")) != 0:
        context.output = pipeline_output("my_pipeline")
        pipeline_data = context.output.copy(deep=True)
        pipeline_data.loc[pipeline_data["universe"], ("alpha")] = (
            pipeline_data["momentum_sector_neutral_1y"]
            .subtract(pipeline_data["momentum_sector_neutral_1m"])
            .add(pipeline_data["Mean_Reversion_5Day_Sector_Neutral"] * 0.5)
        )
        pipeline_data = pipeline_data[pipeline_data["universe"]]
        all_assets = pipeline_data.index
        hist = data.history(all_assets, "close", 1250, "1d").dropna(how="all")
        risk_model = calculate_risk(hist)
        optimal_weights = calculate_optimal(pipeline_data[["alpha"]], risk_model)
        record(universe_size=len(all_assets))

        for asset in optimal_weights.index:
            if asset not in context.get_open_orders():
                order_target_percent(asset, optimal_weights.loc[asset].item())

        # Remove any assets that should no longer be in our portfolio.
        positions = context.portfolio.positions
        for asset in viewkeys(positions) - set(optimal_weights.index):
            # This will fail if the asset was removed from our portfolio because it
            # was delisted.
            print("---".join([asset.symbol, "no longer in universe"]))
            if data.can_trade(asset):
                if asset not in context.get_open_orders():
                    order_target(asset, 0)
    else:
        print("No pipeline skipping day")


# %%

## PricingLoader packages
from zipline.assets._assets import Equity  # Required for EquityPricing
from zipline.pipeline.data import EquityPricing
from zipline.data.fx import ExplodingFXRateReader
from zipline.pipeline.loaders import USEquityPricingLoader, EquityPricingLoader


class PricingLoader(object):
    def __init__(self, bundle_data):
        self.loader = EquityPricingLoader(
            bundle_data.equity_daily_bar_reader,
            bundle_data.adjustment_reader,
            ExplodingFXRateReader(),
        )

    def get_loader(self, column):
        # TODO: Fix exception handling below
        # if column not in EquityPricing.columns:
        # raise Exception('Column not in EquityPricing')
        return self.loader

    def get(self, column):
        # if column not in EquityPricing.columns:
        # raise Exception('Column not in EquityPricing')
        return self.loader


pricing_loader = PricingLoader(bundle_data)

result = run_algorithm(
    start=start,  # Set start
    end=end,  # Set end
    initialize=initialize,  # Define startup function
    capital_base=starting_cap,  # Set initial capital
    data_frequency="daily",  # Set data frequency
    bundle=EOD_BUNDLE_NAME,
    trading_calendar=calendar,
    custom_loader=pricing_loader,
)

result.to_pickle(create_output_path("backtest_1.pkl"))
print("Ready to analyze result")


# %%
# Create a benchmark dataframe
def create_benchmark(fname):
    bench = pd.read_csv(
        "{}.csv".format(fname),
        index_col="Date",
        parse_dates=True,
    )
    bench_series = pd.Series(bench["return"].values, index=bench.index)
    bench_series.rename(fname, inplace=True)
    return bench_series


bench_series = create_benchmark("../STX40")


# %%
result.index = result.index.normalize()  # to set the time to 00:00:00
bench_series = bench_series[
    bench_series.index.isin(result.index.tz_localize(None))
].tz_localize("UTC")

returns, positions, transactions = pf.utils.extract_rets_pos_txn_from_zipline(result)

returns_table = pf.create_simple_tear_sheet(returns, benchmark_rets=bench_series)
returns_table.to_csv(create_output_path("returns_table.csv"))
plt.savefig(create_output_path("returns_tear_sheet.png"), bbox_inches="tight")
plt.close()

# %%
