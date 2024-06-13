# %%
import sys

sys.path.append("..")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pyfolio as pf

file_path = "/Users/evanfotopoulos/Projects/EquityBacktest/analysis/test/20240530_1400_backtest.pkl"
result = pd.read_pickle(file_path)


returns, positions, transactions = pf.utils.extract_rets_pos_txn_from_zipline(result)
pf.create_full_tear_sheet(returns, positions, transactions)

# %%

test = pf.create_simple_tear_sheet(returns, positions, transactions)
# %%
from pyfolio.plotting import show_perf_stats

test = show_perf_stats(returns, positions=positions, transactions=transactions)
# %%
