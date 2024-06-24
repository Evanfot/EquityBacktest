# %%
import sys

sys.path.append("..")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pyfolio as pf
from pyfolio.plotting import show_perf_stats

file_path = "../examples/20240620_2102_backtest_1.pkl"
result = pd.read_pickle(file_path)


returns, positions, transactions = pf.utils.extract_rets_pos_txn_from_zipline(result)
perf_stats = show_perf_stats(returns, positions=positions, transactions=transactions)

pf.create_full_tear_sheet(returns, positions, transactions)

# %%
