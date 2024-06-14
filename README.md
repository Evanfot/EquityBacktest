# EquityBacktest
Factor backtesting using zipline


### Data description:
- This code does not illustrate the data cleaning and preparation process required before backtesting
- The price data used is adjusted for splits, consolidations, dividends, unbundlings, ticker changes etc. and is survivorship bias free (i.e. includes delistings).
- The data adjustments are applied to share prices. That is, prices will be total return indices rebased to the latest day in the data  
- Prior to running this code, equity data must be imported into a zipline bundle.

### Code description:
- *src.alphalens_factor_analysis* uses alphalens to perform a quick alpha factor analysis. The results do not reflect costs or any portfolio risk limits/constraints/optimisation
- *src.backtester* performs a full backtest that includes costs, a statistical risk model portfolio optimisation. The optimisation function is minimise the L1 norm of the difference between stock weights and stock alphas subject to meeting stock weight, factor loading and portfolio risk constraints.
- *src.backtest_analyser* generates charts and statistics that summarise a backtest. Any other backtest analysis should also be performed here
- The examples folder includes ipynb files that illustrate how each of these are run and the outputs that each generate.

## Installation steps:
- Create a new virtual environment
### In terminal
#### Install CVXPY on Mac M1:
- Install Xcode from the app store
- brew install cmake
- brew install openblas
- pip install setuptools
- xcode-select --install
- pip install ecos=2.0.5
- pip install clarabel
- pip install cvxpy
#### Install zipline packages
- brew install ta-lib
- brew install hdf5
- pip install alphalens-reloaded
- pip install pyfolio-reloaded
- pip install zipline-reloaded

## package patch required
Patch the start date in the trading calendar: (read about here: https://pypi.org/project/zipline-norgatedata/#patch-to-allow-backtesting-before-20-years-ago)

in env\Lib\site-packages\exchange_calendars 
change ***FROM*** *GLOBAL_DEFAULT_START = pd.Timestamp.now().floor("D") - pd.DateOffset(years=20)* ***TO*** *GLOBAL_DEFAULT_START = pd.Timestamp('1970-01-01')*