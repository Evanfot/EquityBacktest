# EquityBacktest
Factor backtesting using zipline

### Install CVXPY on Mac M1:
Install Xcode from the app store
- brew install cmake
- brew install openblas (not clear in hindsight if this step was needed)

### in terminal
- pip install setuptools
- xcode-select --install
- pip install ecos=2.0.5
- pip install clarabel
- pip install cvxpy
- pip install alphalens-reloaded
- pip install pyfolio-reloaded

### install zipline
- brew install ta-lib
- brew install hdf5
- pip install zipline-reloaded


### package patches
Patch the start date in the trading calendar: (read about here: https://pypi.org/project/zipline-norgatedata/#patch-to-allow-backtesting-before-20-years-ago)

in username>\miniconda3\envs\zip310\Lib\site-packages\exchange_calendars 
change from 

GLOBAL_DEFAULT_START = pd.Timestamp.now().floor("D") - pd.DateOffset(years=20)

to 

GLOBAL_DEFAULT_START = pd.Timestamp('1970-01-01')
