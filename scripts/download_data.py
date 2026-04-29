#!/usr/bin/env python
# coding: utf-8


from openbb import obb
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import numpy as np
import pyarrow
from pathlib import Path

load_dotenv()
obb.user.credentials.fmp_api_key = os.getenv('FMP_API_KEY')
obb.user.credentials.fred_api_key = os.getenv('FRED_API_KEY')

startdate = '1990-01-01' #extra data for ~63 trading days buffer
enddate = '2025-12-30'
SPX = obb.equity.price.historical(symbol = '^GSPC', start_date = startdate, end_date = enddate, interval = '1d', provider = 'yfinance').to_df()['close']
VIX = obb.equity.price.historical(symbol = '^VIX', start_date = startdate, end_date = enddate, interval = '1d', provider = 'yfinance').to_df()['close']
T_3M = obb.economy.fred_series(symbol = 'DGS3MO', start_date= startdate, enddate = enddate).to_df()
T_10Y = obb.economy.fred_series(symbol = 'DGS10', start_date= startdate, enddate = enddate).to_dataframe()
CS = obb.economy.fred_series(symbol = 'BAA10Y', start_date= startdate, enddate = enddate).to_df()
Growth = obb.economy.fred_series(symbol = 'INDPRO', start_date= startdate, enddate = enddate).to_df()
Inflation = obb.economy.fred_series(symbol = 'CPIAUCSL', start_date= startdate, enddate = enddate).to_df()
PR = obb.economy.fred_series(symbol = 'PAYEMS', start_date= startdate, enddate = enddate).to_df()
M2 = obb.economy.fred_series(symbol = 'M2SL', start_date= startdate, enddate = enddate).to_df()

# Derived Datasets
SPX_log = np.log(SPX/SPX.shift(1))
SPX_7d_vol = SPX_log.rolling(7).std() * np.sqrt(252) * 100
SPX_21d_vol = SPX_log.rolling(21).std() * np.sqrt(252) * 100
SPX_63d_vol = SPX_log.rolling(63).std() * np.sqrt(252) * 100
SPX_7d_ret = SPX_log.rolling(7).sum() * 100
SPX_21d_ret = SPX_log.rolling(21).sum()  * 100
SPX_63d_ret = SPX_log.rolling(63).sum()  * 100
Vol_Ratio = SPX_7d_vol/SPX_63d_vol
IV_RV = VIX - SPX_21d_vol
Curve_10y1m = T_10Y['DGS10'] - T_3M['DGS3MO']

#Target Variable
SPX_21d_vol_fw = SPX_21d_vol.shift(-21)

#Macro Variables
Growth = Growth.set_axis(pd.to_datetime(Growth.index)).sort_index().shift(1).resample('D').ffill().reindex(SPX.index, method = 'ffill')
Inflation = Growth.set_axis(pd.to_datetime(Growth.index)).sort_index().shift(1).resample('D').ffill().reindex(SPX.index, method = 'ffill')
PR = Growth.set_axis(pd.to_datetime(Growth.index)).sort_index().shift(1).resample('D').ffill().reindex(SPX.index, method = 'ffill')
M2 = Growth.set_axis(pd.to_datetime(Growth.index)).sort_index().shift(1).resample('D').ffill().reindex(SPX.index, method = 'ffill')

# Join
keys = [
    'SPX_21d_vol_fw',
    'SPX',
    'VIX',
    'T_3M',
    'T_10Y',
    'Credit_Spread',
    'Growth',
    'Inflation',
    'Payrolls',
    'M2',
    'vol_7d',
    'vol_21d',
    'vol_63d',
    'ret_7d',
    'ret_21d',
    'ret_63d',
    'vol_ratio',
    'iv_rv',
    '10y1m'
]
df = pd.concat(
    [SPX_21d_vol_fw, SPX, VIX, T_3M, T_10Y, CS, Growth, Inflation, PR, M2, SPX_7d_vol, SPX_21d_vol, SPX_63d_vol, SPX_7d_ret, SPX_21d_ret,SPX_63d_ret, Vol_Ratio, IV_RV, Curve_10y1m], 
    keys = keys, axis = 1)
df.columns = df.columns.droplevel(1)


# Filter Index - Ignore first 3 months due to 63 trailing, and ignore last 3 months for same reason.
df.index = pd.to_datetime(df.index)
df = df[(df.index > '1990-04-01') & (df.index < '2026-01-01')].dropna()

# Check for NA's
df.isna().sum()

# Output to data folder
OUT_DIR = Path("../data")
OUT_DIR.mkdir(parents = True, exist_ok= True)
df.to_csv(OUT_DIR/"processed.csv", index = True)
df.to_parquet(OUT_DIR/"processed.parquet", engine = 'pyarrow', index = True)

