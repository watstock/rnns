"""
Watstock data module.
"""

from __future__ import print_function
import os
import quandl
import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd

def get_daily_stock_data(ticker, start_date=None, end_date=None):
  quandl.ApiConfig.api_key = os.environ['QUANDL_API_KEY']

  data = quandl.get('WIKI/%s' % ticker, start_date=start_date, end_date=end_date)  
  return data[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Volume', 'Adj. Close']]

def get_aos_data(ticker, start_date=None, end_date=None):
  quandl.ApiConfig.api_key = os.environ['QUANDL_API_KEY']

  data = quandl.get('AOS/%s' % ticker, start_date=start_date, end_date=end_date)
  return data

def get_nasdaq100_one_min_data(ticker):
  quandl.ApiConfig.api_key = os.environ['QUANDL_API_KEY2']

  quandl.Database("ASN100").bulk_download_to_file("./ASN100.zip")

def get_rolling_mean(df, window=20):
  df = df.sort_index()

  data = df.rolling(window=window).mean()
  rm_df = pd.DataFrame(data=data.values, index=df.index.values, columns=['Rolling Mean'])

  return rm_df

def get_rolling_std(df, window=20):
  df = df.sort_index()

  data = df.rolling(window=window).std()
  rstd_df = pd.DataFrame(data=data.values, index=df.index.values, columns=['Rolling Std'])

  return rstd_df

# end_date = datetime.datetime.now()
# start_date = end_date - relativedelta(years=10)

# print(get_nasdaq100_one_min_data('AAPL'))
