"""
Watstock data module.
"""

from __future__ import print_function
import os
import quandl
import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd

quandl.ApiConfig.api_key = os.environ['QUANDL_API_KEY']

def get_stock_data(ticker, start_date=None, end_date=None):
  data = quandl.get('WIKI/%s' % ticker, start_date=start_date, end_date=end_date)  
  return data[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Volume', 'Adj. Close']]

def get_aos_data(ticker, start_date=None, end_date=None):
  data = quandl.get('AOS/%s' % ticker, start_date=start_date, end_date=end_date)

  return data

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

# print(get_aos_data('AAPL'))
