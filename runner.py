from __future__ import print_function
import json
import os
import pandas as pd
import time
import datetime

import model


def plot_data(df):

  import matplotlib.pyplot as plt

  df.plot()
  plt.show()


def save_prediction(data):

  # serialize
  json_data = json.dumps(data, sort_keys=True, indent=4, separators=(',', ': '))

  # build file name
  current_time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
  output_name = 'results/%s_%s.json' % (data['symbol'], current_time)

  with open(output_name, 'w') as file:
    file.write(json_data)


def symbol_to_path(symbol, base_dir="data"):
  """Return CSV file path given ticker symbol."""
  return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbol, dates=None, usecols=['Date', 'Adj Close'], index_col='Date', date_parser=None):
  """Read stock data (adjusted close) for given symbols from CSV files."""
  
  df = pd.read_csv(symbol_to_path(symbol), 
                   index_col=index_col, usecols=usecols,
                   parse_dates=True, infer_datetime_format=True,
                   date_parser=date_parser,
                   na_values=['nan'])
  
  if dates is not None:
    df_index = pd.DataFrame(index=dates)
    df = df_index.join(df)

  return df


def to_daily_returns(df):
  """Compute and return the daily return values."""
  daily_returns = df.copy()
  daily_returns[1:] = (df[1:]/df[:-1].values) # to avoid index matching
  daily_returns.ix[0, :] = 1
  return daily_returns


def runner(param_sequence):

  for params in param_sequence:

    results = model.run(params)

    # adjust params for printing
    df = params.get('df')
    params['features'] = df.columns.values.tolist()
    params['samples'] = df.shape[0]
    params.pop('df', None)

    print('Params:', params)
    print('Train Accuracy:', results.get('train_accuracy'))
    print('Test Accuracy:', results.get('test_accuracy'))

    save_prediction(results)

def date_from_timestamp(timestamp_str):
    return datetime.datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%SZ').date()

def add_vtex_data(df, symbol):
  df_vtex = get_data('squawkrbot_daily',
    usecols=['SYMBOL', 'TIMESTAMP_UTC', 'BULL_MINUS_BEAR'], 
    index_col='TIMESTAMP_UTC', date_parser=date_from_timestamp)
  
  # rename index
  df_vtex.index.names = ['Date']

  # filter by symbol
  df_vtex = df_vtex[df_vtex['SYMBOL'] == symbol]
  df_vtex = df_vtex[['BULL_MINUS_BEAR']]

  df_vtex = df_vtex.join(df)
  return df_vtex

def add_aos_data(df, symbol):
  df_sentiment = get_data('AOS-%s' % symbol, usecols=['Date', 'Article Sentiment', 'Impact Score'])
  
  df_sentiment = df_sentiment.join(df)
  return df_sentiment

def add_day_of_year_data(df):
  df_dayofyear = pd.to_datetime(df.index.values).dayofyear
  dayofyear_df = pd.DataFrame(data=df_dayofyear, index=df.index.values, columns=['Day of year'])
  
  dayofyear_df = dayofyear_df.join(df)
  return dayofyear_df

def main():

  symbol = 'AAPL'
  # dates = pd.date_range('2016-09-28', '2016-12-04')

  # Get stock data
  df = get_data(symbol, usecols=['Date', 'Adj Close'])
 
  # Add VTEX data
  df = add_vtex_data(df, symbol)

  # Add sentiment data
  df = add_aos_data(df, symbol)

  # Drop N/a values
  df = df.dropna()
  print(df.shape)
  print(df)
  return

  param_sequence = [
    {
      'symbol': symbol,
      'df': df,
      'layers': [20],
      'timesteps': 3,
      'test_set': 10,
      'val_set': 5,
      'batch_size': 1,
      'epochs': 500,
      'dropout': None,
      'early_stopping_patience': 5
    }
  ]

  runner(param_sequence)

if __name__ == "__main__":
  main()