from __future__ import print_function
import json
import os
import pandas as pd
import time

import watstock


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


def get_data(symbol, dates, usecols=['Date', 'Adj Close']):
  """Read stock data (adjusted close) for given symbols from CSV files."""
  df = pd.DataFrame(index=dates)

  df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                        parse_dates=True, usecols=usecols,
                        na_values=['nan'])
  df = df.join(df_temp)

  return df


def to_daily_returns(df):
  """Compute and return the daily return values."""
  daily_returns = df.copy()
  daily_returns[1:] = (df[1:]/df[:-1].values) # to avoid index matching
  daily_returns.ix[0, :] = 1
  return daily_returns


def runner(param_sequence):

  for params in param_sequence:

    results = watstock.run(params)

    params.pop('df', None)
    print('Params:', params)
    print('Accuracy:', results.get('test_accuracy'))

    save_prediction(results)


def main():

  symbol = 'AAPL'
  dates = pd.date_range('2012-08-26', '2016-12-05')

  # Get stock data
  df = get_data(symbol, dates, usecols=['Date', 'Adj Close'])
 
  # Add sentiment data
  df_sentiment = get_data('AOS-AAPL', dates, usecols=['Date', 'Article Sentiment', 'Impact Score'])
  df = df.join(df_sentiment)

  # Add day of year data
  # df_dayofyear = pd.to_datetime(df.index.values).dayofyear
  # dayofyear_df = pd.DataFrame(data=df_dayofyear, index=df.index.values, columns=['Day of year'])
  # df = df.join(dayofyear_df)

  # Reordering columns so Ads Close is the last
  df = df[['Article Sentiment', 'Impact Score', 'Adj Close']]

  # Drop N/a values
  df = df.dropna()

  param_sequence = [
    {
      'symbol': symbol,
      'df': df,
      'layers': [300],
      'timesteps': 120,
      'test_set': 30,
      'val_set': 30,
      'batch_size': 10,
      'epochs': 500,
      'dropout': None,
      'early_stopping_patience': 5
    },
    {
      'symbol': symbol,
      'df': df,
      'layers': [500],
      'timesteps': 120,
      'test_set': 30,
      'val_set': 30,
      'batch_size': 10,
      'epochs': 500,
      'dropout': None,
      'early_stopping_patience': 5
    },
    {
      'symbol': symbol,
      'df': df,
      'layers': [1000],
      'timesteps': 120,
      'test_set': 30,
      'val_set': 30,
      'batch_size': 10,
      'epochs': 500,
      'dropout': None,
      'early_stopping_patience': 5
    },
    # # architectures
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [50],
    #   'timesteps': 15,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [100],
    #   'timesteps': 15,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [200],
    #   'timesteps': 15,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [300],
    #   'timesteps': 15,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [500],
    #   'timesteps': 15,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [1000],
    #   'timesteps': 15,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [2000],
    #   'timesteps': 15,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },

    # # 1-50 time steps
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [50],
    #   'timesteps': 5,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [50],
    #   'timesteps': 10,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [50],
    #   'timesteps': 20,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [50],
    #   'timesteps': 30,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [50],
    #   'timesteps': 60,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [50],
    #   'timesteps': 90,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [50],
    #   'timesteps': 120,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },

    # # 1-100 time steps
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [100],
    #   'timesteps': 5,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [100],
    #   'timesteps': 10,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [100],
    #   'timesteps': 20,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [100],
    #   'timesteps': 30,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [100],
    #   'timesteps': 60,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [100],
    #   'timesteps': 90,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [100],
    #   'timesteps': 120,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },

    # # 1-300 time steps
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [300],
    #   'timesteps': 5,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [300],
    #   'timesteps': 10,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [300],
    #   'timesteps': 20,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [300],
    #   'timesteps': 30,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [300],
    #   'timesteps': 60,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [300],
    #   'timesteps': 90,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [300],
    #   'timesteps': 120,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },

    # # 1-500 time steps
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [500],
    #   'timesteps': 5,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [500],
    #   'timesteps': 10,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [500],
    #   'timesteps': 20,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [500],
    #   'timesteps': 30,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [500],
    #   'timesteps': 60,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [500],
    #   'timesteps': 90,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [500],
    #   'timesteps': 120,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },

    # # 1-1000 time steps
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [1000],
    #   'timesteps': 5,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [1000],
    #   'timesteps': 10,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [1000],
    #   'timesteps': 20,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [1000],
    #   'timesteps': 30,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [1000],
    #   'timesteps': 60,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [1000],
    #   'timesteps': 90,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },
    # {
    #   'symbol': symbol,
    #   'df': df,
    #   'layers': [1000],
    #   'timesteps': 120,
    #   'test_set': 30,
    #   'val_set': 30,
    #   'batch_size': 10,
    #   'epochs': 500,
    #   'dropout': None,
    #   'early_stopping_patience': 5
    # },
  ]

  runner(param_sequence)

if __name__ == "__main__":
  main()