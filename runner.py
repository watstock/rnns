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

def save_prediction_to_db(data):

  from pymongo import MongoClient
  
  MOGON_CONNECTION = os.environ['MONGO_CONNECTION']
  client = MongoClient(MONGO_CONNECTION)
  
  db = client.watstock
  tests = db.tests

  test = data
  test['date'] = datetime.datetime.utcnow()

  test_id = tests.insert_one(test).inserted_id
  print('Test data saved to the db:', test_id)

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

def date_from_timestamp(timestamp_str):
    return datetime.datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%SZ').date()

def get_stock_data(symbol, dates=None):  
  df_stock = get_data(symbol, dates=dates, usecols=['Date', 'Volume', 'Adj Close'])

  return df_stock

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

def add_rolling_mean(df, window=20):
  df = df.sort_index()
  df = df.dropna()

  data = df['Adj Close'].rolling(window=window).mean()
  rm_df = pd.DataFrame(data=data.values, index=df.index.values, columns=['Rolling Mean'])

  rm_df = rm_df.join(df)
  return rm_df

def add_rolling_std(df, window=20):
  df = df.sort_index()
  df = df.dropna()

  data = df['Adj Close'].rolling(window=window).std()
  rstd_df = pd.DataFrame(data=data.values, index=df.index.values, columns=['Rolling Std'])

  rstd_df = rstd_df.join(df)
  return rstd_df

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

def main():

  symbol = 'TSLA'
  dates = pd.date_range('2006-12-05', '2016-12-05')

  # Get stock data: Volume, Adj Close
  df = get_stock_data(symbol, dates=dates)

  # Add rolling mean
  df = add_rolling_mean(df, window=3)

  # Add rolling std
  df = add_rolling_std(df, window=3)

  # Add VTEX data: BULL_MINUS_BEAR
  df = add_vtex_data(df, symbol)

  # Add sentiment data: Article Sentiment, Impact Score
  df = add_aos_data(df, symbol)

  # Drop N/a values
  df = df.dropna()

  param_sequence = [
    # 3 time steps
    {
      'symbol': symbol,
      'df': df,
      'layers': [30],
      'timesteps': 3,
      'test_set': 10,
      'val_set': 5,
      'batch_size': 1,
      'epochs': 500,
      'dropout': None,
      'early_stopping_patience': 5
    },
    {
      'symbol': symbol,
      'df': df,
      'layers': [50],
      'timesteps': 3,
      'test_set': 10,
      'val_set': 5,
      'batch_size': 1,
      'epochs': 500,
      'dropout': None,
      'early_stopping_patience': 5
    },
    {
      'symbol': symbol,
      'df': df,
      'layers': [100],
      'timesteps': 3,
      'test_set': 10,
      'val_set': 5,
      'batch_size': 1,
      'epochs': 500,
      'dropout': None,
      'early_stopping_patience': 5
    },
    {
      'symbol': symbol,
      'df': df,
      'layers': [150],
      'timesteps': 3,
      'test_set': 10,
      'val_set': 5,
      'batch_size': 1,
      'epochs': 500,
      'dropout': None,
      'early_stopping_patience': 5
    },
    {
      'symbol': symbol,
      'df': df,
      'layers': [200],
      'timesteps': 3,
      'test_set': 10,
      'val_set': 5,
      'batch_size': 1,
      'epochs': 500,
      'dropout': None,
      'early_stopping_patience': 5
    },
    {
      'symbol': symbol,
      'df': df,
      'layers': [300],
      'timesteps': 3,
      'test_set': 10,
      'val_set': 5,
      'batch_size': 1,
      'epochs': 500,
      'dropout': None,
      'early_stopping_patience': 5
    },
    {
      'symbol': symbol,
      'df': df,
      'layers': [500],
      'timesteps': 3,
      'test_set': 10,
      'val_set': 5,
      'batch_size': 1,
      'epochs': 500,
      'dropout': None,
      'early_stopping_patience': 5
    },
    {
      'symbol': symbol,
      'df': df,
      'layers': [1000],
      'timesteps': 3,
      'test_set': 10,
      'val_set': 5,
      'batch_size': 1,
      'epochs': 500,
      'dropout': None,
      'early_stopping_patience': 5
    },
    {
      'symbol': symbol,
      'df': df,
      'layers': [1500],
      'timesteps': 3,
      'test_set': 10,
      'val_set': 5,
      'batch_size': 1,
      'epochs': 500,
      'dropout': None,
      'early_stopping_patience': 5
    },
    {
      'symbol': symbol,
      'df': df,
      'layers': [2000],
      'timesteps': 3,
      'test_set': 10,
      'val_set': 5,
      'batch_size': 1,
      'epochs': 500,
      'dropout': None,
      'early_stopping_patience': 5
    },

    # 5 time steps
    {
      'symbol': symbol,
      'df': df,
      'layers': [30],
      'timesteps': 5,
      'test_set': 10,
      'val_set': 5,
      'batch_size': 1,
      'epochs': 500,
      'dropout': None,
      'early_stopping_patience': 5
    },
    {
      'symbol': symbol,
      'df': df,
      'layers': [50],
      'timesteps': 5,
      'test_set': 10,
      'val_set': 5,
      'batch_size': 1,
      'epochs': 500,
      'dropout': None,
      'early_stopping_patience': 5
    },
    {
      'symbol': symbol,
      'df': df,
      'layers': [100],
      'timesteps': 5,
      'test_set': 10,
      'val_set': 5,
      'batch_size': 1,
      'epochs': 500,
      'dropout': None,
      'early_stopping_patience': 5
    },
    {
      'symbol': symbol,
      'df': df,
      'layers': [150],
      'timesteps': 5,
      'test_set': 10,
      'val_set': 5,
      'batch_size': 1,
      'epochs': 500,
      'dropout': None,
      'early_stopping_patience': 5
    },
    {
      'symbol': symbol,
      'df': df,
      'layers': [200],
      'timesteps': 5,
      'test_set': 10,
      'val_set': 5,
      'batch_size': 1,
      'epochs': 500,
      'dropout': None,
      'early_stopping_patience': 5
    },
    {
      'symbol': symbol,
      'df': df,
      'layers': [300],
      'timesteps': 5,
      'test_set': 10,
      'val_set': 5,
      'batch_size': 1,
      'epochs': 500,
      'dropout': None,
      'early_stopping_patience': 5
    },
    {
      'symbol': symbol,
      'df': df,
      'layers': [500],
      'timesteps': 5,
      'test_set': 10,
      'val_set': 5,
      'batch_size': 1,
      'epochs': 500,
      'dropout': None,
      'early_stopping_patience': 5
    },
    {
      'symbol': symbol,
      'df': df,
      'layers': [1000],
      'timesteps': 5,
      'test_set': 10,
      'val_set': 5,
      'batch_size': 1,
      'epochs': 500,
      'dropout': None,
      'early_stopping_patience': 5
    },
    {
      'symbol': symbol,
      'df': df,
      'layers': [1500],
      'timesteps': 5,
      'test_set': 10,
      'val_set': 5,
      'batch_size': 1,
      'epochs': 500,
      'dropout': None,
      'early_stopping_patience': 5
    },
    {
      'symbol': symbol,
      'df': df,
      'layers': [2000],
      'timesteps': 5,
      'test_set': 10,
      'val_set': 5,
      'batch_size': 1,
      'epochs': 500,
      'dropout': None,
      'early_stopping_patience': 5
    },

    # 7 time steps
    {
      'symbol': symbol,
      'df': df,
      'layers': [30],
      'timesteps': 7,
      'test_set': 10,
      'val_set': 5,
      'batch_size': 1,
      'epochs': 500,
      'dropout': None,
      'early_stopping_patience': 5
    },
    {
      'symbol': symbol,
      'df': df,
      'layers': [50],
      'timesteps': 7,
      'test_set': 10,
      'val_set': 5,
      'batch_size': 1,
      'epochs': 500,
      'dropout': None,
      'early_stopping_patience': 5
    },
    {
      'symbol': symbol,
      'df': df,
      'layers': [100],
      'timesteps': 7,
      'test_set': 10,
      'val_set': 5,
      'batch_size': 1,
      'epochs': 500,
      'dropout': None,
      'early_stopping_patience': 5
    },
    {
      'symbol': symbol,
      'df': df,
      'layers': [150],
      'timesteps': 7,
      'test_set': 10,
      'val_set': 5,
      'batch_size': 1,
      'epochs': 500,
      'dropout': None,
      'early_stopping_patience': 5
    },
    {
      'symbol': symbol,
      'df': df,
      'layers': [200],
      'timesteps': 7,
      'test_set': 10,
      'val_set': 5,
      'batch_size': 1,
      'epochs': 500,
      'dropout': None,
      'early_stopping_patience': 5
    },
    {
      'symbol': symbol,
      'df': df,
      'layers': [300],
      'timesteps': 7,
      'test_set': 10,
      'val_set': 5,
      'batch_size': 1,
      'epochs': 500,
      'dropout': None,
      'early_stopping_patience': 5
    },
    {
      'symbol': symbol,
      'df': df,
      'layers': [500],
      'timesteps': 7,
      'test_set': 10,
      'val_set': 5,
      'batch_size': 1,
      'epochs': 500,
      'dropout': None,
      'early_stopping_patience': 5
    },
    {
      'symbol': symbol,
      'df': df,
      'layers': [1000],
      'timesteps': 7,
      'test_set': 10,
      'val_set': 5,
      'batch_size': 1,
      'epochs': 500,
      'dropout': None,
      'early_stopping_patience': 5
    },
    {
      'symbol': symbol,
      'df': df,
      'layers': [1500],
      'timesteps': 7,
      'test_set': 10,
      'val_set': 5,
      'batch_size': 1,
      'epochs': 500,
      'dropout': None,
      'early_stopping_patience': 5
    },
    {
      'symbol': symbol,
      'df': df,
      'layers': [2000],
      'timesteps': 7,
      'test_set': 10,
      'val_set': 5,
      'batch_size': 1,
      'epochs': 500,
      'dropout': None,
      'early_stopping_patience': 5
    },
  ]

  runner(param_sequence)

if __name__ == "__main__":
  main()