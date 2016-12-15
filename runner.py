from __future__ import print_function
import os
import pandas as pd
import time
import datetime
from dateutil.relativedelta import relativedelta
from pymongo import MongoClient

import model
import data

def plot_data(df):

  import matplotlib.pyplot as plt

  df.plot()
  plt.show()

def save_prediction_to_db(data):

  MONGODB_CONNECTION = os.environ['MONGODB_CONNECTION']
  client = MongoClient(MONGODB_CONNECTION)
  
  db = client.watstock
  collection = db.predictions

  prediction = data

  now = datetime.datetime.utcnow()
  prediction['timestamp'] = now
  prediction['date'] = now.strftime('%Y-%m-%d')

  prediction_id = collection.insert_one(prediction).inserted_id
  print('Prediction saved to the db, id:', prediction_id)


def add_day_of_year_data(df):
  df_dayofyear = pd.to_datetime(df.index.values).dayofyear
  dayofyear_df = pd.DataFrame(data=df_dayofyear, index=df.index.values, columns=['Day of year'])
  
  dayofyear_df = dayofyear_df.join(df)
  return dayofyear_df

def runner(param_sequence, predict=1, verbose=0):

  for params in param_sequence:

    results = model.run(params, predict=predict, verbose=verbose)

    # adjust params for printing
    df = params.get('df')
    params['features'] = df.columns.values.tolist()
    params['samples'] = df.shape[0]
    params.pop('df', None)

    print('\nParams:', params)
    print('Prediction Accuracy:', results.get('prediction_accuracy'))

    save_prediction_to_db(results)

def train_symbol(symbol):

  print('\nTraining models for %s' % symbol)

  # define max data range: 10 years
  end_date = datetime.datetime.now()
  start_date = end_date - relativedelta(years=10)

  # get daily stock data
  print('Loading daily stock data...')
  df = data.get_stock_data(symbol, start_date=start_date, end_date=end_date)  
  adjclose_column = 'Adj. Close'
  adjsclose_df = df[adjclose_column]

  # add tech data
  rolling_window = 20

  # get rolling mean
  df_rmeam = data.get_rolling_mean(adjsclose_df, window=rolling_window)

  # get rolling std
  df_rstd = data.get_rolling_std(adjsclose_df, window=rolling_window)

  df = df.join(df_rmeam)
  df = df.join(df_rstd)
  
  # get AOS data
  print('Loading AOS data...')
  df_aos = data.get_aos_data(symbol)
  df = df.join(df_aos)

  # fill gaps
  df['Article Sentiment'].fillna(value=0., inplace=True)
  df['Impact Score'].fillna(value=0., inplace=True)

  # re-order data, so Adj. Close is the last column
  columns = df.columns.tolist()
  adjclose_index = columns.index(adjclose_column)
  columns = columns[:adjclose_index] + columns[adjclose_index+1:] + [adjclose_column]
  df = df[columns]

  # drop n/a values
  df = df.dropna()

  param_sequence = [
    # 10 time steps    
    {
      'symbol': symbol,
      'df': df,
      'layers': [100],
      'timesteps': 10,
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
      'layers': [300],
      'timesteps': 10,
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
      'timesteps': 10,
      'test_set': 30,
      'val_set': 30,
      'batch_size': 10,
      'epochs': 500,
      'dropout': None,
      'early_stopping_patience': 5
    }
  ]

  runner(param_sequence, predict=10)

def main():

  symbols = ['NFLX', 'NVDA', 'PCLN', 'TSLA']
  for symbol in symbols:
    train_symbol(symbol)

if __name__ == "__main__":
  main()
