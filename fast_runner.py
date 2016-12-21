"""
Fast model runner.
"""

from __future__ import print_function
import datetime
from dateutil.relativedelta import relativedelta

import model
import data
import utils

def runner(param_sequence, verbose=0):

  for params in param_sequence:
    results = model.run(params, verbose=verbose)

    # adjust params for printing
    df = params.get('df')
    params['features'] = df.columns.values.tolist()
    params['samples'] = df.shape[0]
    params.pop('df', None)

    print('\nParams:', params)
    print('Prediction Accuracy:', results.get('accuracy'))

    utils.save_prediction_to_db(results)

def load_data(symbol):
  
  # define max date range: 10 years
  end_date = datetime.datetime.now()
  start_date = end_date - relativedelta(years=10)

  # get daily stock data
  print('Loading daily stock data...')
  df = data.get_daily_stock_data(symbol, start_date=start_date, end_date=end_date)

  # select columns
  adjclose_column = 'Adj. Close'
  df = df[[adjclose_column]]

  # drop n/a values
  df = df.dropna()

  return df

def train_symbol(symbol):

  print('\nTraining models for %s' % symbol)

  # load data
  df = load_data(symbol)

  # build param sequence
  params = utils.build_model_params(
    architectures=(
      [[300], None],
    ),
    timesteps=(5,),
    steps_ahead=range(1, 11)
  )

  param_sequence = []
  for p in params:
    param = {
      'symbol': symbol,
      'df': df,
      'layers': p[0],
      'dropout': p[1],
      'timesteps': p[2],
      'steps_ahead': p[3],
      'test_set': 30,
      'val_set': 30,
      'batch_size': 10,
      'epochs': 500,
      'early_stopping_patience': 5
    }
    param_sequence.append(param)


  # start runner
  runner(param_sequence, verbose=1)

def main():

  symbols = ['AAPL', 'AMZN', 'FB', 'GOOGL', 'GRPN', 'NFLX', 'NVDA', 'PCLN', 'TSLA']
  for symbol in symbols:
    train_symbol(symbol)

if __name__ == "__main__":
  main()
