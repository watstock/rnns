"""
Watstock's market prediction neural network.
"""

from __future__ import print_function
import time
import os
import numpy as np
import pandas as pd
import math
import json

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dropout
from keras.layers import Activation
from keras.callbacks import EarlyStopping

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# fixing random generator
np.random.seed(1234)


def create_dataset(df, timesteps=1):
  """Converts a dataframe of values into a matrix"""
  dataX, dataY = [], []
  for i in range(len(df)-timesteps):
    dataX.append(df[i:(i+timesteps), :])
    dataY.append(df[i + timesteps, :])

  return np.array(dataX), np.array(dataY)

def split_dataset(dataframe, timesteps=90, testset=30):
  # split into train and test datasets
  testset_ratio = 1.0 * (testset + timesteps)/len(dataframe)
  train_size = int(len(dataframe) * (1 - testset_ratio))
  test_size = len(dataframe) - train_size
  train_df, test_df = dataframe[0:train_size,:], dataframe[train_size:,:]

  # Reshape into X=...,t-3,t-2,t-1,t and Y=t+1
  X_train, Y_train = create_dataset(train_df, timesteps)
  X_test, Y_test = create_dataset(test_df, timesteps)

  # Reshape inputs from [samples, features] to [samples, timesteps, features] format
  X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
  X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

  # Leave only columns for prediction
  Y_train = Y_train[:,-1:]
  Y_test = Y_test[:,-1:]

  return X_train, Y_train, X_test, Y_test

def build_model(layers, sequence_length):
  model = Sequential()

  for i in xrange(len(layers) - 1):
    return_sequences = i < len(layers) - 2
    layer = None
    if i == 0:
      layer = GRU(layers[i+1],
                  input_shape=(sequence_length, layers[i]),
                  return_sequences=return_sequences)
    else:
      layer = GRU(layers[i+1])

    model.add(layer)

  model.add(Dense(layers[-1]))
  model.add(Activation('linear')) # Since we are doing a regression, its activation is linear

  model.compile(loss='mse', optimizer='rmsprop')

  return model


def train_model(model, data, batch_size=1, epochs=100):
  X_train, Y_train = data

  early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0)

  start_time = time.time()
  model.fit(X_train, 
            Y_train, 
            batch_size=batch_size, 
            nb_epoch=epochs,
            verbose=1,
            validation_split=0.1, 
            callbacks=[early_stopping])

  return (time.time() - start_time)


def save_prediction(data):

  # serialize
  json_data = json.dumps(data, sort_keys=True, indent=4, separators=(',', ': '))

  # build file name
  current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
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


def main():

  # params
  symbol = 'AAPL'
  date_from = '2006-12-02'
  date_to = '2016-12-02'
  tsteps = 60
  testset = 30
  layers = [300]

  batch_size = 1
  epochs = 100

  print('Symbol:', symbol)
  print('Date from:', date_from)
  print('Date to:', date_to)
  print('Time steps:', tsteps)
  print('Test set:', testset)
  print('Layers:', layers)

  # Define a date range
  dates = pd.date_range(date_from, date_to)

  # Get stock data
  df = get_data(symbol, dates, usecols=['Date', 'Volume', 'Adj Close'])
  df = df.dropna()

  # Add sentiment data
  # df_sentiment = get_data('AOS-AAPL', dates, usecols=['Date', 'Article Sentiment', 'Impact Score'])
  # df = df.join(df_sentiment)

  # Add day of year data
  # df_dayofyear = pd.to_datetime(df.index.values).dayofyear
  # dayofyear_df = pd.DataFrame(data=df_dayofyear, index=df.index.values, columns=['Day of year'])
  # df = df.join(dayofyear_df)

  # Reordering columns so Ads Close is the last
  #df = df[['Day of year', 'Volume', 'Adj Close']]
  # df = df[['Article Sentiment', 'Impact Score', 'Volume', 'Adj Close']]

  features = df.shape[1]
  print('Features:', features)

  # Making df to be devisible by batch size (to use with statefull LSTMs)
  # df_offset = (len(df) - 2 * tsteps ) % batch_size
  # df = df.ix[df_offset:]

  # Normalize the dataset    
  dataset = df.values
  dataset = dataset.astype('float32')
  
  scaler = MinMaxScaler(feature_range=(0, 1))
  dataset = scaler.fit_transform(dataset)

  # Split into train and test sets
  X_train, Y_train, X_test, Y_test = split_dataset(dataset, timesteps=tsteps, testset=testset)
  print('Train set:', len(X_train), ', test set:', len(X_test))

  # Create and fit the RNN
  architecture = [features] + layers + [1]
  print('Architecture:', architecture)

  model = build_model(architecture, sequence_length=tsteps)
  train_duration = train_model(model, (X_train, Y_train), batch_size=batch_size, epochs=epochs)

  # Make predictions
  print('Predicting...')
  Y_train_prediction = model.predict(X_train, batch_size=batch_size)
  Y_test_prediction = model.predict(X_test, batch_size=batch_size)

  # Invert predictions
  df_shape = np.zeros((X_train.shape[0], X_train.shape[2]))
  df_shape[:,-1:] = Y_train_prediction
  Y_train_prediction = scaler.inverse_transform(df_shape)[:,-1]

  df_shape = np.zeros((X_train.shape[0], X_train.shape[2]))
  df_shape[:,-1:] = Y_train
  Y_train = scaler.inverse_transform(df_shape)[:,-1]

  df_shape = np.zeros((X_test.shape[0], X_test.shape[2]))
  df_shape[:,-1:] = Y_test_prediction
  Y_test_prediction = scaler.inverse_transform(df_shape)[:,-1]
  
  df_shape = np.zeros((X_test.shape[0], X_test.shape[2]))
  df_shape[:,-1:] = Y_test
  Y_test = scaler.inverse_transform(df_shape)[:,-1]

  # Calculate root mean squared error
  # trainScore = math.sqrt(mean_squared_error(Y_train, Y_train_prediction))
  # testScore = math.sqrt(mean_squared_error(Y_test, Y_test_prediction))
  # print('Train Score: %.4f RMSE' % (trainScore))
  # print('Test Score: %.4f RMSE' % (testScore))

  # Shift train predictions for plotting
  # train_ds = np.empty((dataset.shape[0], 1))
  # train_ds[:, :] = np.nan
  # train_ds[tsteps:Y_train_prediction.shape[0] + tsteps, :] = Y_train_prediction
  # train_df = pd.DataFrame(data=train_ds, index=df.index.values, columns=['Training set prediction'])

  # # Shift test predictions for plotting
  # test_ds = np.empty((dataset.shape[0], 1))
  # test_ds[:, :] = np.nan
  # test_ds[Y_train_prediction.shape[0] + 2*tsteps:len(dataset), :] = Y_test_prediction
  # test_df = pd.DataFrame(data=test_ds, index=df.index.values, columns=['Test set prediction'])

  # Calculate accuracy as Mean absolute percentage error
  train_error_ds = abs(Y_train - Y_train_prediction) / Y_train
  test_error_df = abs(Y_test - Y_test_prediction) / Y_test
  print('Train Mean Absolute Percentage Error:', train_error_ds.mean() * 100)
  print('Test Mean Absolute Percentage Error:', test_error_df.mean() * 100)
  
  test_index = df.index[-Y_test.shape[0]:].strftime('%Y-%m-%d')

  prediction = {
    'symbol': symbol,
    'date_from': date_from,
    'date_to': date_to,
    'train_set': len(X_train),
    'test_set': len(X_test),
    'dates': test_index.tolist(),
    'price': Y_test.tolist(),
    'prediction': Y_test_prediction.tolist(),
    'sequence_length': tsteps,
    'architecture': architecture,
    'train_accuracy': (1 - train_error_ds.mean()) * 100,
    'test_accuracy': (1 - test_error_df.mean()) * 100,
    'train_duration': train_duration
  }

  print(prediction)

  print('Saving prediction...')
  save_prediction(prediction)

if __name__ == "__main__":
  main()