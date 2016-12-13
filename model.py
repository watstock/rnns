"""
Watstock's market prediction neural network.
"""

from __future__ import print_function
import time

import numpy as np
import math

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

def build_model(layers, sequence_length, dropout=None):
  model = Sequential()

  for i in xrange(len(layers) - 2):
    return_sequences = i < len(layers) - 3
    layer = None
    if i == 0:
      layer = GRU(layers[i+1],
                  input_shape=(sequence_length, layers[i]),
                  return_sequences=return_sequences)
    else:
      layer = GRU(layers[i+1],
                  return_sequences=return_sequences)

    model.add(layer)

  # adding dropout
  if dropout != None:
    model.add(Dropout(dropout))

  model.add(Dense(layers[-1]))
  model.add(Activation('linear')) # Since we are doing a regression, its activation is linear

  model.compile(loss='mse', optimizer='rmsprop')

  return model


def train_model(model, data, batch_size=1, epochs=100, valset=30, patience=5, verbose=0):
  X_train, Y_train = data

  early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=verbose)
  val_ratio = 1.0 * valset / len(X_train)

  start_time = time.time()
  model.fit(X_train, 
            Y_train, 
            batch_size=batch_size, 
            nb_epoch=epochs,
            verbose=verbose,
            validation_split=val_ratio, 
            callbacks=[early_stopping])

  return (time.time() - start_time)


def run(params, verbose=0):

  symbol = params.get('symbol')
  df = params.get('df')
  date_from = df.index[0].strftime('%Y-%m-%d')
  date_to = df.index[-1].strftime('%Y-%m-%d')
  features = df.shape[1]

  layers = params.get('layers', [300])
  timesteps = params.get('timesteps', 15)
  testset = params.get('test_set', 30)
  valset = params.get('val_set', 30)
  batch_size = params.get('batch_size', 10)
  
  epochs = params.get('epochs', 500)
  dropout = params.get('dropout', None)
  early_stopping_patience = params.get('early_stopping_patience', 5)

  architecture = [features] + layers + [1]

  if verbose == 1:
    print('Symbol:', symbol)
    print('Date from:', date_from)
    print('Date to:', date_to)
    print('Time steps:', timesteps)
    print('Test set:', testset)
    print('Batch size:', batch_size)
    print('Features:', df.columns.values)
    print('Architecture:', architecture)
    print('Dropout:', dropout)

  # Normalize the dataset
  dataset = df.copy().values
  dataset = dataset.astype('float32')
  scaler = MinMaxScaler(feature_range=(0, 1))
  dataset = scaler.fit_transform(dataset)

  # Split into train and test sets
  X_train, Y_train, X_test, Y_test = split_dataset(dataset, timesteps=timesteps, testset=testset)
  if verbose == 1:
    print('Train set:', len(X_train), ', test set:', len(X_test))

  # Create and fit the RNN
  model = build_model(architecture, sequence_length=timesteps, dropout=dropout)
  train_duration = train_model(model, (X_train, Y_train), batch_size=batch_size, 
    epochs=epochs, valset=valset, patience=early_stopping_patience, verbose=verbose)

  # Make predictions
  if verbose == 1:
    print('Predicting...')

  Y_train_prediction = model.predict(X_train, batch_size=batch_size)
  Y_test_prediction = model.predict(X_test, batch_size=batch_size)

  # Scale data back
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

  # Calculate Mean Absolute Error (MAE)
  train_mae = (abs(Y_train - Y_train_prediction)).mean()
  test_mae = (abs(Y_test - Y_test_prediction)).mean()
  if verbose == 1:
    print('Train MAE: %.4f' % train_mae)
    print('Test MAE: %.4f' % test_mae)

  # Calculate Root Mean Squared Error (RMSE)
  train_rmse = math.sqrt(mean_squared_error(Y_train, Y_train_prediction))
  test_rmse = math.sqrt(mean_squared_error(Y_test, Y_test_prediction))
  if verbose == 1:
    print('Train RMSE: %.4f' % train_rmse)
    print('Test RMSE: %.4f' % test_rmse)

  # Calculate accuracy as Mean Absolute Percentage Error
  train_ape_ds = abs(Y_train - Y_train_prediction) / abs(Y_train) * 100
  test_ape_ds = abs(Y_test - Y_test_prediction) / abs(Y_test) * 100
  
  train_mape = train_ape_ds.mean()
  train_ape_max = train_ape_ds.max()

  test_mape = test_ape_ds.mean()
  test_ape_max = test_ape_ds.max()

  if verbose == 1:
    print('Train MAPE: %.4f' % train_mape)
    print('Train APE max: %.4f' % train_ape_max)
    print('Test MAPE: %.4f' % test_mape)
    print('Test APE max: %.4f' % test_ape_max)

  test_index = df.index[-Y_test.shape[0]:].strftime('%Y-%m-%d')

  prediction = {
    'symbol': symbol,
    'date_from': date_from,
    'date_to': date_to,
    'train_set': len(X_train),
    'val_set': valset,
    'test_set': len(X_test),
    'dates': test_index.tolist(),
    'features': df.columns.values.tolist(),
    'history': Y_test.tolist(),
    'prediction': Y_test_prediction.tolist(),
    'timesteps': timesteps,
    'architecture': architecture,
    'dropout': dropout,
    'train_rmse': train_rmse,
    'test_rmse': test_rmse,
    'train_mae': train_mae,
    'test_mae': test_mae,
    'train_ape_max': train_ape_max,
    'train_mape': train_mape,
    'test_ape_max': test_ape_max,
    'test_mape': test_mape,
    'train_accuracy': (100 - train_mape),
    'test_accuracy': (100 - test_mape),
    'train_duration': train_duration,
    'batch_size': batch_size,
    'epochs': epochs,
    'early_stopping_patience': early_stopping_patience
  }

  if verbose == 1:
    print('Prediction results:')
    print(prediction)

  return prediction
