"""
Watstock's market prediction neural network.
"""

from __future__ import print_function
import time
import datetime
from dateutil.relativedelta import relativedelta
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

import utils

# fixing random generator
np.random.seed(1234)


def create_dataset(df, steps_ahead=1, timesteps=1):
  """Converts a dataframe of values into a matrix"""
  dataX, dataY = [], []
  for i in xrange(len(df) - timesteps - steps_ahead + 1):
    dataX.append(df[i:(i + timesteps), :])
    dataY.append(df[i + timesteps + steps_ahead - 1, :])

  return np.array(dataX), np.array(dataY)

def split_dataset(dataframe, steps_ahead=1, timesteps=30, testset=30):
  # split into train and test datasets
  testset_ratio = 1.0 * (testset + timesteps + steps_ahead - 1) / len(dataframe)
  train_size = int(len(dataframe) * (1 - testset_ratio))
  train_df, test_df = dataframe[0:train_size,:], dataframe[train_size:,:]

  # Reshape into X=...,t-3,t-2,t-1,t and Y=t+steps_ahead
  X_train, Y_train = create_dataset(train_df, steps_ahead=steps_ahead, timesteps=timesteps)
  X_test, Y_test = create_dataset(test_df, steps_ahead=steps_ahead, timesteps=timesteps)

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

  # output layer
  model.add(Dense(layers[-1]))
  model.add(Activation('linear')) # Since we are doing a regression, its activation is linear

  # choose optimizer
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

def normalize_dataframe(df):

  scaler = MinMaxScaler(feature_range=(0, 1))

  dataset = df.copy().values
  dataset = dataset.astype('float32')  
  dataset = scaler.fit_transform(dataset)

  return dataset, scaler

def calculate_model(symbol, dataset, scaler, architecture, steps_ahead=1, 
  timesteps=15, testset=30, valset=30, dropout=None, batch_size=10, epochs=500, 
  early_stopping_patience=5, verbose=1):

  # Split into train and test sets
  X_train, Y_train, X_test, Y_test = split_dataset(dataset, timesteps=timesteps, testset=testset, steps_ahead=steps_ahead)
  if verbose == 1:
    print('Train set:', len(X_train), ', test set:', len(X_test))

  # Build model
  if verbose == 1:
    print('Building model...')

  model = build_model(architecture, sequence_length=timesteps, dropout=dropout)

  # Train model
  if verbose == 1:
    print('Training model...')

  train_duration = train_model(model, (X_train, Y_train), batch_size=batch_size, 
    epochs=epochs, valset=valset, patience=early_stopping_patience, verbose=verbose)

  # Estimate model
  if verbose == 1:
    print('Estimating model...')

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
    print('Train APE max: %.4f' % train_ape_max)
    print('Train MAPE: %.4f' % train_mape)
    print('Test APE max: %.4f' % test_ape_max)
    print('Test MAPE: %.4f' % test_mape)
  
  # Predicting next point
  sequence = dataset[-timesteps:]
  sequence = np.reshape(sequence, (1, timesteps, dataset.shape[1]))
  prediction = model.predict(sequence, batch_size=batch_size)

  # Scaling back prediction
  df_shape = np.zeros((1, dataset.shape[1]))
  df_shape[:,-1:] = prediction
  prediction = scaler.inverse_transform(df_shape)[0,-1]

  params = {
    'train_set': len(X_train),
    'val_set': valset,
    'test_set': len(X_test),
    'train_rmse': train_rmse,
    'test_rmse': test_rmse,
    'train_mae': train_mae,
    'test_mae': test_mae,
    'train_ape_max': train_ape_max,
    'test_ape_max': test_ape_max,
    'train_mape': train_mape,
    'test_mape': test_mape,
    'train_accuracy': (100 - train_mape),
    'test_accuracy': (100 - test_mape),
    'train_duration': train_duration
  }

  return model, prediction, params


def run(params, verbose=1):

  symbol = params.get('symbol')
  df = params.get('df')
  date_from = df.index[0].strftime('%Y-%m-%d')
  date_to = df.index[-1].strftime('%Y-%m-%d')
  features = df.shape[1]

  layers = params.get('layers', [300])
  timesteps = params.get('timesteps', 15)
  steps_ahead = params.get('steps_ahead', 1)
  testset = params.get('test_set', 30)
  valset = params.get('val_set', 30)
  batch_size = params.get('batch_size', 10)
  
  epochs = params.get('epochs', 500)
  dropout = params.get('dropout', None)
  early_stopping_patience = params.get('early_stopping_patience', 5)

  architecture = [features] + layers + [1]

  if verbose == 1:
    print('\nSymbol:', symbol)
    print('Date from:', date_from)
    print('Date to:', date_to)
    print('Time steps:', timesteps)
    print('Test set:', testset)
    print('Batch size:', batch_size)
    print('Features:', df.columns.values)
    print('Architecture:', architecture)
    print('Dropout:', dropout)
    print('Predict: %s step(s) ahead' % steps_ahead)

  # Normalize the dataset
  dataset, scaler = normalize_dataframe(df)

  # Building and training model
  _, prediction, model_params = calculate_model(symbol, dataset, scaler, architecture, 
    steps_ahead=steps_ahead, timesteps=timesteps, testset=testset, valset=valset, dropout=dropout, 
    batch_size=batch_size, epochs=epochs, early_stopping_patience=early_stopping_patience, 
    verbose=verbose)

  # Building n-day prediction
  last_date = df.index[-1]
  prediction_date = utils.next_work_day(last_date, distance=steps_ahead).strftime('%Y-%m-%d')

  result = {
    'symbol': symbol,
    'history_from': date_from,
    'history_to': date_to,
    'timesteps': timesteps,
    'steps_ahead': steps_ahead,
    'architecture': architecture,
    'features': df.columns.values.tolist(),
    'dropout': dropout,    
    'batch_size': batch_size,
    'epochs': epochs,
    'early_stopping_patience': early_stopping_patience,
    'prediction_date': prediction_date,
    'prediction': prediction,
    'model': model_params,
    'accuracy': model_params.get('test_accuracy')
  }

  if verbose == 1:
    print('\nPrediction results:')
    print(result)

  return result
