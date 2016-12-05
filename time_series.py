
from __future__ import print_function
import os
import numpy as np
import pandas as pd
import math

# import matplotlib.pyplot as plt

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

# def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
#   """Plot stock prices with a custom title and meaningful axis labels."""
#   ax = df.plot(title=title, fontsize=12)
#   ax.set_xlabel(xlabel)
#   ax.set_ylabel(ylabel)
#   plt.show()

def compute_daily_returns(df):
  """Compute and return the daily return values."""
  daily_returns = df.copy()
  daily_returns[1:] = (df[1:]/df[:-1].values) - 1 # to avoid index matching
  daily_returns.ix[0, :] = 0
  return daily_returns


def create_dataset(df, look_back=1):
  """Converts a dataframe of values into a matrix"""
  dataX, dataY = [], []
  for i in range(len(df)-look_back):
    dataX.append(df[i:(i+look_back), :])
    dataY.append(df[i + look_back, :])
  return np.array(dataX), np.array(dataY)

def main():

  # dates = pd.date_range('2014-01-01', '2015-01-01')
  # df1 = get_data('AAPL', dates)
  # df1 = df1.dropna()
  # df2 = get_data('AOS-AAPL', dates, usecols=['Date', 'Article Sentiment', 'Impact Score'])
  # print(df2['Article Sentiment'].mean())
  # df2.ix[:,0] = df2.ix[:,0] / 2 * 100.0 + 50.0

  # df = df1.join(df2)
  # df.plot(title='AAPL', label='AAPL')
  # plt.show()
  # return

  # Define a date range
  #dates = pd.date_range('2014-01-01', '2015-01-01') # AOS sentiment test data
  dates = pd.date_range('2001-12-02', '2016-12-02')

  # params
  tsteps = 50
  batch_size = 1
  epochs = 100
  testset_ratio = 0.5
  symbol = 'AAPL'

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
  # train_size = (int((len(dataset) - 2 * tsteps) * 0.9) // batch_size) * batch_size + tsteps
  train_size = int(len(dataset) * (1 - testset_ratio))
  test_size = len(dataset) - train_size
  train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
  print('Train set:', len(train), ', test set:', len(test))

  # Reshape into X=t and Y=t+1
  trainX, trainY = create_dataset(train, tsteps)
  testX, testY = create_dataset(test, tsteps)

  # Reshape input to be [samples, time steps, features], currently we have [samples, features]
  trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], trainX.shape[2]))
  testX = np.reshape(testX, (testX.shape[0], testX.shape[1], testX.shape[2]))

  # Using just Adj close for prediction
  trainYClose = trainY[:,-1:]
  testYClose = testY[:,-1:]

  # Create and fit the LSTM network
  print('Creating Model...')
  model = Sequential()
  model.add(GRU(100,
                input_shape=(tsteps, features),
                return_sequences=False))
  # model.add(Dropout(0.2)) # 20% dropout
  # model.add(GRU(300,
  #               return_sequences=False))
  # model.add(Dropout(0.2)) # 20% dropout
  model.add(Dense(1))
  model.add(Activation('linear')) # Since we are doing a regression, its activation is linear

  model.compile(loss='mse', optimizer='rmsprop')

  print('Training...')
  early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0)
  model.fit(trainX, 
            trainYClose, 
            batch_size=batch_size, 
            nb_epoch=epochs,
            verbose=1,
            validation_split=0.1, 
            callbacks=[early_stopping])
  # for i in xrange(epochs):
  #   print('Epoch', i, '/', epochs)
  #   model.fit(trainX, 
  #             trainY, 
  #             batch_size=batch_size,
  #             verbose=1,
  #             nb_epoch=1, 
  #             shuffle=False)
  #   model.reset_states() # reseting the state after the whole sequence has been processed

  # Make predictions
  print('Predicting...')
  trainPredict = model.predict(trainX, batch_size=batch_size)
  # model.reset_states()
  testPredict = model.predict(testX, batch_size=batch_size)

  # Invert predictions
  trainPredictDataset = np.empty((trainX.shape[0], trainX.shape[2]))
  trainPredictDataset[:,:] = .0
  trainPredictDataset[:,-1:] = trainPredict
  trainPredict = scaler.inverse_transform(trainPredictDataset)
  trainPredict[:,:-1] = np.nan
  trainY = scaler.inverse_transform(trainY)

  testPredictDataset = np.empty((testX.shape[0], testX.shape[2]))
  testPredictDataset[:,:] = .0
  testPredictDataset[:,-1:] = testPredict
  testPredict = scaler.inverse_transform(testPredictDataset)
  testPredict[:,:-1] = np.nan
  testY = scaler.inverse_transform(testY)

  # Calculate root mean squared error
  trainScore = math.sqrt(mean_squared_error(trainY[:,-1], trainPredict[:,-1]))
  print('Train Score: %.4f RMSE' % (trainScore))
  testScore = math.sqrt(mean_squared_error(testY[:,-1], testPredict[:,-1]))
  print('Test Score: %.4f RMSE' % (testScore))

  # Shift train predictions for plotting
  trainPredictPlot = np.empty((dataset.shape[0], 1))
  trainPredictPlot[:, :] = np.nan
  trainPredictPlot[tsteps:len(trainPredict) + tsteps, :] = trainPredict[:,-1:]
  train_df = pd.DataFrame(data=trainPredictPlot, index=df.index.values, columns=['Training set prediction'])

  # Shift test predictions for plotting
  testPredictPlot = np.empty((dataset.shape[0], 1))
  testPredictPlot[:, :] = np.nan
  testPredictPlot[len(trainPredict) + 2*tsteps:len(dataset), :] = testPredict[:,-1:]
  test_df = pd.DataFrame(data=testPredictPlot, index=df.index.values, columns=['Test set prediction'])

  # Calculate accuracy as Mean absolute percentage error
  train_accuracy_df = pd.DataFrame(data=(abs(df.values[:,-1:]-train_df.values)/df.values[:,-1:]*100), index=df.index.values, columns=['Error'])
  print ('Train Mean Absolute Percentage Error:', train_accuracy_df['Error'].mean())

  test_accuracy_df = pd.DataFrame(data=(abs(df.values[:,-1:]-test_df.values)/df.values[:,-1:]*100), index=df.index.values, columns=['Error'])
  print ('Test Mean Absolute Percentage Error:', test_accuracy_df['Error'].mean())

  # Plot baseline and predictions
  # df = df.ix[-test_size:]
  # train_df = train_df.ix[-test_size:]
  # test_df = test_df.ix[-test_size:]

  # price_df = df[['Adj Close']]
  # price_df = price_df.rename(columns={'Adj Close': symbol})
  # ax = price_df.plot(title='1-day prediction', fontsize=12)
  # ax.set_ylabel('Price')
  # # train_df.plot(label='Training set prediction', ax=ax)
  # test_df.plot(label='Test set prediction', ax=ax)

  # plt.show()

if __name__ == "__main__":
  main()