import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def symbol_to_path(symbol, base_dir="data"):
  """Return CSV file path given ticker symbol."""
  return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates):
  """Read stock data (adjusted close) for given symbols from CSV files."""
  df = pd.DataFrame(index=dates)
  if 'SPY' not in symbols:  # add SPY for reference, if absent
      symbols.insert(0, 'SPY')

  for symbol in symbols:
      df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                            parse_dates=True, usecols=['Date', 'Adj Close'],
                            na_values=['nan'])
      df_temp = df_temp.rename(columns={'Adj Close': symbol})
      df = df.join(df_temp)
      if symbol == 'SPY':
          df = df.dropna(subset=['SPY'])

  return df

def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
  """Plot stock prices with a custom title and meaningful axis labels."""
  ax = df.plot(title=title, fontsize=12)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  plt.show()

def compute_daily_returns(df):
  """Compute and return the daily return values."""
  daily_returns = df.copy()
  daily_returns[1:] = (df[1:]/df[:-1].values) - 1 # to avoid index matching
  daily_returns.ix[0, :] = 0
  return daily_returns


def create_dataset(dataset, look_back=1):
  """Converts an array of values into a dataset matrix"""
  dataX, dataY = [], []
  for i in range(len(dataset)-look_back-1):
    a = dataset[i:(i+look_back), 0]
    dataX.append(a)
    dataY.append(dataset[i + look_back, 0])
  return np.array(dataX), np.array(dataY)

def test_run():

    # Define a date range
    dates = pd.date_range('2015-11-28', '2016-11-25')

    # Choose stock symbols to read
    symbols = ['SPY']
    
    # Get stock data
    df = get_data(symbols, dates)
    #plot_data(df)

    # Compute daily returns
    daily_returns = compute_daily_returns(df)

    # Prepare data for RNN
    dataset = daily_returns.values
    dataset = dataset.astype('float32')

    # Split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    print(len(train), len(test))

    # Reshape into X=t and Y=t+1
    look_back = 3
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # Reshape input to be [samples, time steps, features], currently we have [samples, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # Create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_dim=look_back))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, nb_epoch=30, batch_size=1, verbose=2)

    # Make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # Calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
    print('Train Score: %.4f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY, testPredict[:,0]))
    print('Test Score: %.4f RMSE' % (testScore))

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
    # plot baseline and predictions
    plt.plot(dataset)
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()

    # plot_data(daily_returns, title="Daily returns", ylabel="Daily returns")

if __name__ == "__main__":
    test_run()