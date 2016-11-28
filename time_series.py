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
from sklearn.preprocessing import MinMaxScaler

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

    # Normalize the dataset    
    dataset = df.values
    dataset = dataset.astype('float32')
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # Split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    print(len(train), len(test))

    # Reshape into X=t and Y=t+1
    look_back = 5
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # Reshape input to be [samples, time steps, features], currently we have [samples, features]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    # Create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_dim=1))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, nb_epoch=50, batch_size=1, verbose=2)

    # Make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # Invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    # Calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.4f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.4f RMSE' % (testScore))

    # Shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    train_df = pd.DataFrame(data=trainPredictPlot, index=df.index.values, columns=['Training dataset'])

    # Shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
    test_df = pd.DataFrame(data=testPredictPlot, index=df.index.values, columns=['Test dataset'])
    
    # Plot baseline and predictions
    ax = df.plot(title='SPY prediction', label='SPY')
    train_df.plot(label='Training dataset', ax=ax)
    test_df.plot(label='Test dataset', ax=ax)

    plt.show()

if __name__ == "__main__":
    test_run()