import pandas as pd
import numpy as np
import tensorflow
import yfinance as yf
import requests
import json
import datetime as dt

# from matplotlib.pylab import rcParams
# rcParams['figure.figsize'] = 20, 10
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics


def queryByCompany(searchparam):
    ticker_company_list = []
    ticker_list = []
    url = f"http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={searchparam}&region=1&lang=en"
    results = requests.get(url).json()
    for x in results['ResultSet']['Result']:
        ticker_company_list.append(x['name'])
        ticker_list.append(x['symbol'])


def getStockData(symbol):

    ticker = yf.Ticker(symbol)
    return ticker


def createDataFrame(ticker):
    df = pd.DataFrame(ticker.history(period="5y")).reset_index()
    return df


def getStockDates(ticker):
    t = getStockData(ticker)
    df = createDataFrame(t)
    df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
    dates = df.Date.apply(lambda x: x.strftime("%Y-%m-%d"))

    data = dates.to_list()

    return data


def getStockClose(ticker):
    t = getStockData(ticker)
    df = createDataFrame(t)
    close = df['Close'].to_list()

    return close


def getCompanyName(ticker):
    t = getStockData(ticker)
    cName = t.info['longName']

    return cName


def startModeling(ticker):
    t = getStockData(ticker)
    df = createDataFrame(t)

    # labelDates = df['Date'].values

    trading_days = df['Date'].count()
    training_set = .7 * trading_days

    data = df.sort_index(ascending=True, axis=0)
    new_dataset = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])
    for i in range(0, len(data)):
        new_dataset["Date"][i] = data['Date'][i]
        new_dataset["Close"][i] = data["Close"][i]

    new_dataset.index = new_dataset.Date
    new_dataset.drop("Date", axis=1, inplace=True)

    final_dataset = new_dataset.values

    train_data = final_dataset[0:int(training_set), :]
    valid_data = final_dataset[int(training_set):, :]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(final_dataset)

    x_train_data, y_train_data = [], []

    for i in range(60, len(train_data)):
        x_train_data.append(scaled_data[i - 60:i, 0])
        y_train_data.append(scaled_data[i, 0])

    x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)
    x_train_data = np.reshape(x_train_data,(x_train_data.shape[0], x_train_data.shape[1], 1))

    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train_data.shape[1], 1)))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    lstm_model.fit(x_train_data, y_train_data, epochs=3, batch_size=3, verbose=2)

    inputs_data = new_dataset[len(new_dataset) - len(valid_data) - 60:].values
    inputs_data = inputs_data.reshape(-1, 1)
    inputs_data = scaler.transform(inputs_data)

    X_test = []

    for i in range(60, inputs_data.shape[0]):
        X_test.append(inputs_data[i - 60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_closing_price = lstm_model.predict(X_test)
    predicted_closing_price = scaler.inverse_transform(predicted_closing_price)

    train_data = new_dataset[:int(training_set)]
    valid_data = new_dataset[int(training_set):]
    valid_data['Predictions'] = predicted_closing_price

    closing_data_df = pd.DataFrame({"TrainData": train_data['Close'], "ActualData": valid_data['Close'], "PredictedData": valid_data['Predictions']})

    train = closing_data_df['TrainData'].dropna().values
    train_dates = closing_data_df['TrainData'].dropna().index.values
    actual = closing_data_df['ActualData'].dropna().values
    actual_dates = closing_data_df['ActualData'].dropna().index.values
    predict = closing_data_df['PredictedData'].dropna().values
    predict_dates = closing_data_df['PredictedData'].dropna().index.values

    data = {'train':train, 'traindates':train_dates, 'actual':actual, 'actualdates':actual_dates, 'predict':predict, 'predictdates':predict_dates}

    return data

# getStockData('DIS')