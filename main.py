import pandas as pd
import yfinance as yf
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import Sequential
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.src.layers import LSTM, Dense
from IPython.display import display

# making output from pandas more readable
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

scaler = MinMaxScaler()
train_ticker = ['GOOG']
train_tickers = ['GOOG', 'FLOT', 'TRI', 'DDD', 'WVVI', 'AZO', 'UNP', 'RY', 'RRR', 'CSIQ', 'STG', 'ARVL', 'SGN', 'REFI',
                 'CDRO', 'SDPI', 'SQFTP', 'AXTI', 'MODD', 'TETEU', 'LBTYB', 'SNTI', 'GABK', 'EVCM', 'ADN', 'WWR']
test_ticker = 'CYRX'


def main():
    data = get_data(train_ticker)
    data_mult = get_data(train_tickers)
    init_scaler(data_mult['Close'])
    # train_model_single(data)
    # train_model_mult(data_mult)
    # predict_on_new_dataset(model=keras.saving.load_model('finforecast_single_model.keras'), ticker_symbol=test_ticker)
    predict_on_new_dataset(model=keras.saving.load_model('finforecast_mult_model.keras'), ticker_symbol=test_ticker)


def get_data(ticker_symbols: [str]) -> pd.DataFrame:
    """
    :param ticker_symbols:
    :return: downloaded dataframes
    """
    data = pd.DataFrame()
    for ticker_symbol in ticker_symbols:
        data = pd.concat((data, yf.Ticker(ticker_symbol).history(period='4y', interval='1d')))
    data = data.dropna(axis=1)
    data = data[['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits', 'Close']]
    return data


def init_scaler(data: pd.DataFrame):
    """
    initialize the scaler
    :param data: dataframe with data to init the scaler on
    :return:
    """
    scaler.fit(np.array(data).reshape(-1, 1))


def build_model(data: pd.DataFrame) -> Sequential:
    """
    Build, train and evaluate the model
    :param data: Dataframe with data to train model on
    :return: trained Keras model
    """
    # split data into test and train
    x = data.drop('Close', axis=1)
    y = data['Close']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # normalize data
    y_train = scaler.transform(np.array(y_train).reshape(-1, 1))
    y_test = scaler.transform(np.array(y_test).reshape(-1, 1))

    # make model and train it
    model = keras.models.Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=100)

    # evaluate the model
    test_loss = model.evaluate(x_test, y_test)
    print('Test loss = %f' % np.sqrt(test_loss))

    # testing model and visualizing tests
    y_pred = model.predict(x_test)
    y_pred = scaler.inverse_transform(np.array(y_pred).reshape(-1, 1))
    y_test = scaler.inverse_transform(np.array(y_test).reshape(-1, 1))
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print('RMSE = %f, MAE = %f' % (rmse, mae))
    # plt.plot(y_test, label='Actual')
    # plt.plot(y_pred, label='Predicted')
    # plt.legend()
    # plt.show()

    return model


def train_model_single(data: pd.DataFrame):
    """
    Train model from single stock history from yfinance and save the model
    :param data: Dataframe with stock history to train on
    :return:
    """
    model = build_model(data)

    # saving model for later use
    model.save('finforecast_single_model.keras')


def train_model_mult(data: pd.DataFrame):
    """
    Train model from multiple stock history from yfinance and save the model
    :param data: Dataframe with stock history to train on
    :return:
    """
    model = build_model(data)

    # saving model for later use
    model.save('finforecast_mult_model.keras')


def predict_on_new_dataset(model: Sequential, ticker_symbol: str):
    """
    test the model on new dataset and visualize the predicted values
    :param model: trained model
    :param ticker_symbol: ticker symbol of the new dataset
    :return:
    """
    ticker_test_forecast = get_data([ticker_symbol])
    x_test_forecast = ticker_test_forecast.drop('Close', axis=1)
    y_pred_test_forecast = model.predict(x_test_forecast)
    y_pred_test_forecast = scaler.inverse_transform(np.array(y_pred_test_forecast).reshape(-1, 1))
    x_test_forecast['Close'] = y_pred_test_forecast
    y_test_forecast = ticker_test_forecast['Close']
    plt.plot(y_test_forecast, label='Actual', color='green')
    plt.plot(x_test_forecast['Close'], label='Predicted', alpha=0.7, color='red')
    plt.title(ticker_symbol)
    plt.legend()
    plt.show()
    rmse = np.sqrt(mean_squared_error(y_test_forecast, y_pred_test_forecast))
    mae = mean_absolute_error(y_test_forecast, y_pred_test_forecast)
    print('RMSE = %f, MAE = %f' % (rmse, mae))


def predict_for_timesteps(model: Sequential, days: int, ticker: str) -> pd.DataFrame:
    """
    :param model: Sequential model to predict with
    :param days: number of days to predict for
    :param ticker: ticker to predict for
    :return: dataframe with predicted values
    """
    prediction = pd.DataFrame()
    ticker_data = get_data(ticker).iloc[-1].drop('Close', axis=1)
    pd.concat((prediction, ticker_data))
    for day in range(1, days + 1):
        prediction_day = model.predict(prediction.iloc[-1])
        pd.concat((prediction, prediction_day))
    ticker_data['Close'] = prediction
    return prediction


if __name__ == '__main__':
    main()

# resetting the options
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')
pd.reset_option('display.max_colwidth')
