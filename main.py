import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import keras
from IPython.core.display_functions import display
from keras import Sequential
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.src.layers import LSTM, Dense
from datetime import timedelta
import seaborn as sns

# making output from pandas more readable
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

scaler = MinMaxScaler()
train_ticker = ['IBM']
train_tickers = ['NVR', 'AZO', 'AVGO', 'ADBE', 'MSFT', 'GOOG', 'TEAM', 'HLT', 'AMZN', 'GRMN', 'OTTR', 'TEX', 'CSCO',
                 'BEPC', 'PSO', 'INFN', 'KODK', 'KRON', 'AMBO', 'WF', 'OBK', 'VVV', 'VET', 'DDD', 'CTRA', 'RRR', 'CSIQ']
test_ticker = ['GRIN', 'BHR', 'WGS', 'CHT', 'BFRG', 'AORT', 'NI', 'LEGN', 'TMO', 'IGTA', 'NNVC', 'AVNT', 'LYTS', 'INLX',
               'MS', 'CNP', 'GTY', 'HPE', 'NGL']


def main():
    data = get_data(train_ticker)
    data_mult = get_data(train_tickers)
    init_scaler(pd.concat((data_mult, get_data(test_ticker)))['Close'])
    # train_model_single(data)
    # train_model_mult(data_mult)
    # predict_on_new_dataset(model=keras.saving.load_model('finforecast_single_model.keras'),
    #                        ticker_symbols=train_ticker)
    # predict_on_new_dataset(model=keras.saving.load_model('finforecast_mult_model.keras'), ticker_symbols=['AVNT'])
    predict_for_timesteps(model=keras.saving.load_model('finforecast_mult_model.keras'), days=30, ticker=['AVNT'])


def get_data(ticker_symbols: [str]) -> pd.DataFrame:
    """
    :param ticker_symbols:
    :return: downloaded dataframes
    """
    data = pd.DataFrame()
    for ticker_symbol in ticker_symbols:
        data = pd.concat((data, yf.Ticker(ticker_symbol).history(period='5y', interval='1d', actions=False)))
    data = data.dropna(axis=1)
    data = data[['Open', 'High', 'Low', 'Volume', 'Close']]
    return data


def init_scaler(data: pd.DataFrame):
    """
    initialize the scaler
    :param data: dataframe with data to init the scaler on
    :return:
    """
    num = np.arange(-100, 8000)
    scaler.fit(num.reshape(-1, 1))
    # scaler.fit(np.array(data).reshape(-1, 1))


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
    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=7, verbose=1, mode='auto', start_from_epoch=25,
                                             restore_best_weights=True)
    model = keras.models.Sequential()
    model.add(LSTM(units=400, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=200, return_sequences=True))
    model.add(LSTM(units=200, return_sequences=True))
    # model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=200))
    model.add(Dense(units=1))
    model.compile(loss=keras.losses.mean_absolute_percentage_error,
                  optimizer=keras.optimizers.Adamax(),
                  metrics=['mean_absolute_error'],)
    model.fit(x_train, y_train, epochs=100)

    # evaluate the model
    test_loss = model.evaluate(x_test, y_test)
    print('Test loss = %f %%\nMAE = %f' % (np.sqrt(test_loss[0]), test_loss[1]))

    # testing model and visualizing tests
    y_pred = model.predict(x_test)
    y_pred = scaler.inverse_transform(np.array(y_pred).reshape(-1, 1))
    y_test = scaler.inverse_transform(np.array(y_test).reshape(-1, 1))
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print('RMSE = %f, MAE = %f, MAPE = %f' % (rmse, mae, mape))
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


def predict_on_new_dataset(model: Sequential, ticker_symbols: [str]):
    """
    test the model on new dataset and visualize the predicted values
    :param model: trained model
    :param ticker_symbols: ticker symbol of the new dataset
    :return:
    """
    rmse_cum, mae_cum, mape_cum = 0, 0, 0
    for ticker in ticker_symbols:
        ticker_test_forecast = get_data([ticker])
        x_test_forecast = ticker_test_forecast.drop('Close', axis=1)
        y_pred_test_forecast = model.predict(x_test_forecast)
        y_pred_test_forecast = scaler.inverse_transform(np.array(y_pred_test_forecast).reshape(-1, 1))
        x_test_forecast['Close'] = y_pred_test_forecast
        y_test_forecast = ticker_test_forecast['Close']
        # forecast = predict_for_timesteps(model=model, days=30, ticker=ticker_symbols)
        plt.figure(figsize=(10, 7.5))
        plt.plot(y_test_forecast, label='Actual', color='green')
        plt.plot(x_test_forecast['Close'], label='Predicted', alpha=0.7, color='red')
        # plt.plot(forecast['Close'], label='Forecasted Value', color='blue')
        plt.title(ticker)
        plt.legend()
        plt.show()
        rmse = np.sqrt(mean_squared_error(y_test_forecast, y_pred_test_forecast))
        mae = mean_absolute_error(y_test_forecast, y_pred_test_forecast)
        mape = mean_absolute_percentage_error(y_test_forecast, y_pred_test_forecast) * 100
        rmse_cum = rmse_cum + rmse
        mae_cum = mae_cum + mae
        mape_cum = mape_cum + mape
        print('Ticker = %s \nRMSE = %f \nMAE = %f \nMAPE = %f %%' % (ticker, rmse, mae, mape))
    print('Mean RMSE = %f \nMean MAE = %f \nMean MAPE = %f %%' % (rmse_cum / len(ticker_symbols),
                                                                  mae_cum / len(ticker_symbols),
                                                                  mape_cum / len(ticker_symbols)))


def predict_for_timesteps(model: Sequential, days: int, ticker: [str]) -> pd.DataFrame:
    """
    :param model: Sequential model to predict with
    :param days: number of days to predict for
    :param ticker: list of tickers to predict for
    :return: dataframe with predicted values
    """
    predictions = pd.DataFrame
    for tick in ticker:
        ticker_data = get_data(tick)
        prediction = ticker_data.tail(days)
        prediction.reset_index(inplace=True)
        x_forecast = prediction.copy(deep=True)
        x_forecast.drop('Close', axis=1, inplace=True)
        x_forecast.drop('Date', axis=1, inplace=True)
        for day in range(0, days):
            x_day = np.array(x_forecast.iloc[day]).reshape(1, 4)
            prediction_day = scaler.inverse_transform(model.predict(x_day)).flatten()[0]
            prediction.loc[day, 'Close'] = prediction_day
            if day < days - 1:
                prediction.loc[(day + 1), 'Open'] = prediction_day
            prediction.loc[day, 'Date'] = prediction.loc[day, 'Date'] + timedelta(days=days)
    # plt.figure(figsize=(10, 7.5))
    # plt.plot(prediction['Close'], label='Predicted', alpha=0.7, color='red')
    # sns.lineplot(data=prediction, x=prediction['Date'], y=prediction['Close'], color='blue')
    # plt.title(ticker[0])
    # plt.show()
    return prediction


if __name__ == '__main__':
    main()

# resetting the options
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')
pd.reset_option('display.max_colwidth')
