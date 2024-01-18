import pandas as pd
import yfinance as yf
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.src.layers import LSTM, Dense
from IPython.display import display

# making output from pandas more readable
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

# download data
ticker = yf.Ticker('GOOG')
data = ticker.history(period='1y', interval='1d')
# print(data.columns)
sns.lineplot(data=data, x='Date', y='Close')
plt.show()
# print(data.describe())

# split data into test and train
x = data.drop('Close', axis=1)
y = data['Close']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# normalize data
scaler = MinMaxScaler()
y_train = scaler.fit_transform(np.array(y_train).reshape(-1, 1))
y_test = scaler.transform(np.array(y_test).reshape(-1, 1))

# make model and train it
model = keras.models.Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=100)

# evaluate the model
test_loss = model.evaluate(x_test, y_test)
print('Test loss = %f' % np.sqrt(test_loss))

# predicting values and visualizing them
y_pred = model.predict(x_test)
y_pred = scaler.inverse_transform(np.array(y_pred).reshape(-1, 1))
y_test = scaler.inverse_transform(np.array(y_test).reshape(-1, 1))
y_train = scaler.inverse_transform(np.array(y_train).reshape(-1, 1))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print('RMSE = %f, MAE = %f' % (rmse, mae))
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()

# resetting the options
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')
pd.reset_option('display.max_colwidth')
