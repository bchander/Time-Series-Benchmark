# -*- coding: utf-8 -*-
"""
Prepared by: Bhanu Chander V, Created on Fri Aug  8,  

Time series prediction of Covid confirmed cases in India between
Jan 2020  to July 2021

models compared
1. LSTM
2. Simple RNN
3. ARMA 

References: https://www.kaggle.com/code/greysky/covid-19-case-prediction-with-lstm/notebook
https://analyticsindiamag.com/complete-guide-to-sarimax-in-python-for-time-series-modeling/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, SimpleRNN

# CONFIRMED

df_confirmed = pd.read_csv(r'C:\Users\E0507737\OneDrive - Danfoss\Documents\Bhanu Work Files\OneDrive - Danfoss\Work Files\Temporary Laptop Data\Works\Implementations\Regression - bench marking\datasets\LSTM covid data/time_series_covid19_confirmed_global.csv')

df_confirmed.drop(columns=['Province/State', 'Lat', 'Long'], inplace=True)
df_confirmed = df_confirmed.groupby(['Country/Region']).sum()
df_confirmed.columns = pd.to_datetime(df_confirmed.columns)

df_confirmed_daily = df_confirmed - df_confirmed.shift(1, axis=1, fill_value=0)
df_confirmed_daily_moving = df_confirmed_daily.rolling(window=7, axis=1).mean()

# DEATHS

df_deaths = pd.read_csv(r'C:\Users\E0507737\OneDrive - Danfoss\Documents\Bhanu Work Files\OneDrive - Danfoss\Work Files\Temporary Laptop Data\Works\Implementations\Regression - bench marking\datasets\LSTM covid data/time_series_covid19_deaths_global.csv')

df_deaths.drop(columns=['Province/State', 'Lat', 'Long'], inplace=True)
df_deaths = df_deaths.groupby(['Country/Region']).sum()
df_deaths.columns = pd.to_datetime(df_deaths.columns)

df_deaths_daily = df_deaths - df_deaths.shift(1, axis=1, fill_value=0)
df_deaths_daily_moving = df_deaths_daily.rolling(window=7, axis=1).mean()

# RECOVERED

df_recovered = pd.read_csv(r'C:\Users\E0507737\OneDrive - Danfoss\Documents\Bhanu Work Files\OneDrive - Danfoss\Work Files\Temporary Laptop Data\Works\Implementations\Regression - bench marking\datasets\LSTM covid data/time_series_covid19_recovered_global.csv')

df_recovered.drop(columns=['Province/State', 'Lat', 'Long'], inplace=True)
df_recovered = df_recovered.groupby(['Country/Region']).sum()
df_recovered.columns = pd.to_datetime(df_recovered.columns)

df_recovered_daily = df_recovered - df_recovered.shift(1, axis=1, fill_value=0)
df_recovered_daily_moving = df_recovered_daily.rolling(window=7, axis=1).mean()

mpl.rcParams['figure.dpi'] = 300
plt.figure(figsize=(16, 6))

plt.xlabel('Date', fontsize=16)
plt.ylabel('Cases', fontsize=16)
plt.title('Covid-19 confirmed cases (India, US, Australia)', fontsize=16)
plt.plot(df_confirmed_daily_moving.loc['India'])
plt.plot(df_confirmed_daily_moving.loc['US'])
plt.plot(df_confirmed_daily_moving.loc['Australia'])
plt.legend(['India', 'US', 'Australia'])
plt.show()

country = 'India'
nfeatures = 1
nsteps = 7

feature_1 = df_confirmed_daily.loc[country]

dataset = np.column_stack([feature_1])

data_len = len(dataset[:, 0])
train_len = int(0.8 * data_len)
test_len = data_len - train_len

train_data = dataset[:train_len, :]
test_data = dataset[train_len:, :]

arima_data = dataset


# Feature scaling

scaler = MinMaxScaler(feature_range=(0, 1))

train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

#train_data2 = scaler.fit_transform(train_data2)
#test_data2 = scaler.transform(test_data2)


train_x = np.array([train_data[(i-nsteps):i, :] for i in range(nsteps, train_len)])
train_y = np.array([train_data[i, 0] for i in range(nsteps, train_len)])

test_x = np.array([test_data[(i-nsteps):i, :] for i in range(nsteps, test_len)])
test_y = np.array([test_data[i, 0] for i in range(nsteps, test_len)])

'''................LSTM..................'''

model_LSTM = Sequential([
    LSTM(30, input_shape=(nsteps, nfeatures), return_sequences=True),
    LSTM(units=30),
    Dense(units=5),
    Dense(units=nfeatures)
])

model_LSTM.compile(optimizer='adam', loss='mean_squared_error')

model_LSTM.fit(x=train_x, y=train_y, batch_size=1, epochs=10)

predictions_LSTM = model_LSTM.predict(test_x)
predictions_LSTM = scaler.inverse_transform(predictions_LSTM)


import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300
plt.figure(figsize=(16, 8))
rcParams['font.size'] = 14
plt.title(f'Covid-19 confirmed cases of {country} with LSTM', fontsize=18)
time_series = feature_1
train_time_series = time_series.iloc[640:train_len]
#train_time_series = time_series.iloc[640:train_len]
test_time_series = time_series.iloc[train_len-1:]
pred_time_series = pd.Series(data=predictions_LSTM[:, 0], index=test_time_series.index[nsteps+1:])
plt.plot(train_time_series, linewidth = 3)
plt.plot(test_time_series, linewidth = 3)
plt.plot(pred_time_series, linewidth = 3)
plt.legend(['train', 'test', 'pred'], fontsize=16)
plt.show()


print(f'Prediction of tomorrow is {int(predictions[-1, 0])}')


'''................RNN..................'''

model_RNN = Sequential([
    SimpleRNN(30, input_shape=(nsteps, nfeatures), return_sequences=True),
    SimpleRNN(units=30),
    Dense(units=25),
    Dense(units=nfeatures)
])

model_RNN.compile(optimizer='adam', loss='mean_squared_error')

model_RNN.fit(x=train_x, y=train_y, batch_size=1, epochs=5)

predictions_RNN = model_RNN.predict(test_x)
predictions_RNN = scaler.inverse_transform(predictions_RNN)

mpl.rcParams['figure.dpi'] = 300
plt.figure(figsize=(16, 8))
rcParams['font.size'] = 14
plt.title(f'Covid-19 confirmed cases of {country} with RNN', fontsize=18)
time_series = feature_1
train_time_series = time_series.iloc[640:train_len]
test_time_series = time_series.iloc[train_len-1:]
pred_time_series = pd.Series(data=predictions_RNN[:, 0], index=test_time_series.index[nsteps+1:])
plt.plot(train_time_series, linewidth = 3)
plt.plot(test_time_series, linewidth = 3)
plt.plot(pred_time_series, linewidth = 3)
plt.legend(['train', 'test', 'pred'], fontsize=16)
plt.show()


'''..............ARIMA................'''

'''SARIMAX(Seasonal Auto-Regressive Integrated Moving Average with 
eXogenous factors) is an updated version of the ARIMA model. ARIMA 
includes an autoregressive integrated moving average, while SARIMAX 
includes seasonal effects and eXogenous factors with the autoregressive 
and moving average component in the model. Therefore, we can say SARIMAX 
is a seasonal equivalent model like SARIMA and Auto ARIMA.'''


# Seasonal Decompose to check staionarity 

from statsmodels.tsa.seasonal import seasonal_decompose
from pylab import rcParams

rcParams['figure.figsize'] = 22, 12
rcParams['font.size'] = 18
decompose_data = seasonal_decompose(feature_1, model="additive")
fig = decompose_data.plot()
plt.show()

#We can also extract the plot of the season for proper visualization of the seasonality.
seasonality = decompose_data.seasonal
seasonality.plot(color='green')

#To perform forecasting using the ARIMA model, we required a stationary 
#time series. Stationary time series is a time series that is unaffected 
#by these four components. Most often, it happens when the data is non-stationary 
#the predictions we get from the ARIMA model are worse or not that accurate.

#If the data is not stationary, we can do one thing: either make the data stationary or use the SARIMAX model.

#To know more about the time series stationarity, we can perform the 
#ADfuller test, a test based on hypothesis, where if the p-value is less than 0.05, 
#then we can consider the time series is stationary, and if the P-value is greater 
#than 0.05, then the time series is non-stationary.

from statsmodels.tsa.stattools import adfuller

dftest = adfuller(feature_1, autolag = 'AIC')
print("1. ADF : ",dftest[0])
print("2. P-Value : ", dftest[1])
print("3. Num Of Lags : ", dftest[2])
print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
print("5. Critical Values :")
for key, val in dftest[4].items():
    print("\t",key, ": ", val)
    
# We can make the time series stationary with differencing methods. In this case, we are going ahead with the rolling mean differencing methods.
# Often with the data where the effect of seasonality is in excess, we use the rolling mean differencing.
#ARIMA model is useful in the cases where the time series is non-stationary. And the differencing is required to make the time series stationary. 
#Since the p-alue is less than 0.05 we assume the time series is stationary and we don't need differencing, ARMA should work but let's try both ARMA and ARIMA 

#ARIMA model is generally denoted as ARIMA(p, d, q) and parameter p, d, q are defined as follow:
#p: the lag order or the number of time lag of autoregressive model AR(p)
#d: degree of differencing or the number of times the data have had subtracted with past value
#q: the order of moving average model MA(q)

# AR: an AutoregRegressive model which represents a type of random process. The output of the model is linearly dependent on its own previous value i.e. some number of lagged data points or the number of past observations
# MA: a Moving Average model which output is dependent linearly on the current and various past observations of a stochastic term
# I: integrated here means the differencing step to generate stationary time series data, i.e. removing the seasonal and trend components

import statsmodels.tsa as sm
model_arima = sm.arima_model.ARIMA(arima_data, order =(2, 0, 2))

#model_sarimax = sm.tsa.statespace.SARIMAX(order=(1, 1, 1),seasonal_order=(1,1,1,12))

history_arima = model_arima.fit()

#history_arima.summary()

predictions_arima = model_arima.predict(test_data)
#predictions_arima = scaler.inverse_transform(predictions_arima)

mpl.rcParams['figure.dpi'] = 300
plt.figure(figsize=(16, 8))
rcParams['font.size'] = 14
plt.title(f'Covid-19 confirmed cases of {country} with ARIMA', fontsize=18)
time_series = feature_1
train_time_series = time_series.iloc[640:train_len]
test_time_series = time_series.iloc[train_len-1:]
pred_time_series = pd.Series(data=predictions_arima[train_len-1:], index=test_time_series.index)
plt.plot(train_time_series, linewidth = 3)
plt.plot(test_time_series, linewidth = 3)
plt.plot(pred_time_series, linewidth = 3)
plt.legend(['train', 'test', 'pred'], fontsize=16)
plt.show()

'''.....EValuation Metrics............'''

from sklearn.metrics import mean_squared_error as mse, r2_score
from tabulate import tabulate

test_time_series = time_series.iloc[train_len+nsteps:]

rmse_LSTM_model = mse(predictions_LSTM, test_time_series, squared=True)
r2score_LSTM_model = r2_score(predictions_LSTM, test_time_series)

rmse_RNN_model = mse(predictions_RNN, test_time_series, squared=True)
r2score_RNN_model = r2_score(predictions_RNN, test_time_series)

rmse_arima_model = mse(predictions_arima[747:,], test_time_series, squared=True)
r2score_arima_model = r2_score(predictions_arima[747:,], test_time_series)

data = [
        ['LSTM', round(rmse_LSTM_model,3), round(r2score_LSTM_model,3)],
        ['RNN', round(rmse_RNN_model,3), round(r2score_RNN_model,3)],
        ['ARIMA', round(rmse_arima_model,3), round(r2score_arima_model,3)]
        ]

head = ["Model", 'RMSE', 'r2 Score']

metric_table = tabulate(data, headers=head, tablefmt = "grid")

print(metric_table)

