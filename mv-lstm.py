from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from math import sqrt
from scipy import stats
from numpy import concatenate
from statsmodels.tsa.stattools import grangercausalitytests
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from tensorflow import keras
from keras.layers import LSTM
from math import sqrt
from numpy import concatenate
import pandas as pd

sns.set(rc={'figure.figsize':(12,8)})

df = pd.read_csv('datacb_corr_multiv_goog.csv', index_col='Date')
# df = df[2:]
df = df.fillna(method='pad')
df.isnull().sum()

nobs = int(len(df)*0.3)

X_train, X_test = df[0:-nobs], df[-nobs:]
test = 'ssr_chi2test'
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):
    X_train = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in X_train.columns:
        for r in X_train.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=6, verbose=False)
            p_values = [round(test_result[i+1][0][test][1], 4) for i in range(6)]
            if verbose:
                print(f' Y = {r}, X ={c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            X_train.loc[r, c] = min_p_value
    X_train.columns = [var + '_x' for var in variables]
    X_train.index = [var + '_y' for var in variables]
    return X_train

grangers_causation_matrix(X_train, variables = X_train.columns)

def parse(x):
 return pd.datetime.strptime(x, '%Y %m %d')

values = df.values
# specify columns to plot
groups = [0, 1, 2, 3, 4, 5]
i = 1
# plot each column
plt.figure()
for group in groups:
 plt.subplot(len(groups), 1, i)
 plt.plot(values[:, group])
 plt.title(df.columns[group], y=0.5, loc='right')
 i += 1
# plt.show()

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

encoder = LabelEncoder()
values[:,1] = encoder.fit_transform(values[:,1])
# ensure all data is float
values = values.astype('float32')
# normalize features

##
scaler = MinMaxScaler(feature_range=(0, 1)).fit(values)
##

scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
print(reframed.head())

# split into train and test sets
values = reframed.values
Train_size = int(len(df)*0.7) - 1
train = values[:Train_size, :]
test = values[Train_size:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1] # -1, -1
test_X, test_y = test[:, :-1], test[:, -1] # test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

model = Sequential()
model.add(LSTM(500, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
model.add(LSTM(500))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
model.save('MV_lstm_long_mape_9.h5')
# model = keras.models.load_model('/content/drive/My Drive/data-mortgage/MV_lstm_long_mape_9.h5')
model = keras.models.load_model('MV_lstm_long_mape_9.h5')

yhat = model.predict(test_X, verbose=0)
n_features = 2
reframed = series_to_supervised(scaled)

train_X, train_y = train[:, :Train_size], train[:, -n_features]
test_X, test_y = test[:, :Train_size], test[:, -n_features]
print(train_X.shape, len(train_X), train_y.shape)


train_X = train_X.reshape((train_X.shape[0], 1, test_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

test_X = test_X.reshape((test_X.shape[0], 6))
# invert scaling for forecast
yhat = yhat.reshape((len(yhat), 1))
inv_yhat = concatenate((yhat, test_X[:, -2:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, -2:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

Test = df.Money[Train_size+1:]

def mape(y_true, y_model):
    y_true, y_model = np.array(y_true), np.array(y_model)
    return np.mean(np.abs((y_true - y_model) / y_true)) * 100

inv_yhat = pd.DataFrame(data={'inv_yhat':inv_yhat})
inv_yhat = inv_yhat.set_index(pd.DatetimeIndex(df.Money[Train_size+1:].index))

Test = pd.DataFrame(data={'Test':df.Money[Train_size+1:]})
Test = Test.set_index(pd.DatetimeIndex(df.Money[Train_size+1:].index))

Train = pd.DataFrame(data={'Train':df.Money[:Train_size+2]})
Train = Train.set_index(pd.DatetimeIndex(df.Money[:Train_size+2].index))

ind = []
for i in range(0, len(df.index), 6):
    ind.append(df.index[i])

mape_mv_lstm = round(mape(df.Money[Train_size+1:], inv_yhat.inv_yhat), 2)

ind = []
for i in range(0, len(df.Money.index), 6):
  ind.append(df.Money.index[i])

# after connecting with ARIMAX
# plt.plot(df.Money[:len(train)+1], color='blue', label='Train')
# plt.plot(df.Money[len(train):], color='green', label='Test')
# plt.plot(inv_yhat_new_point.index, inv_yhat_new_point.MV_STM, color='violet', label=f'MV LSTM MAPE = {mape_mv_lstm} %')
# plt.plot(yhat_new_point.index, yhat_new_point, color='orange', label=f'ARIMAX MAPE {round(mape, 2)} %')
# # plt.xticks(ind, rotation='horizontal')
# plt.title('Forecasting the volume of mortgage lending', fontsize=15)
# plt.xlabel('Date', fontsize=15)
# plt.ylabel('Million rubles', fontsize=15)
# plt.xticks(ind, fontsize=14)
# plt.yticks(fontsize=14)
# plt.legend(fontsize=15)
# plt.savefig('mv_lstm-arimax_eng.png', dpi=300)
# plt.show()



