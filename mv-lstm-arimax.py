import pandas as pd
import tensorflow as tf
from tensorflow import keras
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import sqrt
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import grangercausalitytests
from math import sqrt
from numpy import concatenate
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools
import statsmodels.api as sm

sns.set(rc={'figure.figsize':(12,8)})

'''MV LSTM'''

df = pd.read_csv('lags.csv', index_col='Date')
df = df[2:]

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
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
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

values = df.values
scaler = MinMaxScaler(feature_range=(0, 1)).fit(values)

scaled = scaler.fit_transform(values)
reframed = series_to_supervised(scaled, 1, 1)

# split into train and test sets
values = reframed.values
Train_size = int(len(df)*0.7)
train = values[:Train_size, :]
# test = values[Train_size:, :]
test = values[(Train_size-1):len(values), :]
# split into input and outputs
train_X, train_y = train[:, [0, 1, 2, 4, 5]], train[:, 3]
test_X, test_y = test[:, [0, 1, 2, 4, 5]], test[:, 3]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


df_ = pd.read_csv('cb_corr_econom_without_lags.csv', index_col='Date')

nobs = int(len(df_)*0.3)
X_train_, X_test_ = df_[0:-nobs], df[-nobs:]
test = 'ssr_chi2test'
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False, maxlag=6):
    X_train = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in X_train.columns:
        for r in X_train.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1], 4) for i in range(maxlag)]
            if verbose:
                print(f' Y = {r}, X ={c}, P Values = {p_values}')
            min_p_value = np.average(p_values)
            X_train.loc[r, c] = min_p_value
    X_train.columns = [var + '_x' for var in variables]
    X_train.index = [var + '_y' for var in variables]
    return X_train

grangers_causation_matrix(X_train_, variables = X_train_.columns)
model = Sequential()
model.add(LSTM(500, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
model.add(LSTM(500))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
model.fit(train_X, train_y, epochs=50, batch_size=72, verbose=2, validation_split=0.3)
# model.save('MV_lstm_g.h5')
model = keras.models.load_model('MV_lstm_g.h5')

yhat = model.predict(test_X)
df_val = df.values
check_test_X = df_val[Train_size:]
yhat = yhat.reshape((len(yhat), 1))
inv_yhat = np.concatenate((yhat, check_test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, check_test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

Test = df.Money[Train_size:]

def mape(y_true, y_model):
    y_true, y_model = np.array(y_true), np.array(y_model)
    return np.mean(np.abs((y_true - y_model) / y_true)) * 100

inv_yhat = pd.DataFrame(data={'inv_yhat':inv_yhat})
inv_yhat = inv_yhat.set_index(pd.DatetimeIndex(df.Money[Train_size:].index))

Test = pd.DataFrame(data={'Test':df.Money[Train_size:]})
Test = Test.set_index(pd.DatetimeIndex(df.Money[Train_size:].index))

Train = pd.DataFrame(data={'Train':df.Money[:Train_size+1]})
Train = Train.set_index(pd.DatetimeIndex(df.Money[:Train_size+1].index))
mape_mv_lstm = round(mape(Test.Test, inv_yhat.inv_yhat), 2)

'''ARIMAX'''
df_arimax = pd.read_csv('lags.csv', index_col='Date')
df_train = df_arimax[:int(len(df_arimax)*0.7)]
df_test = df_arimax[int(len(df_arimax)*0.7):]
history = [x for x in df_train.Money.values]
ex1 = [x for x in df_train['yand'].values]
ex2 = [x for x in df_train['usd_5'].values]
ex = np.transpose(np.array([ex1, ex2]))
predictions = list()
p = range(5)
d = range(5)
q = range(3)
pdq = list(itertools.product(p, d, q))

ex1_t = [x for x in df_test['yand'].values]
ex2_t = [x for x in df_test['usd_5'].values]
ex_test = np.transpose(np.array([ex1, ex2]))

# ans = []
# for comb in pdq:
#     try:
#         mod = sm.tsa.statespace.SARIMAX(history,
#                                                 order=comb,
#                                                 seasonal_order=(0, 0, 0, 0),
#                                                 exog=ex,
#                                                 enforce_invertibility=False)


#         output = mod.fit()
#         ans.append([comb, output.bic])
#         print('SARIMAX {} x: AIC Calculated ={}'.format(comb, output.aic))
#     except:
#          continue

#     # Find the parameters with minimal BIC value

#     # Convert into dataframe
# ans_df = pd.DataFrame(ans, columns=['pdq', 'aic'])

#     # Sort and return top 5 combinations
# ans_df = ans_df.sort_values(by=['aic'],ascending=True)[0:5]

# ans_df

for t in range(len(df_test)):
    model = SARIMAX(history, exog=ex, order=(3, 1, 2), seasonal_order=(0, 0, 0, 0))
    model_fit = model.fit()
    exog1 = []
    exog1.append(df_test['yand'].values[t])
    exog2 = []
    exog2.append(df_test['usd_5'].values[t])
    exog = np.transpose(np.array([exog1, exog2]))
    # output = model_fit.predict(len(df_test), exog=exog)
    output = model_fit.forecast(exog=exog)
    yhat = output[0]
    predictions.append(yhat)
    obs = df_test['Money'].values[t]
    history.append(obs)
    ex = np.vstack((ex, exog))
    # print('predicted=%f, expected=%f' % (yhat, obs))

mape_arimax = mape(df_test.Money, predictions)

inv_yhat_new_point = [df.Money[len(train)-1]]
for i in range(len(inv_yhat)):
  inv_yhat_new_point.append(inv_yhat.inv_yhat[i])

inv_yhat_new_point = pd.DataFrame({'MV_STM': inv_yhat_new_point})
inv_yhat_new_point = inv_yhat_new_point.set_index(df.Money[len(train)-1:].index)

yhat_new_point = [df.Money[len(train)-1]]
for i in range(len(predictions)):
  yhat_new_point.append(predictions[i])

yhat_new_point = pd.DataFrame({'ARIMAX': yhat_new_point})
yhat_new_point = yhat_new_point.set_index(df.Money[len(train)-1:].index)

plt.plot(df.Money[:len(train)], color='blue', label='Train')
plt.plot(df.Money[len(train)-1:], color='green', label='Test')
plt.plot(inv_yhat_new_point.MV_STM, color='magenta', label=f'MV LSTM MAPE = {mape_mv_lstm} %')
plt.plot(yhat_new_point.ARIMAX, color='m', label=f'ARIMAX MAPE {round(mape_arimax, 2)} %')
plt.title('Forecasting the volume of mortgage lending', fontsize=15)
plt.xlabel('Date', fontsize=15)
plt.ylabel('Million rubles', fontsize=15)
plt.xticks(df_arimax.index[0:-1:6], fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=15)
# plt.savefig('arimax-mv-lstm.png', dpi=300)
plt.show()