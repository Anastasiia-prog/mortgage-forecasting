%pylab inline
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import warnings
from itertools import product
%matplotlib inline
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame, concat
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from tensorflow import keras
from datetime import datetime
from google.colab import drive

drive.mount('/content/drive')

sns.set(rc={'figure.figsize':(12,8)})

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

'''BASELINE'''

df = pd.read_csv('data_cb_jan.csv',',', index_col=['Date'], parse_dates=['Date'], dayfirst=True, encoding='cp1251')
df = df[8:]
values = DataFrame(df.values)
dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
X = dataframe.values
train_size = int(len(X) * 0.7)
Train, Test = X[1:train_size], X[train_size:]
train_X, train_y = Train[:,0], Train[:,1]
test_X, test_y = Test[:,0], Test[:,1]

def model_persistence(x):
    return x


predictions_base = list()
for x in test_X:
    yhat = model_persistence(x)
    predictions_base.append(yhat)
test_score = mean_squared_error(test_y, predictions_base)
predictions_base = pd.DataFrame(data={'baseline':predictions_base})
test = df[int(len(df)*0.7):]
predictions_base = predictions_base.set_index(pd.DatetimeIndex(test.index))
mape_base = round(mean_absolute_percentage_error(test, predictions_base.baseline), 2)

'''ARIMA'''
train = df[:int(len(df)*0.7)]
test = df[int(len(df)*0.7):]
X = train.Money
result = adfuller(X)
if  result[1] > 0.05:
  d = 1
else:
  d = 0

# sm.graphics.tsa.plot_acf(train)
# sm.graphics.tsa.plot_pacf(train)
# plt.show()

ps = [0, 1, 4, 9, 12, 13]
qs = range(0, 5)

parameters = product(ps, qs)
parameters_list = list(parameters)
print ("Number of analysed models:", len(parameters_list))

# for parameter in parameters_list:
#     mod = sm.tsa.ARMA(train.Money.dropna(), order=(parameter[0], parameter[1]))
#     res = mod.fit();
#     print("Order: {}\nBIC: {}\nAIC: {}\n".format((parameter[0], parameter[1]), res.bic, res.aic))

X = df.values
size = int(len(X) * 0.7)
trainn, testt = X[0:size], X[size:len(X)]
history = [x for x in trainn]
predictions = []
for t in range(len(testt)):
    model = sm.tsa.ARIMA(history, order=(1,d,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = testt[t]
    history.append(obs)
    # print('predicted=%f, expected=%f' % (yhat, obs))
pred = []
for i in range(len(predictions)):
    for j in range(len(predictions[i])):
        pred.append(predictions[i][j])

pred_arima = pd.DataFrame(data={'pred':pred})
pred_arima = pred_arima.set_index(pd.DatetimeIndex(test.index))
mape_arima = round(mean_absolute_percentage_error(test, pred_arima), 2)


'''LSTM'''
size = int(len(df) * 0.7)
df_val = df.values
df_val = df_val.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
df_val = scaler.fit_transform(df_val)
w_len = 1
df_train, df_test = df_val[0:size, :], df_val[(size-w_len):len(df_val), :]
np.random.seed(7)

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        b = dataset[i+look_back, 0]
        dataY.append(b)
    return np.array(dataX), np.array(dataY)

trainX, trainY = create_dataset(df_train, look_back=w_len)
testX, testY = create_dataset(df_test, look_back=w_len)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

BATCH_SIZE = 72

model = Sequential()
model.add(LSTM(500, input_shape=(1, w_len), return_sequences=True))
model.add(LSTM(500))
model.add(Dense(1))
#adam = keras.optimizers.Adam(lr=4e-5)
model.compile(loss='mae', optimizer='adam')
model.fit(trainX, trainY, epochs = 50, batch_size=BATCH_SIZE, verbose=2, validation_split=0.3)
# model.save('lstm_01_06_2021.h5')
model = keras.models.load_model('lstm_01_06_2021.h5')

predictions_lstm = list()
for t in range(len(testX)):
    X = testX[t]
    yhat = model.predict(np.reshape(X, (1,1,w_len)))
    predictions_lstm.append(yhat[0])
    expected = testX[t][0]

testPredict = scaler.inverse_transform([np.concatenate(predictions_lstm)])
testY = np.hstack(testY)
testY = scaler.inverse_transform([np.concatenate([np.expand_dims(i,axis=0) for i in testY])])

predicted_lstm = pd.DataFrame(testPredict[0])
test_df = pd.DataFrame(testY[0])
mape_lstm = round(mean_absolute_percentage_error(testY, testPredict), 2)


'''Transfer Learning'''
df_aux = pd.read_csv('/content/drive/My Drive/data-mortgage/usd.csv')
df_aux = df_aux.fillna(df_aux.mean())

df_aux['Date'] = pd.to_datetime(df_aux['Date'])
df_aux = df_aux.resample('D', on='Date').sum()

size_aux = int(len(df_aux) * 0.7)
df_val_aux = df_aux.Close
df_val_aux = df_val_aux.astype('float32')
scaler_aux = MinMaxScaler(feature_range=(0, 1))
df_val_aux = np.array([df_val_aux]).reshape(-1, 1)
df_val_aux = scaler_aux.fit_transform(df_val_aux)
w_len = 1
df_train_aux, df_test_aux = df_val_aux[0:size_aux, :], df_val_aux[(size_aux-w_len):len(df_val_aux), :]

trainX_aux, trainY_aux = create_dataset(df_train_aux, look_back=w_len)
testX_aux, testY_aux = create_dataset(df_test_aux, look_back=w_len)

trainX_aux = np.reshape(trainX_aux, (trainX_aux.shape[0], 1, trainX_aux.shape[1]))
testX_aux = np.reshape(testX_aux, (testX_aux.shape[0], 1, testX_aux.shape[1]))

BATCH_SIZE = 72

model_aux = Sequential()
model_aux.add(LSTM(500, input_shape=(1, w_len), return_sequences=True))
model_aux.add(LSTM(500))
model_aux.add(Dense(1))
model_aux.compile(loss='mae', optimizer='adam')
model_aux.fit(trainX_aux, trainY_aux, epochs = 50, batch_size=BATCH_SIZE, verbose=2, validation_split=0.3)


df = pd.read_csv('data_cb_jan.csv', index_col='Date')
df = df[8:]
size = int(len(df) * 0.7)
df_val = df.values
df_val = df_val.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
df_val = scaler.fit_transform(df_val)
w_len = 1
df_train, df_test = df_val[0:size, :], df_val[(size-w_len):len(df_val), :]

trainX, trainY = create_dataset(df_train, look_back=w_len)
testX, testY = create_dataset(df_test, look_back=w_len)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

BATCH_SIZE = 72

model = Sequential()
model.add(LSTM(500, input_shape=(1, w_len), return_sequences=True))
model.add(LSTM(500))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

model.set_weights(weights = model_aux.get_weights())

model.fit(trainX, trainY, epochs = 50, batch_size=BATCH_SIZE, verbose=2, validation_split=0.3)
# model.save('transfer_02_06_2021.h5')
model = keras.models.load_model('transfer_02_06_2021.h5')

predictions_transfer = list()
for t in range(len(testX)):
    X = testX[t]
    yhat = model.predict(np.reshape(X, (1,1,w_len)))
    predictions_transfer.append(yhat[0])
    expected = testX[t][0]

testPredict = scaler.inverse_transform([np.concatenate(predictions_transfer)])
testY = np.hstack(testY)
testY = scaler.inverse_transform([np.concatenate([np.expand_dims(i,axis=0) for i in testY])])

predicted_transfer = pd.DataFrame(testPredict[0])
mape_transfer = round(mean_absolute_percentage_error(testY, testPredict), 2)


predict_s_new_point = [df.Money[len(df_train)-1]]
for i in range(len(predictions)):
    predict_s_new_point.append(predictions[i])

predict_s_new_point = pd.DataFrame({'ARIMA': predict_s_new_point})
predict_s_new_point = predict_s_new_point.set_index(df.Money[len(df_train)-1:].index)

predictions_new_point = [df.Money[len(df_train)-1]]
for i in range(len(predictions_base)):
    predictions_new_point.append(predictions_base.values[i])

predictions_new_point = pd.DataFrame({'Baseline': predictions_new_point})
predictions_new_point = predictions_new_point.set_index(df.Money[len(df_train)-1:].index)

predictions_noTrans_new_point = [df.Money[len(df_train)-1]]
for i in range(len(predicted_lstm)):
    predictions_noTrans_new_point.append(predicted_lstm.values[i])

predictions_noTrans_new_point = pd.DataFrame({'LSTM': predictions_noTrans_new_point})
predictions_noTrans_new_point = predictions_noTrans_new_point.set_index(df.Money[len(df_train)-1:].index)

predictions_withTrans_new_point = [df.Money[len(df_train)-1]]
for i in range(len(predicted_transfer)):
    predictions_withTrans_new_point.append(predicted_transfer.values[i])

predictions_withTrans_new_point = pd.DataFrame({'LSTM': predictions_withTrans_new_point})
predictions_withTrans_new_point = predictions_withTrans_new_point.set_index(df.Money[len(df_train)-1:].index)

plt.plot(df.Money[:len(trainn)], color='blue', label='Train')
plt.plot(df[len(trainn)-1:], color='green', label='Test')

plt.plot(predictions_noTrans_new_point, color='orange', label=f'LSTM MAPE = {mape_lstm} %')
plt.plot(predictions_withTrans_new_point, color='black', label=f'Transfer Learning MAPE = {mape_transfer} %')


plt.plot(predict_s_new_point, color='brown',  label=f"ARIMA MAPE = {mape_arima} %")
plt.plot(predictions_new_point, color='violet', label=f'Baseline MAPE = {mape_base} %')

plt.yticks(fontsize=14)
plt.xticks(df.index[0:-1:6])

plt.title('Forecasting the volume of mortgage lending', fontsize=15)
plt.xlabel('Date', fontsize=15)
plt.ylabel('Million rubles', fontsize=15)
plt.legend(fontsize=15)
# plt.savefig('arima-base-lstm-transfer-eng.png', dpi=300)