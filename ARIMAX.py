import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import itertools
import statsmodels.api as sm
import seaborn as sns

sns.set()

df = pd.read_csv('datacb_corr_multiv_goog.csv', index_col='Date')
df = df[2:]
df_train = df[:int(len(df)*0.7)]
df_test = df[int(len(df)*0.7):]
history = [x for x in df_train.Money.values]
ex1 = [x for x in df_train['yand'].values]
ex2 = [x for x in df_train['usd'].values]
ex = np.transpose(np.array([ex1, ex2]))
predictions = list()

def MAPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

sm.graphics.tsa.plot_acf(history)
sm.graphics.tsa.plot_pacf(history)

plt.show()

p = range(3)
d = range(3)
q = range(3)
pdq = list(itertools.product(p, d, q))

ans = []
for comb in pdq:
    try:
        mod = sm.tsa.statespace.SARIMAX(history,
                                        order=comb,
                                        seasonal_order=(0, 0, 0, 0),
                                        exog=ex,
                                        enforce_invertibility=False)

        output = mod.fit()
        ans.append([comb, output.aic])
        print('SARIMAX {} x: AIC Calculated ={}'.format(comb, output.aic))
    except:
        continue

ans_df = pd.DataFrame(ans, columns=['pdq', 'aic'])

# Sort and return top 5 combinations
ans_df = ans_df.sort_values(by=['aic'], ascending=True)[0:5]
print(ans_df)

ex1_t = [x for x in df_test['yand'].values]
ex2_t = [x for x in df_test['usd'].values]
ex_test = np.transpose(np.array([ex1, ex2]))

# fit model
model = SARIMAX(history, exog=ex, order=(0, 2, 2), seasonal_order=(0, 0, 0, 0))
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.predict(len(df_test), exog=ex_test)
print(yhat)

mape = MAPE(df_test.Money, yhat[1:])

ind = []
test_ind = df.index[len(history):]
for i in range(0, len(df.index[:len(history)]), 6):
    ind.append(df.index[i])
for i in range(0, len(df_test), 6):
    ind.append(test_ind[i])

plt.subplots(figsize=(12, 8))
plt.plot(df.index[:len(history)+1], df.Money[:len(history)+1], label='Обучение')
plt.plot(df.index[len(history):], df_test.Money, color='green', label='Тест')
plt.plot(df.index[len(history):], yhat[1:], color='orange', label=f'Прогноз MAPE {round(mape, 2)} %')

plt.xticks(ind, rotation='horizontal')

plt.xlabel('Дата')
plt.ylabel('Млн. руб.')
plt.title('Прогнозирование данных ЦБ с помощью ARIMAX')
plt.legend()
plt.show()
plt.close()
