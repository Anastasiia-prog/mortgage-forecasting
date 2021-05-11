import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from scipy import stats

sns.set(rc={'figure.figsize': (11, 8)})

cb_data = pd.read_csv('data_cb_jan.csv', index_col='Date')

def move_corr(cb_data, google):
    corr, p = spearmanr(cb_data, yand)
    plt_lst = [corr]
    print('Дата ЦБ c', cb_data.index[0], 'по', cb_data.index[-1])
    print('Дата Яндекса c', yand.index[0], 'по', yand.index[-1])
    print('Корреляция равна', corr)
    for i in range(1, int(len(yand) // 2 + 1), 1):
        corr, p = spearmanr(cb_data[i:], yand[:-i])
        plt_lst.append(corr)
        print('Дата ЦБ c', cb_data.index[i], 'по', cb_data.index[-1])
        print('Дата Яндекса c', yand.index[0], 'по', yand.index[-i - 1])
        print('Корреляция равна', corr)
        print('--------')

    plt.plot(plt_lst)
    plt.title(f'График корреляционных сдвигов данных ЦБ и {input()}')
    plt.xlabel('Лаг')
    plt.ylabel('Значение корреляции')
    plt.show()

df = pd.read_csv('corr_between_data-Copy1.csv', index_col='Date')
yand = df.drop(['data_cb', 'mortg', 'google', 'offer'], axis=1)
mortg = pd.read_csv('mortg_month_jan.csv', index_col='date')
cb_data = cb_data[4:]

move_corr(cb_data, yand)
move_corr(cb_data, mortg)

cb_econ = pd.read_csv('cb_corr_econom_without_lags.csv', index_col='Date')
move_corr(cb_econ.Money, cb_econ.key_rate)
move_corr(cb_econ.Money, cb_econ.IPC)
move_corr(cb_econ.Money, cb_econ.diff_of_costs)
move_corr(cb_econ.Money, cb_econ.usd)
move_corr(cb_econ.Money, cb_econ.euro)
move_corr(cb_econ.Money, cb_econ.crude_oil)

matr = pd.read_csv('lags_for_disser.csv', index_col='Date')
matr_ = matr.rename({'Money': 'ЦБ', 'key_rate_4': 'Кл.ставка', 'IPC_0': 'ИПЦ', 'usd_5': 'Курс дол/руб',
                            'crude_oil_5': 'Курс нефти', 'mortg':'Яндекс_ипотека', 'yand':'Яндекс',
                                       'diff_of_costs_6': 'Сред.цены', 'euro_0':'Курс евро/руб'}, axis=1)
cor_matr = matr_.corr(method='spearman')
print('corr', cor_matr)

plt.figure(figsize=(12, 8))
plt.title('Корреляционная матрица данных', fontsize=15)
sns.heatmap(cor_matr, annot=True)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.show()
