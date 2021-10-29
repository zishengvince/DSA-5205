import numpy as np
import pandas as pd
from datetime import date
import numpy.random as npr
import matplotlib.pyplot as plt
from pylab import mpl
import scipy.optimize as sco
plt.rcParams['font.family'] = 'Arial Unicode MS'
plt.rcParams['axes.unicode_minus']=False


def getAnnualReturn(_ret_df):
    # columns is the stock code: 10 columns
    # index is the date: a month
    return _ret_df.mean()*252

def getCovMatrix(_ret_df):
    return _ret_df.cov()*252


def portSimulation(_ret_annual, _cov_matrix, _stock_amt = 10, _port_count = 50000):
    assert _ret_annual.shape[1] == _stock_amt, "return dataframe shape error"
    assert _cov_matrix.shape[1] == _stock_amt, "covariance matrix shape error"
    port_rets = []
    port_vols = []
    port_wght = []
    sharpe_ratio = []

    count = 0
    while count < _port_count:
        weights = np.random.random(_stock_amt)#权重
        weights = weights/(np.sum(weights))#权重归一化
        port_wght.append(weights)
        returns = np.dot(weights, _ret_annual)#投资组合收益率
        volatility = np.sqrt(np.dot(weights.T,np.dot(_cov_matrix,weights)))#投资组合波动率
        port_rets.append(returns)
        port_vols.append(volatility)
        sharpe = returns/volatility#计算夏普比率
        sharpe_ratio.append(sharpe)
        count += 1
    port_rets_arr = np.array(port_rets)
    port_vols_arr = np.array(port_vols)
    return port_rets_arr, port_vols_arr, sharpe_ratio


def plotSharpe(_port_rets_arr, _port_vols_arr, _sharpe_ratio):
    plt.style.use('seaborn-dark')
    plt.figure(figsize=(9, 5))
    plt.scatter(_port_vols_arr, _port_rets_arr, c=_sharpe_ratio,cmap='RdYlGn', edgecolors='black',marker='o')
    plt.grid(True)
    plt.xlabel('expected volatility')
    plt.ylabel('expected return')
    plt.colorbar(label='Sharpe ratio')
    plt.savefig('sharpe_ratio.png',dpi=500,bbox_inches = 'tight')
    plt.show()
    return



def findBest(_ret_df, _stock_amt = 10, _port_count = 50000):

    ret_annual = getAnnualReturn(_ret_df)
    cov_matrix = getCovMatrix(_ret_df)

    port_rets_arr, port_vols_arr, sharpe_ratio = portSimulation(ret_annual, cov_matrix, _stock_amt, _port_count)
    plotSharpe(port_rets_arr, port_vols_arr, sharpe_ratio)

    def statistics(weights):
        weights = np.array(weights)
        pret = np.sum(_ret_df.mean() * weights) * 252
        pvol = np.sqrt(np.dot(weights.T, np.dot(_ret_df.cov() * 252, weights)))
        return np.array([pret, pvol, pret / pvol])

    def min_func_sharpe(weights):
        return -statistics(weights)[2]

    bnds = tuple((0, 1) for x in range(_stock_amt))
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    opts = sco.minimize(min_func_sharpe, _stock_amt * [1. / _stock_amt,], method='SLSQP',  bounds=bnds, constraints=cons)
    print(opts['x'].round(3)) #得到各股票权重
    print(statistics(opts['x']).round(3)) #得到投资组合预期收益率、预期波动率以及夏普比率
