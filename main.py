import pandas as pd
import numpy as np
import json
import GeneticAlgorithm as ga
import LSTM
# import SVM
import MarkowitzEF as mef
import data
import matplotlib.pyplot as plt


def plotGA():
    evaluate = [0.0811202685842475,
                0.014596082643752808,
                0.03907662825074533,
                0.041451419125390354,
                0.05280777855542134,
                0.0487986761647919,
                0.06628836750610771,
                0.03793430938592556,
                0.04653119010106193,
                0.020383633290018886]

    plt.style.use('seaborn-dark')
    plt.figure(figsize=(9, 5))
    plt.plot(evaluate)
    plt.grid(True)
    plt.ylim(0,0.15)
    plt.xlabel('genetic algorithm iteration')
    plt.ylabel('portfolio return')
    plt.title('Genetic Algorithm Result')
    plt.savefig('ga_return.png',dpi=500,bbox_inches = 'tight')
    plt.show()

def trendPlot():
    best_para_tuple = (('WDAY.O', 0.10543933054393305),
                        ('CTSH.O', 0.10209205020920502),
                        ('MRNA.O', 0.10125523012552301),
                        ('ADSK.O', 0.100418410041841),
                        ('CMCSA.O', 0.099581589958159),
                        ('LULU.O', 0.099581589958159),
                        ('NTES.O', 0.099581589958159),
                        ('ADBE.O', 0.09874476987447699),
                        ('SPLK.O', 0.09707112970711297),
                        ('XEL.O', 0.09623430962343096))
    best_ga_stocks = []
    best_ga_weights = []
    for stock, weight in best_para_tuple:
        best_ga_stocks.append(stock)
        best_ga_weights.append(weight)

    worst_para_tuple = (('FOXA.O', 0.10404624277456648),
                        ('CHKP.O', 0.10239471511147812),
                        ('MRNA.O', 0.10239471511147812),
                        ('SWKS.O', 0.10074318744838975),
                        ('ALGN.O', 0.09991742361684558),
                        ('NTES.O', 0.0990916597853014),
                        ('VRTX.O', 0.0990916597853014),
                        ('JD.O', 0.09826589595375723),
                        ('PAYX.O', 0.09744013212221304),
                        ('MTCH.O', 0.09661436829066887))
    worst_ga_stocks = []
    worst_ga_weights = []
    for stock, weight in worst_para_tuple:
        worst_ga_stocks.append(stock)
        worst_ga_weights.append(weight)

    ef_stocks = ['SBUX.O','FB.O','ASML.O','QCOM.O','MSFT.O','AAPL.O','CDW.O','NVDA.O','AMD.O','CERN.O']
    ef_weights = [0.226, 0,    0,    0.292, 0.001, 0.328, 0,    0.153, 0,    0   ]
    best_ga_port = [10000*best_ga_weights[i]/data.PRICE_TRUE[best_ga_stocks].iloc[0,i] for i in range(len(best_ga_weights))]
    best_ga_port_value = np.dot(best_ga_port,data.PRICE_TRUE[best_ga_stocks].T)
    worst_ga_port = [10000*worst_ga_weights[i]/data.PRICE_TRUE[worst_ga_stocks].iloc[0,i] for i in range(len(worst_ga_weights))]
    worst_ga_port_value = np.dot(worst_ga_port,data.PRICE_TRUE[worst_ga_stocks].T)
    ef_port = [10000*ef_weights[i]/data.PRICE_TRUE[ef_stocks].iloc[0,i] for i in range(len(ef_weights))]
    ef_port_value = np.dot(ef_port,data.PRICE_TRUE[ef_stocks].T)
    x_axis = list(range(len(ef_port_value)))
    plt.style.use('seaborn-dark')
    plt.figure(figsize=(9, 5))
    plt.plot(x_axis,best_ga_port_value,color='blue',label="best_ga_port_value")
    plt.plot(x_axis,worst_ga_port_value,color='green',label="worst_ga_port_value")
    plt.plot(x_axis,ef_port_value,color='red',label="ef_port_value")
    plt.legend()
    plt.grid(True)
    # plt.ylim(0,0.15)
    plt.xlabel('day')
    plt.ylabel('dollar')
    plt.title('Portfolio value')
    plt.savefig('port_value.png',dpi=500,bbox_inches = 'tight')
    plt.show()

if __name__ == "__main__":
    for i in range(10):
        data.logger.info("LSTM + Genetic Algorithm")
        best = ga.gaStockSelection()
        gene = best.gene
        gene_arr = ga.cutText(gene)
        weight_arr = np.array([ga.decodeGene(sub_gene) for sub_gene in gene_arr])
        max_index_list = ga.getMaxNIndex(weight_arr)
        weight_arr = weight_arr[max_index_list]
        weight_arr /= sum(weight_arr)
        stock_arr = data.STOCK_ARR[max_index_list]
        for i in range(len(stock_arr)):
            print((stock_arr[i],weight_arr[i]))


    # data.logger.info("SVM + Markowitz Efficient Frontier")
    # SVM_result = ['SBUX.O','FB.O','ASML.O','QCOM.O','MSFT.O','AAPL.O','CDW.O','NVDA.O','AMD.O','CERN.O']
    # keys = list(set(SVM_result))

    # ret_df = data.RET_PREDICT[keys]
    # cov_mat = mef.getCovMatrix(ret_df)
    # print(cov_mat)
    # cov_mat.to_csv("cov_mat.csv")
    # weights = mef.findBest(ret_df)
    # weight_arr = np.array(weights)

    # ga.evaluate(keys,weight_arr)

    # trendPlot()


