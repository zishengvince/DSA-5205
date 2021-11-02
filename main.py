import pandas as pd
import numpy as np
import json
import GeneticAlgorithm as ga
import LSTM
# import SVM
import MarkowitzEF as mef
import data




if __name__ == "__main__":
    # for i in range(10):
    #     data.logger.info("LSTM + Genetic Algorithm")
    #     best = ga.gaStockSelection()
    #     gene = best.gene
    #     gene_arr = ga.cutText(gene)
    #     weight_arr = np.array([ga.decodeGene(sub_gene) for sub_gene in gene_arr])
    #     max_index_list = ga.getMaxNIndex(weight_arr)
    #     weight_arr = weight_arr[max_index_list]
    #     weight_arr /= sum(weight_arr)
    #     stock_arr = data.STOCK_ARR[max_index_list]
    #     for i in range(len(stock_arr)):
    #         print((stock_arr[i],weight_arr[i]))
    #     stock_json.append(stock_arr)
    #     weight_json.append(weight_arr)
    #     performance_json.append(ga.evaluate(list(stock_arr),weight_arr))


    data.logger.info("SVM + Markowitz Efficient Frontier")
    SVM_result = ['FB.O','GOOG.O','GOOGL.O','SWKS.O','CHTR.O','SBUX.O','MSFT.O','CDW.O','ASML.O','QCOM.O']
    keys = list(set(SVM_result))

    ret_df = data.RET_TRUE_2021[keys]
    weights = mef.findBest(ret_df)
    weight_arr = np.array(weights)

    ga.evaluate(keys,weight_arr)

