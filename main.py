import pandas as pd
import numpy as np
import GeneticAlgorithm as ga
import LSTM
import SVM
import MarkowitzEF as mef
import data



if __name__ == "__main__":
    # best = ga.gaStockSelection()
    # gene = best.gene
    # gene_arr = ga.cutText(gene)
    # weight_arr = np.array([ga.decodeGene(sub_gene) for sub_gene in gene_arr])
    # max_index_list = ga.getMaxNIndex(weight_arr)
    # weight_arr = weight_arr[max_index_list]
    # weight_arr /= sum(weight_arr)
    # stock_arr = data.STOCK_ARR[max_index_list]
    # for i in range(len(stock_arr)):
    #     print((stock_arr[i],weight_arr[i]))

    mef.findBest(data.RET_DF_10)
