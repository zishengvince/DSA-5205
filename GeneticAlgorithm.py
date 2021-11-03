import re
import heapq
import numpy as np
import GAClass
import params
import data

def decodeGene(_sub_gene):
    assert isinstance(_sub_gene, str), "_gene should be a string"
    # sign = 1 if _sub_gene[0] == '0' else -1
    # amt = _sub_gene[1:]
    sign = 1
    amt = _sub_gene
    return float(int(amt,2))*sign


def cutText(_text):
    text_arr = re.findall('.{'+str(params.stock_gene_len)+'}', _text)
    text_arr.append(_text[(len(text_arr)*params.stock_gene_len):])
    return text_arr[0:-1]


def getMaxNIndex(_arr, n = 10):
    # _arr = list(abs(_arr))
    _arr = list(_arr)
    max_index_list = []
    Inf = 0
    i = 0
    while i < n:
        max_index_list.append(_arr.index(max(_arr)))
        _arr[_arr.index(max(_arr))] = Inf
        i += 1
    return max_index_list


def calculatePE(_weight_arr):
    max_index_list = getMaxNIndex(_weight_arr)
    _weight_arr = _weight_arr[max_index_list]
    _weight_arr /= sum(_weight_arr)
    stock_arr = data.STOCK_ARR[max_index_list]
    stock_list = list(stock_arr)

    # ret_df = data.RET_PREDICT[stock_list]
    ret_df = data.accurate_return[stock_list]
    annual_ret = ret_df.mean()*252
    cov_ret = ret_df.cov()*252
    returns = np.dot(_weight_arr, annual_ret)
    volatility = np.sqrt(np.dot(_weight_arr.T,np.dot(cov_ret,_weight_arr)))

    return returns/volatility


def matchFun(_unit):
    gene = _unit.gene
    gene_arr = cutText(gene)
    weight_arr = np.array([decodeGene(sub_gene) for sub_gene in gene_arr])

    return calculatePE(weight_arr)


def gaStockSelection(_cross_rate = params.cross_rate,
                    _mutation_rate = params.mutation_rate,
                    _unit_amount = params.unit_amount,
                    _gene_length = params.gene_length,
                    _generation = params.generation):

    ga_population = GAClass.GAList(_cross_rate, _mutation_rate, _unit_amount, _gene_length, matchFun)

    count = 0
    while count < _generation:
        data.logger.info("Generation #" + str(count+1))
        ga_population.nextGeneration()
        count += 1
        stock_arr, weight_arr = decodeBest(ga_population.best)
        evaluate(list(stock_arr),weight_arr,data.PRICE_PREDICT)
        evaluate(list(stock_arr),weight_arr,data.PRICE_TRUE)


    return ga_population.best

def evaluate(_stocks_list, _weights_arr,_df):
    assert isinstance(_stocks_list, list), "_stocks_list should be a list"
    assert isinstance(_weights_arr, np.ndarray), "_weights_arr should be an array"

    price_df = _df[_stocks_list]
    oct_ret = price_df.iloc[-1,:].div(price_df.iloc[0,:] , axis=0) -1
    data.logger.info(str(oct_ret))
    data.logger.info(str(np.dot(_weights_arr,oct_ret)))

def decodeBest(_best):
    gene = _best.gene
    gene_arr = cutText(gene)
    weight_arr = np.array([decodeGene(sub_gene) for sub_gene in gene_arr])
    max_index_list = getMaxNIndex(weight_arr)
    weight_arr = weight_arr[max_index_list]
    weight_arr /= sum(weight_arr)
    stock_arr = data.STOCK_ARR[max_index_list]
    for i in range(len(stock_arr)):
        data.logger.info(str((stock_arr[i],weight_arr[i])))
    return stock_arr, weight_arr
