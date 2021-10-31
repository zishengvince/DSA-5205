import re
import heapq
import numpy as np
import GAClass
import params
import data

def decodeGene(_sub_gene):
    assert isinstance(_sub_gene, str), "_gene should be a string"
    return float(int(_sub_gene,2))


def cutText(_text):
    text_arr = re.findall('.{'+str(params.stock_gene_len)+'}', _text)
    text_arr.append(_text[(len(text_arr)*params.stock_gene_len):])
    return text_arr[0:-1]


def getMaxNIndex(_arr, n = 10):
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

    ret_df = data.RET_DF[stock_list]
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
        print("Generation #" + str(count+1))
        ga_population.nextGeneration()
        count += 1

    return ga_population.best
