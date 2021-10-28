import re
import GAClass

def decodeGene(_sub_gene):
    return

def matchFun(_unit):
    gene = _unit.gene

    def cutText(_text,_length = 12):
        text_arr = re.findall('.{'+str(_length)+'}', _text)
        text_arr.append(_text[(len(text_arr)*_length):])
        return text_arr

    gene_arr = cutText(gene)

    stock_arr = []
    weight_arr = []
    for sub_gene in gene_arr:
        stock_id = decodeGene(sub_gene[0:7])
        weight = decodeGene(sub_gene[7:12])
        stock_arr.append(stock_id)
        weight_arr.append(weight)

    def calculatePE(_stock_arr, _weight_arr, _price_df):
        return

    price_df = ""

    return calculatePE(stock_arr, weight_arr, price_df)

def gaStockSelection(_cross_rate = 0.4, _mutation_rate = 0.4, _unit_amount = 1000, _gene_length = 12*10, _generation = 1000):
    ga_population = GAClass.GAList(_cross_rate, _mutation_rate, _unit_amount, _gene_length, matchFun)

    count = 0
    while count < _generation:
        ga_population.nextGeneration()
        count += 1

    return ga_population.best
