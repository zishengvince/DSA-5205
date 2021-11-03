import pandas as pd
import numpy as np
import logging

PRICE_PREDICT = pd.read_csv('N100CLOSE_PREDICT.csv')
RET_PREDICT = pd.read_csv('N100RETURN_PREDICT.csv')
# POSI_RET_PREDICT = RET_PREDICT.iloc[:,(RET_PREDICT.mean()>0).values]

PRICE_TRUE = pd.read_csv('N100CLOSE_TRUE.csv')
RET_TRUE = pd.read_csv('N100RETURN_TRUE.csv')
RET_TRUE_SEP = pd.read_csv('N100RETURN_TRUE_SEP.csv')
PRICE_TRUE_2021 = pd.read_csv('close_price.csv')
RET_TRUE_2021 = PRICE_TRUE_2021.pct_change().dropna()

# POSI_STOCK_ARR = np.array(POSI_RET_PREDICT.columns)
STOCK_ARR = np.array(PRICE_TRUE.columns)

accurate_prediction = pd.read_csv('prediction.csv')
accurate_return = accurate_prediction.pct_change().dropna()

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.DEBUG)
handler = logging.FileHandler("log.log")
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)-5s - %(filename)s:%(funcName)s - %(message)s')
handler.setFormatter(formatter)

console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
console.setFormatter(formatter)

logger.addHandler(handler)
logger.addHandler(console)

