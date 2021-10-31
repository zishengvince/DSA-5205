import pandas as pd
import numpy as np

PRICE_DF = pd.read_csv('prediction.csv')
STOCK_ARR = np.array(PRICE_DF.columns)

RET_DF = PRICE_DF.diff().dropna()/PRICE_DF.iloc[0:-1,:]
RET_DF_10 = RET_DF.iloc[:,0:10]



