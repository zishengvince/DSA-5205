import pandas as pd
import numpy as np

PRICE_DF = pd.read_csv('prediction.csv')
STOCK_ARR = np.array(PRICE_DF.columns)

RET_DF = PRICE_DF.diff().dropna()
# RET_DF = PRICE_DF.diff().loc[1:,:].div(PRICE_DF.diff().loc[0:-1,:],axis=1)


