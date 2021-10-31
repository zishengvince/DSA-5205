# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 22:33:00 2021

@author: pc
"""
#--------------------------
import numpy as np
import pandas as pd
from WindPy import *
import talib as ta
from talib.abstract import *
w.start()
from collections import OrderedDict
from datetime import *
from  math  import *
import statsmodels.api as sm
import numpy.linalg as la   #用来做线性代数运算
#--------------------------
codes = ['AAPL.O',
 'ADBE.O',
 'ADI.O',
 'ADP.O',
 'ADSK.O',
 'AEP.O',
 'ALGN.O',
 'AMAT.O',
 'AMD.O',
 'AMGN.O',
 'AMZN.O',
 'ANSS.O',
 'ASML.O',
 'ATVI.O',
 'AVGO.O',
 'BIDU.O',
 'BIIB.O',
 'BKNG.O',
 'CDNS.O',
 'CDW.O',
 'CERN.O',
 'CHKP.O',
 'CHTR.O',
 'CMCSA.O',
 'COST.O',
 'CPRT.O',
 'CRWD.O',
 'CSCO.O',
 'CSX.O',
 'CTAS.O',
 'CTSH.O',
 'DLTR.O',
 'DOCU.O',
 'DXCM.O',
 'EA.O',
 'EBAY.O',
 'EXC.O',
 'FAST.O',
 'FB.O',
 'FISV.O',
 'FOX.O',
 'FOXA.O',
 'GILD.O',
 'GOOG.O',
 'GOOGL.O',
 'HON.O',
 'IDXX.O',
 'ILMN.O',
 'INCY.O',
 'INTC.O',
 'INTU.O',
 'ISRG.O',
 'JD.O',
 'KDP.O',
 'KHC.O',
 'KLAC.O',
 'LRCX.O',
 'LULU.O',
 'MAR.O',
 'MCHP.O',
 'MDLZ.O',
 'MELI.O',
 'MNST.O',
 'MRNA.O',
 'MRVL.O',
 'MSFT.O',
 'MTCH.O',
 'MU.O',
 'NFLX.O',
 'NTES.O',
 'NVDA.O',
 'NXPI.O',
 'OKTA.O',
 'ORLY.O',
 'PAYX.O',
 'PCAR.O',
 'PDD.O',
 'PEP.O',
 'PTON.O',
 'PYPL.O',
 'QCOM.O',
 'REGN.O',
 'ROST.O',
 'SBUX.O',
 'SGEN.O',
 'SIRI.O',
 'SNPS.O',
 'SPLK.O',
 'SWKS.O',
 'TCOM.O',
 'TEAM.O',
 'TMUS.O',
 'TSLA.O',
 'TXN.O',
 'VRSK.O',
 'VRSN.O',
 'VRTX.O',
 'WBA.O',
 'WDAY.O',
 'XEL.O',
 'XLNX.O',
 'ZM.O']
date = '2015-01-01'
date = datetime.strptime(date,'%Y-%m-%d')
df = w.wss(','.join(codes), "ipo_date",usedf=True)[1]
select_codes = df.loc[df['IPO_DATE']<=date].index.tolist()
len(select_codes)
#--------------------------
#估值因子
def get_values_factor(dates,stocks):
    dict_df = OrderedDict()
    for i in range(len(dates)-1):
        date=dates[i]

        #估值因子value_factor
        factors_codes= "pe_ttm,pe_lyr,pb_mrq_gsd,pb_lyr,pcf_ocf_ttm,ps_ttm,ps_lyr"
        factors_names=['EP_TTM','EP_LYR','BP_LF','BP_LYR','OCF_TTM','SP_TTM','SP_LYR']
        factors_value=w.wss(stocks,factors_codes,"tradeDate="+date)
        factors_value=pd.DataFrame(factors_value.Data,index=factors_names,columns=factors_value.Codes).T
        factors_value=1/factors_value


        #获取企业价值倍数
        factors_value['EV/EBITDA']=w.wss(stocks, "ev2_to_ebitda","tradeDate="+date).Data[0]

        #获取股息率
        #factors_value['DYR']=w.wss(stocks, "dividendyield2","tradeDate="+date).Data[0]

        dict_df[date]=factors_value
    factors_values=pd.concat(dict_df.values(),keys=dict_df.keys())
    return factors_values
#规模因子
#def get_size_factor(dates,stocks):
#    dict_df = OrderedDict()
#    for i in range(len(dates)-1):
#        date=dates[i]
#        size_factors=w.wss(stocks, "val_lnmv,val_lnfloatmv,val_lntotassets","tradeDate="+date)
#        factors_names=['LN_MV','LN_FLOAT_MV','LN_TOTAL_ASSETS']
#        size_factors=pd.DataFrame(size_factors.Data,index=factors_names,columns=size_factors.Codes).T

#        ev3=np.array(w.wss(stocks, "ev3","tradeDate="+date).Data[0]
        #mktcap=np.array(w.wss(stocks, "mkt_cap_float","tradeDate="+date).Data[0]
        #wgsd_assets=np.array(w.wss(stocks, "wgsd_assets","tradeDate="+date).Data[0]

#        size_factors['LN_EV3']=log(ev3)
        #size_factors['LN_MKTCAP']=log(mktcap)
        #size_factors['LN_ASSET']=log(wgsd_assets)

#        size_factors=pd.DataFrame(size_factors.Data,index=factors_names,columns=size_factors.Codes).T

#        size_factors.drop(['LN_MV','LN_FLOAT_MV','LN_TOTAL_ASSETS'],axis=1)
#        dict_df[date]=size_factors.iloc[:,:]
        #print(dict_df.values())
        #print(dict_df.keys())
#    size_factors=pd.concat(dict_df.values(),keys=dict_df.keys())
#    return size_factors

#杠杆因子
def get_leverage_factors(dates,stocks,factors_codes,factors_names):
    dict_df = OrderedDict()
    for i in range(len(dates)-1):
        date=dates[i]
        leverage_factors=w.wss(stocks,factors_codes,"tradeDate="+date)
        leverage_factors=pd.DataFrame(leverage_factors.Data,index=factors_names,columns=leverage_factors.Codes).T
        dict_df[date]=leverage_factors
    leverage_factors=pd.concat(dict_df.values(),keys=dict_df.keys())
    return leverage_factors
#技术因子
def get_Technical_factors(dates,stocks):
    dict_df = OrderedDict()
    for i in range(len(dates)-1):
        date=dates[i]
        factors_codes= "pe_ttm,pe_lyr,pb_mrq_gsd,pb_lyr,pcf_ocf_ttm,ps_ttm,ps_lyr"
        factors_names=['EP_TTM','EP_LYR','BP_LF','BP_LYR','OCF_TTM','SP_TTM','SP_LYR']
        Technical_factors=w.wss(stocks,factors_codes,"tradeDate="+date)
        Technical_factors=pd.DataFrame(Technical_factors.Data,index=factors_names,columns=Technical_factors.Codes).T
         #获取RSI指标
        Technical_factors['RSI']=w.wss(stocks, "RSI","tradeDate="+date+";RSI_N=6;priceAdj=F;cycle=D").Data[0]
        #获取DEA异同平均数指标
        Technical_factors['DEA']=w.wss(stocks, "MACD","tradeDate="+date+";MACD_L=26;MACD_S=12;MACD_N=9;MACD_IO=2;priceAdj=F;cycle=D").Data[0]
        #获取MACD指标
        Technical_factors['MACD']=w.wss(stocks, "MACD","tradeDate="+date+";MACD_L=26;MACD_S=12;MACD_N=9;MACD_IO=3;priceAdj=F;cycle=D").Data[0]
        #获取K\D\J
        Technical_factors['K']=w.wss(stocks, "KDJ","tradeDate="+date+";KDJ_N=9;KDJ_M1=3;KDJ_M2=3;KDJ_IO=1;priceAdj=F;cycle=D").Data[0]
        Technical_factors['D']=w.wss(stocks, "KDJ","tradeDate="+date+";KDJ_N=9;KDJ_M1=3;KDJ_M2=3;KDJ_IO=2;priceAdj=F;cycle=D").Data[0]
        Technical_factors['J']=w.wss(stocks, "KDJ","tradeDate="+date+";KDJ_N=9;KDJ_M1=3;KDJ_M2=3;KDJ_IO=3;priceAdj=F;cycle=D").Data[0]

        #Technical_factors.drop(['RVI','RSTR12','CYF','CRY','CR20'],axis=1)
        dict_df[date]=Technical_factors
    Liquidation_factors=pd.concat(dict_df.values(),keys=dict_df.keys())
    return Liquidation_factors
#动量因子
#def get_Momentum_factors(dates,stocks):
#    dict_df=OrderedDict()
#    for i in range(len(dates)-1):
#        date=dates[i]
#        factors_codes="tech_revs5,tech_revs10,tech_revs60,tech_revs120,tech_revs250,tech_revs750,tech_revs1mmax,tech_lnhighlow20d"
#        factors_names=['REV_5D','REV_10D','REV_3M','REV_6M','REV_1Y','REV_3Y','REV_LAST1M_MAX','LN_HIGH-LOW']
#        Momentum_factors=w.wss(stocks,factors_codes,"tradeDate="+date)
#        Momentum_factors=pd.DataFrame(Momentum_factors.Data,index=factors_names,columns=Momentum_factors.Codes).T
#       dict_df[date]=Momentum_factors
#    Momentum_factors=pd.concat(dict_df.values(),keys=dict_df.keys())
#    return Momentum_factors

#获取成长因子,填补缺失值
def get_growth_factors(dates,stocks):
    dict_df = OrderedDict()
    for i in range(len(dates)-1):
        date=dates[i]
        factors_codes= "yoyeps_basic,yoy_tr,yoyebt,yoynetprofit,yoyocf,yoyroe"
        factors_names=['eps_gr_TTM','tr_gr_TTM','ebt_gr_TTM','net_profit_gr_TTM','ocf_gr_TTM','roe_gr_TTM']
        growth_factors=w.wss(stocks,factors_codes,"tradeDate="+date)
        growth_factors=pd.DataFrame(growth_factors.Data,index=factors_names,columns=growth_factors.Codes).T
        #growth_factors['eps_growth_TTM']=w.wss(A_stocks, "yoyeps_basic","rptDate="+date+";N=1").Data[0]  #基本每股收益同比增长率
        #growth_factors['roe_growth_TTM']=w.wss(A_stocks, "growth_roe","rptDate="+date+";N=1").Data[0]  #净资产收益率N年同比增长率
        dict_df[date]=growth_factors
        growth_factors=pd.concat(dict_df.values(),keys=dict_df.keys())
    return growth_factors
#市值因子
def get_assisted_factors(dates,stocks):
    dict_df = OrderedDict()
    for i in range(len(dates)-1):
        date=dates[i]
        assisted_factors=w.wss(stocks, "industry_gics,mkt_cap","tradeDate="+date+';industryType=1;unit=1')
        factors_names=['INDUSTRY_SW','CAP']
        assisted_factors=pd.DataFrame(assisted_factors.Data,index=factors_names,columns=assisted_factors.Codes).T
        dict_df[date]=assisted_factors
    assisted_factors=pd.concat(dict_df.values(),keys=dict_df.keys())
    return assisted_factors
#获取每月交易日期序列
def get_trade_date(start_date, end_date, period='M'):
    data = w.tdays(start_date, end_date, period=period) #获取每月最后一个交易日
    trade_dates = data.Data[0]
    trade_dates = [dt.strftime("%Y-%m-%d") for dt in trade_dates]
    return trade_dates
def get_feature_names(data):  #该函数用于获取数据集中需测试的因子名
    columns = data.columns.tolist()
    fea_names = [i for i in columns if i not in ["INDUSTRY_SW",'CAP'] ]
    return fea_names
def extreme_process_MAD(Data):
    feature_names = get_feature_names(Data)
    median=Data[feature_names].median(axis=0)  #获取中位数
    MAD=abs(Data[feature_names].sub(median,axis=1)).median(axis=0)
    for j in range(len(MAD)):
        for i in range(Data.shape[0]):
            if np.isnan(Data.iloc[i,j]) == False:
                if Data.iloc[i,j] <= median[j]-5*1.4826*MAD[j]:
                    Data.iloc[i,j] = median[j]-5*1.4826*MAD[j]
                if Data.iloc[i,j] >= median[j]+5*1.4826*MAD[j]:
                    Data.iloc[i,j] = median[j]+5*1.4826*MAD[j]

    return Data
def fill_missing_value(Data):
    feature_names = get_feature_names(Data)
    for j in range(len(feature_names)):
        industry_fill_value = Data[feature_names[j]].groupby(Data['INDUSTRY_SW']).mean()
        #print(j,list(industry_fill_value))
        for i in range(Data.shape[0]):
            #if i < 3:
                #print(Data.iloc[i,:]['INDUSTRY_SW'])
            if np.isnan(Data.iloc[i,j]):
                #print(industry_fill_value[Data.iloc[i,-2]])
                Data.iloc[i,j] = industry_fill_value[Data.iloc[i,-2]]
    return Data
#市值中性化
def data_scale_CAP(data):
    feature_names = get_feature_names(data)
    data_=data.copy()
    cap_weight = data_["CAP"]/ data_["CAP"].sum()
    for name in feature_names:
        avg=(data_[name]*cap_weight).sum()
        data_[name]=(data_[name]-avg)/data_[name].std()
    return data_
#行业中性化
def data_scale_neutral(data):
    feature_names = get_feature_names(data)
    data_=data.copy()
    industrys=data['INDUSTRY_SW']  #获取所属申万一级行业代码
    data_med = pd.get_dummies(data,columns=['INDUSTRY_SW'],drop_first=True)
    n = len(data['INDUSTRY_SW'].unique())    #确定产生虚拟变量个数
    X = np.array(data_med[data_med.columns[-(n-1):]])  #行业虚拟变量作为为自变量
    for name in feature_names:
        y = np.array(data_[name])
        if la.matrix_rank(X.T.dot(X)) == (n-1): #当矩阵满秩时，估计回归参数
            beta_ols = la.inv(X.T.dot(X)).dot(X.T).dot(y)
            residual = y - X.dot(beta_ols)      #计算残差，并将其作为剔除行业影响的因子值
        else:
            residual = y   #如果逆不存在的话 则 用原值
        data_[name]=residual
    return data_
#因变量涨跌幅的获取以及处理
def get_pct(dates,stocks):
    dict_df = OrderedDict()
    for i in range(len(dates)-1):
        date=dates[i]
        h = "tradeDate="+date+";cycle=M"
        factors_value=w.wss(stocks,"pct_chg",h,usedf=True)[1]
        dict_df[date]=factors_value
    d=pd.concat(dict_df.values(),keys=dict_df.keys())
    return d
def accuracy(data1,data2):
    n=0
    for i in range(len(data1)):
        if data1[i] == data2[i]:
            n+=1
    acc = n/len(data1)
    return acc
#--------------------------
start_date='20200701'
end_date='20210630'
dates=get_trade_date(start_date, end_date, period='M')
#--------------------------
start_date='20200701'
end_date='20210630'
dates=get_trade_date(start_date, end_date, period='M')
values_factor=get_values_factor(dates,select_codes)
#size_factor=get_size_factor(dates,select_codes)
factors_codes="current,debttoassets,ocftodebt,debttoequity"
factors_names=['CUR','DEBT_TO_ASSETS','CASH_FLOW_LIABILITY','DEBT_TO_EQUITY']
leverage_factors = get_leverage_factors(dates,select_codes,factors_codes,factors_names)
Technical_factors = get_Technical_factors(dates,select_codes)
#Momentum_factors = get_Momentum_factors(dates,select_codes)
assisted_factors = get_assisted_factors(dates,select_codes)
growth_factors=get_growth_factors(dates,select_codes)
Data= pd.concat([values_factor,growth_factors,leverage_factors,Technical_factors,assisted_factors],axis=1)
#--------------------------
Data.to_csv('data/s.csv')
Data = pd.read_csv('data/s.csv',index_col=[0,1])
Data
#--------------------------
Data2 =Data.groupby(level=0).apply(extreme_process_MAD)
Data2.to_csv('data/svm_data_2.csv')
Data2 = pd.read_csv('data/svm_data_2.csv',index_col=[0,1])
Data3=fill_missing_value(Data2)
#--------------------------
industry_fill_value = Data['CAP'].groupby(Data['INDUSTRY_SW']).mean()
#--------------------------
for i in range(Data3.shape[0]):
    if np.isnan(Data3.iloc[i,-1]):
        Data3.iloc[i,-1] = industry_fill_value[Data3.iloc[i,:]['INDUSTRY_SW']]
Data3 = Data3.fillna(Data3.mean())
Data4 = data_scale_CAP(Data3)
Data5 = data_scale_neutral(Data4)
Data6 = Data5.drop(['INDUSTRY_SW'],axis=1)
mean = Data6.mean()
std = Data6.std()
Data7 = (Data6-mean)/std
Data7.to_csv('data/SVM_X.csv')
#--------------------------
select_codes
#--------------------------
pct_start_date='20200801'
pct_end_date='20210731'
pct_dates=get_trade_date(pct_start_date,pct_end_date, period='M')
pct=get_pct(pct_dates,select_codes)
y = pct.dropna()
#--------------------------
y = y.sort_values(by='PCT_CHG')
#--------------------------
train_y = pd.concat([y.head(350),y.tail(350)])
train_y2 = train_y.reset_index()
#--------------------------
pct_dates.insert(0,dates[0])#为了让日期索引对齐加入第一个月份
#将每个月份前退一个月的交易日
for i in range(len(train_y2.iloc[:,0])):
    train_y2.iloc[i,0]=pct_dates[pct_dates.index(train_y2.iloc[i,0])-1]
train_y3 = train_y2.set_index(['level_0','level_1'])
Y = train_y3#采用的是月的涨跌幅所以会比较大
#--------------------------
X = pd.read_csv('data/SVM_X.csv',index_col=[0,1])
X = X.loc[Y.index]
Y.iloc[:350,0]=0
Y.iloc[350:,0]=1
print(np.isnan(X).any())
#--------------------------
X.dropna(inplace=True)
len(X)
#--------------------------
from sklearn.svm import SVC,SVR
from sklearn.model_selection import GridSearchCV
#--------------------------
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
rbf_tuned_parameters = [{'kernel': ['rbf'], 'gamma':[1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1],'C': [0.01,0.03,0.1,0.3, 1, 3, 10]}]
#scores = ['precision']
poly3_tuned_parameters = [{'kernel': ['poly'],'degree':[3],'gamma':[1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1],'C': [0.01,0.03,0.1,0.3, 1, 3, 10]}]
poly7_tuned_parameters = [{'kernel': ['poly'],'degree':[7],'gamma':[1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1],'C': [0.01,0.03,0.1,0.3, 1, 3, 10]}]
linear_tuned_parameters = [{'kernel': ['linear'],'C': [0.01,0.03,0.1,0.3, 1, 3, 10]}]
sigmoid_tuned_parameters = {'kernel': ['sigmoid'],'gamma': [1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1],'C': [0.01,0.03,0.1,0.3, 1, 3, 10]}
#--------------------------
def svc_parameter_select(X,Y,tuned_parameters):
    print(" Tuning hyper-parameters" )
    print()
    # 调用 GridSearchCV，将 SVC(), tuned_parameters, cv=10, 还有 scoring 传递进去，
    clf = GridSearchCV(SVC(), tuned_parameters, cv=10,scoring='precision_macro')#'precision_macro'为分类所用的得分判断标准之一
    # 用训练集训练这个学习器 clf
    clf.fit(X.values, Y.values.ravel())
    print("Best parameters set found on development set:")
    print()
    # 再调用 clf.best_params_ 就能直接得到最好的参数搭配结果
    print(clf.best_params_)
    print()
    #y_true, y_pred = y_test, clf.predict(X_test)

    # 打印在测试集上的预测结果与真实值的分数
    #print(classification_report(y_true, y_pred))

    #print()
    return clf
#--------------------------
SVC_poly3 = svc_parameter_select(X,Y,poly3_tuned_parameters)
print('SVC_poly3',SVC_poly3.best_params_)
SVC_poly7 = svc_parameter_select(X,Y,poly7_tuned_parameters)
print('SVC_poly7',SVC_poly7.best_params_)
SVC_linear = svc_parameter_select(X,Y,linear_tuned_parameters)
print('SVC_linear',SVC_linear.best_params_)
SVC_sigmoid = svc_parameter_select(X,Y,sigmoid_tuned_parameters)
print('SVC_sigmoid',SVC_sigmoid.best_params_)
SVC_rbf = svc_parameter_select(X,Y,rbf_tuned_parameters)
print('SVC_rbf',SVC_rbf.best_params_)
#--------------------------
from sklearn.svm import SVC
poly3clf = SVC(C=0.03,kernel='poly',degree=3, gamma=0.3,probability=True).fit(X,Y)
poly7clf = SVC(C=3,kernel='poly',degree=7,gamma=0.1,probability=True).fit(X,Y)
linear = SVC(C=0.01,kernel='linear',probability=True).fit(X,Y)
sigmoid = SVC(C=3,kernel='sigmoid', gamma=0.0003,probability=True).fit(X,Y)
rbf = SVC(C=1,kernel='rbf',degree=3,gamma=0.01,probability=True).fit(X,Y)
#--------------------------
test_y= y[350:-350]
test_y.index
test_x =X.loc[test_y.index].dropna()
poly3_pre = poly3clf.predict(test_x)
poly7_pre =  poly7clf.predict(test_x)
linear_pre = linear.predict(test_x)
sigmoid_pre = sigmoid.predict(test_x)
rbf_pre = rbf.predict(test_x)
real_y = test_y.loc[test_x.index]
#--------------------------
for i in range(len(real_y)):
    if real_y.iloc[i,0] <=0 :
        real_y.iloc[i,0] = 0
    else:
        real_y.iloc[i,0] = 1
#--------------------------
import matplotlib.pyplot as plt
boundarys = np.linspace(-1,1,100)
l1 = []
l2 = []
l3 = []
l4 = []
l5 = []
for boundary in boundarys:
    real_y2 = test_y.loc[test_x.index]
    for i in range(len(real_y2)):
        if real_y2.iloc[i,0] <=boundary :
            real_y2.iloc[i,0] = 0
        else:
            real_y2.iloc[i,0] = 1
    l1.append(accuracy(real_y2.values,poly3_pre))
    l2.append(accuracy(real_y2.values,poly7_pre))
    l3.append(accuracy(real_y2.values,linear_pre))
    l4.append(accuracy(real_y2.values,sigmoid_pre))
    l5.append(accuracy(real_y2.values,rbf_pre))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.xlim=(-1,1)
ax.plot(l1,'y',label='poly3')
ax.plot(l2,'b',label='poly7')
ax.plot(l3,'r',label='linear')
ax.plot(l4,'p',label='sigmoid')
ax.plot(l5,'g',label='rbf')
ax.legend(loc='best')
ax.set_title('正确率变化情况')

plt.show()
#--------------------------
from sklearn import metrics
#--------------------------
print('poly3 AUC',metrics.roc_auc_score(real_y.values,poly3clf.predict_proba(test_x)[:,1]))
print('poly7 AUC',metrics.roc_auc_score(real_y.values,poly7clf.predict_proba(test_x)[:,1]))
print('linear AUC',metrics.roc_auc_score(real_y.values,linear.predict_proba(test_x)[:,1]))
print('sigmoid AUC',metrics.roc_auc_score(real_y.values,sigmoid.predict_proba(test_x)[:,1]))
print('rbf AUC',metrics.roc_auc_score(real_y.values,rbf.predict_proba(test_x)[:,1]))
#--------------------------
test_start_date='20210801'
test_end_date='20211015'
test_dates=get_trade_date(test_start_date, test_end_date, period='M')
values_factor=get_values_factor(test_dates,select_codes)
#size_factor=get_size_factor(test_dates,select_codes)
factors_codes="current,debttoassets,ocftodebt,debttoequity"
factors_names=['CUR','DEBT_TO_ASSETS','CASH_FLOW_LIABILITY','DEBT_TO_EQUITY']
leverage_factors = get_leverage_factors(test_dates,select_codes,factors_codes,factors_names)
Technical_factors = get_Technical_factors(test_dates,select_codes)
#Momentum_factors = get_Momentum_factors(test_dates,select_codes)
assisted_factors = get_assisted_factors(test_dates,select_codes)
growth_factors=get_growth_factors(test_dates,select_codes)
test_Data= pd.concat([values_factor,growth_factors,leverage_factors,Technical_factors,assisted_factors],axis=1)
#--------------------------
test_Data.to_csv('data/testdata.csv')
test_Data = pd.read_csv('data/testdata.csv',index_col=[0,1])
test_Data
#--------------------------
test_Data2 =test_Data.groupby(level=0).apply(extreme_process_MAD)
#--------------------------
test_Data3=fill_missing_value(test_Data2)
test_Data4 = test_Data3.fillna(test_Data3.mean())
t = test_Data4.CAP.fillna(test_Data4.CAP.mean())
test_Data4.drop(['CAP'],axis=1)
test_Data4['CAP']=t
test_Data5 = data_scale_CAP(test_Data4)
test_Data6 = data_scale_neutral(test_Data5)
test_Data7 = test_Data6.drop(['INDUSTRY_SW'],axis=1)
mean = test_Data7.mean()
std = test_Data7.std()
testData8 = (test_Data7-mean)/std
#--------------------------
test_Data = testData8
test_poly3_pre = poly3clf.predict_proba(test_Data)[:,1].reshape(-1,89)
test_poly7_pre = poly7clf.predict_proba(test_Data)[:,1].reshape(-1,89)
test_linear_pre = linear.predict_proba(test_Data)[:,1].reshape(-1,89)
test_sigmoid_pre = sigmoid.predict_proba(test_Data)[:,1].reshape(-1,89)
test_rbf_pre = rbf.predict_proba(test_Data)[:,1].reshape(-1,89)
#--------------------------
#该函数返回选购股票以及所对应的判断上涨的概率的列表
def select_code_index(model_predict_prob,codes,num=10):
    L =[]
    for i in range(model_predict_prob.shape[0]):
        l = list(model_predict_prob[i,])
        h= []
        for k in sorted(l)[::-1][:num]:
            h.append([codes[l.index(k)],k/sum(sorted(l)[::-1][:num])])
        L.append(h)
    return L
#--------------------------
#此部分输出2021年8月、9月选股结果
poly3_predict = select_code_index(test_poly3_pre,select_codes)
poly7_predict = select_code_index(test_poly7_pre,select_codes)
linear_predict = select_code_index(test_linear_pre,select_codes)
sigmoid_predict=select_code_index(test_sigmoid_pre,select_codes)
rbf_predict = select_code_index(test_rbf_pre,select_codes)
#--------------------------
poly3_predict #三阶多项式核选股结果
poly7_predict #7阶多项式核选股结果
linear_predict #线性核选股结果
sigmoid_predict  #sigmoid核
rbf_predict #高斯核函数
#--------------------------
#--------------------------
#--------------------------
#--------------------------
