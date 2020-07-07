'''
Created on 25-Apr-2020

@author: Neeraj Badal
'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import pmdarima
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
import itertools
from sklearn.metrics import mean_squared_error

#ind 0.3 ewm com seasonal = true
if __name__ == "__main__":
    commonDir = "D:/Mtech/FY/SEM2/PDS/neeraj/data_pds_ml/pds_covid19/data/"
    countryData = ["US.csv","Belgium.csv","Italy.csv","India.csv"]
    
    countryDataFrame = pd.read_csv(commonDir+"/"+countryData[3],sep='\t',index_col=['Date'],
                                   parse_dates=['Date'])
    print(countryDataFrame)
    
    columnUsed = 'Confirmed'
    ewm_com_val = np.arange(0.0,1.1,0.1)
    
    
#     countryDataFrame['log_Confirmed'] = np.log(countryDataFrame[columnUsed]+1)
#     countryDataFrame['log_Confirmed'] = countryDataFrame[columnUsed].ewm(com=0.8).mean()
    
    res_list = []
    
    
    
    p = d= q = range(0,8)
    pdq = list(itertools.product(p,d, q))
    seasonal_pdq = [(x[0],x[1],x[2],7) for x in list(itertools.product(p,d, q))]
     
    outParams = []
    for param in pdq:
        if param[1] > 1:
            continue
        for param_seasonal in seasonal_pdq:
            if param_seasonal[1] != 1:
                continue
            try:
                mod = sm.tsa.statespace.SARIMAX(countryDataFrame['log_Confirmed'],
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False,
                                                trend='c'
                                                )
     
                results = mod.fit(disp=-1)
                outParams.append([param, param_seasonal, results.aic])
                print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue
     
     
     
    outParams = np.array(outParams)
    minAicIndex = np.argmin(outParams[:,2])
    print(outParams[minAicIndex])

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    for com_val in ewm_com_val:
        countryDataFrame['log_Confirmed'] = countryDataFrame[columnUsed].ewm(com=com_val).mean()
        for season_Flag in [False,True]:
            for diff_val in [0,1]:
            
                model = pmdarima.auto_arima(
                countryDataFrame['log_Confirmed'][:],
                start_p=1,
                test='adf',
                start_q=1,
                seasonal=season_Flag,
                max_p=7,
                max_q=7,m=7,d=diff_val,D=1,start_P=0,start_Q=0,
                max_P=5,max_Q=5,
                trace=True,error_action='ignore',
                trend='c',
                enforce_stationarity=False,
                suppress_warnings=True,stepwise=False,
                n_jobs=3,
                out_of_sample_size= 0,
                scoring='mse'
                )
                print(model.summary())    
                pred = model.predict_in_sample(start=0, dynamic=False)
            
                mse_ = mean_squared_error(countryDataFrame[columnUsed].values.squeeze(),pred)
                res_list.append([model,mse_,model.aic(),com_val,season_Flag,diff_val])
        
    res_list = np.array(res_list)
    aic_best = np.argmin(res_list[:,2])
    mse_best = np.argmin(res_list[:,1])    
    print("AIC Best Index ",aic_best,"....",res_list[aic_best])
    print("MSE Best Index ",mse_best,"....",res_list[mse_best])