'''
Created on 29-Apr-2020

@author: Neeraj Badal
'''
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
import pickle

def getStationaritySummary(ts_val):
    
    movingMean = pd.Series(ts_val).rolling(15).mean()
    movinfStd = pd.Series(ts_val).rolling(15).std()

    
    
    print('Dickey-Fuller Test : ')
    test_res = adfuller(ts_val, autolag='AIC')
    df_output = pd.Series(test_res[0:4], index=['T - Statistic','p-val','Lags ','Observations'])
    for key,val in test_res[4].items():
        df_output['Critical Value '+str(key)] = val
    print(df_output)
    
    plt.plot(ts_val, color='blue',label='Original')
    plt.plot(movingMean, color='red', label='Moving Mean')
    plt.plot(movinfStd, color='green', label = 'moving Std')
    plt.title('Moving Mean and Std Dev.')
    plt.legend()
    plt.show()

def inverse_difference(last_ob, value):
    return value + last_ob

def undo_difference(x, d=1):
    if d == 1:
        return np.cumsum(x)
    else:
        x = np.cumsum(x)
        return undo_difference(x, d - 1)

def difference(dataset, interval=1):
    diff = list()
    for i in range(0, len(dataset)):
        if i < interval:
            diff.append(0.0)
        else:
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
    return diff



def automateARIMA(dataFrame):
    p = d = q = range(0, 9)
    pdq = list(itertools.product(p, d, q))
#     seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
     
    outParams = []
    for param in pdq:
        try:
            mod = ARIMA(dataFrame,order=param)
            
            results = mod.fit(disp=-1)
            outParams.append([param,results.aic])
            print('ARIMA{}x{}12 - AIC:{}'.format(param, results.aic))
        except:
#             print('exception')
            continue
     
    outParams = np.array(outParams)
    minAicIndex = np.argmin(outParams[:,1])
    print(outParams[minAicIndex])
#ind 0.3 ewm com seasonal = true
if __name__ == "__main__":
    commonDir = "D:/Mtech/FY/SEM2/PDS/neeraj/data_pds_ml/pds_covid19/data/"
    countryData = ["US.csv","Belgium.csv","Italy.csv","India.csv"]
    
    countryDataFrame = pd.read_csv(commonDir+"/"+countryData[0],sep='\t',index_col=['Date'],
                                   parse_dates=['Date'])
    print(countryDataFrame)
    
    columnUsed = 'Confirmed'
#     columnUsed = 'Recovered'
#     columnUsed = 'Deaths'
    
#     countryDataFrame['log_Confirmed'] = np.log(countryDataFrame[columnUsed]+1)
#     countryDataFrame['log_Confirmed'] = countryDataFrame[columnUsed].ewm(alpha=1.0).mean()
#     countryDataFrame['log_Confirmed'] = countryDataFrame[columnUsed]
    countryDataFrame['log_Confirmed_i'] = countryDataFrame[columnUsed].rolling(center
                                                                             =True,
                                                                              window=3).mean().fillna(countryDataFrame[columnUsed])
#     test_size = int(0.85 * (len(countryDataFrame['log_Confirmed'])))
    
#     countryDataFrame['log_Confirmed'] = countryDataFrame[columnUsed]
    countryDataFrame['log_Confirmed'] = countryDataFrame[columnUsed].rolling(center
                                                                            =True,
                                                                            window=3).mean().fillna(countryDataFrame[columnUsed])
    
    
    countryDataFrame['log_Confirmed_ii'] = countryDataFrame[columnUsed].rolling(center
                                                                            =True,
                                                                            window=3).mean().fillna(countryDataFrame[columnUsed])
    
    countryDataFrame['log_Confirmed_iii'] = countryDataFrame[columnUsed].ewm(alpha=0.2).mean()
#     countryDataFrame['log_Confirmed_iii'] = countryDataFrame[columnUsed].rolling(center
#                                                                             =True,
#                                                                             window=3).mean().fillna(countryDataFrame[columnUsed])
    
    len_used = 98
#     model = pmdarima.auto_arima(
#     countryDataFrame['log_Confirmed_i'][:len_used],
#     start_p=1,
#     test='adf',
#     start_q=1,
#     seasonal=True,
#     max_p=7,
#     max_q=7,m=7,d=1,D=1,start_P=0,start_Q=0,
#     max_P=5,max_Q=5,
#     trace=True,error_action='ignore',
#     enforce_stationarity=False,
#     suppress_warnings=True,stepwise=False,
#     n_jobs=3,
#     out_of_sample_size= 0,
#     scoring='mse'
#     )
#     
#     print(model.summary())
#     best_params = model.get_params()
#     print(best_params)
#     print(best_params['order'])
#     print(best_params['seasonal_order'])
    
#     exit(0)
    
#     model.plot_diagnostics(figsize=(7,5))
#                  
#     plt.show()   
#     
#     pred = model.predict_in_sample(start=0, dynamic=False
#                                 )
#     
#     pred_4 = model.predict(n_periods=22)
#     print(pred_4)
#     pred = np.append(pred,pred_4,0)
    
    
#     diff_order= (1,1,3)
#     seasonal_ord = (1,1,0,7)
#     diff_order= (1,1,3)
#     seasonal_ord = (0,0,0,0)
    diff_order= (1,1,3)
    seasonal_ord = (0,1,1,7)
    mod_1 = sm.tsa.SARIMAX(countryDataFrame['log_Confirmed_i'][:len_used], maxiter=50,order=diff_order,seasonal_order=seasonal_ord,trend='c')
    res_1 = mod_1.fit()
    
    pred = res_1.get_prediction(start=countryDataFrame.index[0], dynamic=False)
    pred = np.array(pred.predicted_mean)
    forecasts_1 = res_1.forecast(steps=22)
    forecasts_1 = np.array(forecasts_1)
#     print(pred_tsa)
#     exit(0)
    pred = np.append(pred,forecasts_1,0)
    
#     diff_order= (1,1,1)
#     seasonal_ord = (0,0,0,0)
#     diff_order= (4,1,0)
#     seasonal_ord = (0,1,1,7)
    diff_order= (5,1,0)
    seasonal_ord = (0,1,0,7)
    mod = sm.tsa.SARIMAX(countryDataFrame['log_Confirmed'][:len_used], maxiter=50,order=diff_order,seasonal_order=seasonal_ord,trend='c')
    res = mod.fit()
    
    
    pred_tsa = res.get_prediction(start=countryDataFrame.index[0], dynamic=False)
    pred_tsa = np.array(pred_tsa.predicted_mean)
    forecasts = res.forecast(steps=22)
    forecasts = np.array(forecasts)
    pred_tsa = np.append(pred_tsa,forecasts,0)
#     pred_tsa = np.exp(pred_tsa)-1
    len_used = 90
    
    diff_order= (1,1,3)
    seasonal_ord = (0,1,1,7)
    mod_2 = sm.tsa.SARIMAX(countryDataFrame['log_Confirmed_ii'][:len_used], maxiter=50,order=diff_order,seasonal_order=seasonal_ord,trend='c')
    res_2 = mod_2.fit()
    
    pred_2 = res_2.get_prediction(start=countryDataFrame.index[0], dynamic=False)
    pred_2 = np.array(pred_2.predicted_mean)
    forecasts_2 = res_2.forecast(steps=22)
    forecasts_2 = np.array(forecasts_2)
#     print(pred_tsa)
#     exit(0)
    pred_2 = np.append(pred_2,forecasts_2,0)
    
    
    diff_order= (1,1,4)
    seasonal_ord = (0,1,0,7)
    mod_3 = sm.tsa.SARIMAX(countryDataFrame['log_Confirmed_iii'][:len_used], maxiter=50,order=diff_order,seasonal_order=seasonal_ord,trend='c')
    res_3 = mod_3.fit()
    
    pred_3 = res_3.get_prediction(start=countryDataFrame.index[0], dynamic=False)
    pred_3 = np.array(pred_3.predicted_mean)
    forecasts_3 = res_3.forecast(steps=22)
    forecasts_3 = np.array(forecasts_3)
#     print(pred_tsa)
#     exit(0)
    pred_3 = np.append(pred_3,forecasts_3,0)
    
    
    
    plt.plot(countryDataFrame[columnUsed].values.squeeze(),label='orig',marker='o')
    plt.plot(pred,label='Training MSE',marker='o')
    plt.plot(pred_tsa,label='TEST MSE',marker='o')
    plt.plot(pred_2,label='PURE TRAIN MSE',marker='+')
    plt.plot(pred_3,label='PURE AIC',marker='o')
#     plt.plot(pred_tsa_2,label='Pickle',marker='o')
    plt.legend()
    plt.show()
    exit(0)
#     seasonal_order=(0, 1, 1, 7),
    '''
    USA 2,1,1 0,1,1,7
    '''
  
    countryDataFrame['diff'] = countryDataFrame['log_Confirmed'].diff(1)
#     countryDataFrame['diff'].fillna(method='bfill',inplace=True)
#     countryDataFrame['diff'].fillna(method='ffill',inplace=True)
    countryDataFrame['diff'].fillna(0,inplace=True)
     
    countryDataFrame['seasoned_diff'] = countryDataFrame['diff'].diff(7)
#     countryDataFrame['seasoned_diff'].fillna(method='bfill',inplace=True)
#     countryDataFrame['seasoned_diff'].fillna(method='ffill',inplace=True)
    countryDataFrame['seasoned_diff'].fillna(0,inplace=True)
    
  
#     mod = sm.tsa.statespace.SARIMAX(countryDataFrame['log_Confirmed'],
#                                 order=(2,1,1),
#                                 seasonal_order=(0,1,1,7),enforce_stationarity=False,    
#                                 enforce_invertibility=True,trend='c')
#   
#     results = mod.fit(disp=-1) 
# #     print(results.summary())
#     results.plot_diagnostics(figsize=(7,5))
#     plt.show()
#     summary_ = results.summary()
#     print(summary_)
    
    
    history_errors = []
    new_preds = []
    c_ = -0.0002
    phi_1 = 0.7210
    phi_2 = 0.2227
    theta_S_1 = -0.9912
    theta_1 = -0.7463  
    
    
    
    
    n_diff = difference(countryDataFrame['log_Confirmed'].values.squeeze(), 1)
    actual_data = countryDataFrame['log_Confirmed'].values.squeeze()
    n_diff_season = difference(n_diff, 7)
    
#     print(countryDataFrame['seasoned_diff'])
#     print(n_diff)
    
#     inverted = [inverse_difference(n_diff[i], n_diff_season[i]) for i in range(len(n_diff_season))]
#     inverted = [inverse_difference(actual_data[i], inverted[i]) for i in range(len(inverted))]
    
   
    for i_ in range(0,len(countryDataFrame)):
        if i_ < 7:
            history_errors.append(0.0)
            new_preds.append(n_diff_season[i_])
        else:
            ar_term = phi_1*n_diff_season[i_-1]
            ar_term += phi_2*n_diff_season[i_-2]
            ar_term += c_
            ar_term -= theta_S_1*history_errors[i_-7]
            ar_term -= theta_1*history_errors[i_-1]
            ar_term += theta_S_1*theta_1*history_errors[i_-8]
            temp_val = inverse_difference(n_diff[i_],ar_term)  
            temp_val = inverse_difference(actual_data[i_], temp_val)
            ar_term = temp_val
            history_errors.append(countryDataFrame['log_Confirmed'][i_]-ar_term)
            new_preds.append(ar_term)
    
    new_preds = np.exp(new_preds)
#     new_preds = undo_difference(new_preds, 7)
#     new_preds = undo_difference(new_preds, 1)
    
#     print(len(history_errors))
#     exit(0)
#     pred = model.predict_in_sample(start=0, dynamic=False
#                                   )
#     pred_2 = model.predict_in_sample(start=77, dynamic=True)
    
#     pred_ = np.ones(pred.shape[0])
#     pred_[:77] = pred_[:77] * pred[:77]
#     pred_[77:] = pred_[77:]*pred_2[:]
#     pred_2 = pred_ 
#     pred = np.exp(pred)
#     pred_2 = np.exp(pred_2)
    
#     pred = results.get_prediction(start=countryDataFrame.index[0], dynamic=False,
#                                   full_result=True)
  
#     pred_2 = results.get_prediction(start=countryDataFrame.index[0], dynamic='2020-04-10',
#                                   full_result=True)
    
    
    

#     pred = np.exp(pred.predicted_mean)
#     pred_2 = np.exp(pred_2.predicted_mean)
    
#     print()
      
#     exit(0)
    plt.plot(countryDataFrame[columnUsed].values.squeeze(),label='orig',marker='o')
    plt.plot(new_preds,label='pred',marker='o')
#     plt.plot(pred_2,label='pred_2',marker='o')
    plt.legend()
    plt.show()
