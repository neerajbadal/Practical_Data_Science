'''
Created on 23-Apr-2020

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


if __name__ == "__main__":
    commonDir = "D:/Mtech/FY/SEM2/PDS/neeraj/data_pds_ml/pds_covid19/data/"
    countryData = ["US.csv","Belgium.csv","Italy.csv","India.csv"]
    
    countryDataFrame = pd.read_csv(commonDir+"/"+countryData[0],sep='\t',index_col=['Date'],
                                   parse_dates=['Date'])
    print(countryDataFrame)
#     plt.plot(countryDataFrame['Confirmed'],marker='o')
#     plt.show()
#     print(help(statsmodels))
#     sm.graphics.tsa.plot_acf(countryDataFrame['Confirmed'].values.squeeze(),lags=40)
#     plt.show()
    
    
    
#     stationarizingDecision = adfuller(countryDataFrame['Confirmed'].values.squeeze())
#     print(stationarizingDecision)
    
    countryDataFrame['log_Confirmed'] = np.log(countryDataFrame['Confirmed'])
    
#     model = pmdarima.auto_arima(
#         countryDataFrame['log_Confirmed'],
#         start_p=1,
#         test='adf',
#         start_q=1,
#         seasonal=True,
#         max_p=6,
#         max_q=6,m=1,d=None,start_P=0,D=0,
#         trace=True,error_action='ignore',
#         suppress_warnings=True,stepwise=True
#          
#         )
#     print(model.summary())
#     model.plot_diagnostics(figsize=(7,5))
#     plt.show()
    
#     vals = countryDataFrame['Confirmed']
#     countryDataFrame['log_Confirmed'] = np.log(countryDataFrame['Confirmed'])
#     moving_window = 15
#     mv_avg = pd.Series(vals).rolling(moving_window).mean()
#     vals_moving_trend_removed = vals - mv_avg
# #     vals_moving_trend_removed.dropna(inplace=True)
#     vals_moving_trend_removed[0:moving_window-1] = vals_moving_trend_removed[moving_window-1] 
# #     print(vals_moving_trend_removed)
# 
#     exp_avg = pd.Series.ewma(countryDataFrame['Confirmed'], span=moving_window).mean()
#     
#     vals_trend_rem = vals - exp_avg
    
    
    
    
#     print(countryDataFrame['Confirmed'])
    
        
#     p = d = q = range(0, 2)
#     pdq = list(itertools.product(p, d, q))
#     seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
#     
#     outParams = []
#     for param in pdq:
#         for param_seasonal in seasonal_pdq:
#             try:
#                 mod = sm.tsa.statespace.SARIMAX(countryDataFrame['log_Confirmed'],
#                                                 order=param,
#                                                 seasonal_order=param_seasonal,
#                                                 enforce_stationarity=False,
#                                                 enforce_invertibility=False)
#     
#                 results = mod.fit(disp=-1)
#                 outParams.append([param, param_seasonal, results.aic])
#                 print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
#             except:
#                 continue
#     
#     
#     
#     outParams = np.array(outParams)
#     minAicIndex = np.argmin(outParams[:,2])
#     print(outParams[minAicIndex])
#     
#     print("before ",mod.polynomial_ar)
#     
#     mod = sm.tsa.statespace.SARIMAX(countryDataFrame['log_Confirmed'],
#                                 order=outParams[minAicIndex,0],
#                                 seasonal_order=outParams[minAicIndex,1],
#                                 enforce_stationarity=False,
#                                 enforce_invertibility=False)
# 
#     results = mod.fit()
# #     results.plot_diagnostics(figsize=(15, 12))
# #     plt.show()
#     
#     print("after ",mod.polynomial_ar)
    
#     exit(0)
#     pred = results.get_prediction(start=countryDataFrame.index[0], dynamic=False,
#                                   full_results=True)
#     pred = np.exp(pred.predicted_mean)
# #     print()
#     
# #     exit(0)
#     plt.plot(countryDataFrame['Confirmed'],label='orig',marker='o')
#     plt.plot(pred,label='pred',marker='o')
#     plt.legend()
#     plt.show()
#     
#     
#     exit(0)
    
    decompos_series = seasonal_decompose(countryDataFrame['log_Confirmed'])
    trend_s = decompos_series.trend
     
    seasonality_ = decompos_series.seasonal
    residual_ = decompos_series.resid
     
#     plt.subplot(411)
#     plt.plot(countryDataFrame['Confirmed'], label='Original')
#     plt.legend(loc='best')
#     plt.subplot(412)
#     plt.plot(trend_s, label='Trend')
#     plt.legend(loc='best')
#     plt.subplot(413)
#     plt.plot(seasonality_,label='Seasonality')
#     plt.legend(loc='best')
#     plt.subplot(414)
#     plt.plot(residual_, label='Residuals')
#     plt.legend(loc='best')
#     plt.tight_layout() 
#     plt.show() 
     
    residual_.dropna(inplace=True)
    model = pmdarima.auto_arima(
        residual_,
        start_p=1,
        test='adf',
        start_q=1,
        seasonal=False,
        max_p=6,
        max_q=6,m=1,d=None,start_P=0,D=0,
        trace=True,error_action='ignore',
        suppress_warnings=True,stepwise=True
         
        )
    print(model.summary())
    model.plot_diagnostics(figsize=(7,5))
    plt.show()
    
    
    
#     getStationaritySummary(residual_)
    
#     sm.graphics.tsa.plot_acf(residual_.values.squeeze(),lags=50)
#     plt.show()
    
    model = ARIMA(residual_, order=(3, 1,1))  
    results_ARIMA = model.fit(disp=-1)
#     pred = results_ARIMA.get_prediction(start=pd.to_datetime('1998-01-01'), dynamic=False)  
    plt.plot(residual_)
    plt.plot(results_ARIMA.fittedvalues, color='red')
     
    plt.show()

    predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
    trend_pd = pd.Series(trend_s, copy=True)
    seasonal_pd = pd.Series(seasonality_, copy=True)
    
    
    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    predictions_ARIMA_log = pd.Series(residual_.ix[0], index=residual_.index)
    predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
#     
    predictions_ARIMA_log = predictions_ARIMA_log.add(trend_pd,fill_value=0)
    predictions_ARIMA_log = predictions_ARIMA_log.add(seasonal_pd,fill_value=0)
    
#     print(predictions_ARIMA_log)
    
#     predictions_ARIMA = predictions_ARIMA_diff
    predictions_ARIMA = np.exp(predictions_ARIMA_log)
    plt.plot(countryDataFrame['Confirmed'],marker='o',label='orig')
    plt.plot(predictions_ARIMA,marker='o',label='pred')
    plt.legend()
    plt.show()
#     decomposition = seasonal_decompose(ts_log)
    
    
    