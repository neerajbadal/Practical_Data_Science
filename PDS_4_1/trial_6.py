'''
Created on 30-Apr-2020

@author: Neeraj Badal
'''
'''
Created on 30-Apr-2020

@author: Neeraj Badal
'''
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
import ast
from sklearn.metrics import mean_squared_error
#ind 0.3 ewm com seasonal = true
if __name__ == "__main__":
    commonDir = "D:/Mtech/FY/SEM2/PDS/neeraj/data_pds_ml/pds_covid19/data/"
    countryData = ["US.csv","Belgium.csv","Italy.csv","India.csv"]
    
    format_dir = "D:/Mtech/FY/SEM2/PDS/neeraj/data_pds_ml/"
    
    model_names = ["usa_confirmed.mod","belgium_confirmed.mod",
                   "italy_confirmed.mod","india_confirmed.mod",
                   "usa_recovery.mod","belgium_recovery.mod",
                   "italy_recovery.mod","india_recovery.mod",
                   "usa_death.mod","belgium_death.mod",
                   "italy_death.mod","india_death.mod"
                   ]
    
    model_df = pd.read_csv(format_dir+"/param_form_v6.dat",sep='\t',header=None)
    print(model_df)
    model_df = model_df.to_numpy()
    i_count = 0
#     cut_off = [93,80,84,85]
    for model_inst in model_df:
        country_index = model_inst[0]
        column_used = model_inst[1]
        input_style = int(model_inst[2])
        simple_order = ast.literal_eval(model_inst[3])
        season_order_ = ast.literal_eval(model_inst[4])
        
        countryDataFrame = pd.read_csv(commonDir+"/"+countryData[country_index],
                                       sep='\t',index_col=['Date'],
                                   parse_dates=['Date'])
        
        if input_style == 1.0:
            countryDataFrame['log_Confirmed'] = countryDataFrame[column_used]
        else:
            countryDataFrame['log_Confirmed'] = countryDataFrame[column_used].rolling(center
                                                                             =True,
                                                                              window=input_style).mean().fillna(countryDataFrame[column_used])
        
        
        mod = sm.tsa.SARIMAX(countryDataFrame['log_Confirmed'], maxiter=50,order=simple_order,
                             seasonal_order=season_order_,trend='c')
        res = mod.fit()
        
        pred_tsa = res.get_prediction(start=countryDataFrame.index[0], dynamic=False)
        pred_tsa = np.array(pred_tsa.predicted_mean)
        forecasts = res.forecast(steps=22)
        forecasts = np.array(forecasts)
        pred_tsa = np.append(pred_tsa,forecasts,0)
        total_len = len(countryDataFrame[column_used].values.squeeze())
        
        till_train_data = countryDataFrame[column_used].values.squeeze()
#         till_train_data = till_train_data[:cut_off[country_index]]
#         test_data = countryDataFrame[column_used].values.squeeze()
#         test_data = test_data[cut_off[country_index]:]
#         
        mse_ = mean_squared_error(till_train_data,pred_tsa[:len(till_train_data)])
#         mse_test = mean_squared_error(test_data,pred_tsa[len(till_train_data):])
        print(res.summary())
        plt.title(model_names[i_count]+"  "+str(mse_)+"  "+str(np.sqrt(mse_)))
        plt.plot(countryDataFrame[column_used].values.squeeze(),label='orig',marker='o')
        plt.plot(pred_tsa,label='TEST MSE',marker='o')
        plt.legend()
        plt.show()
        
            
        pickle.dump(res, open(format_dir+model_names[i_count], 'wb'))
        i_count += 1