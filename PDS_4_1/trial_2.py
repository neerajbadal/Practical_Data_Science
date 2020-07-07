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
    
    param_file = "D:/Mtech/FY/SEM2/PDS/neeraj/data_pds_ml/param_form_v6.dat"
    param_file_test = "D:/Mtech/FY/SEM2/PDS/neeraj/data_pds_ml/param_form_test_v4.dat"
    for activity_ in ['Confirmed','Recovered','Deaths']:
    
        outDataFile = "D:/Mtech/FY/SEM2/PDS/neeraj/data_pds_ml/summary_"+activity_+"_v6.dat"
        cut_off = [99,86,90,91]
        for countr_ in [0,1,2,3]:
        
            countryDataFrame = pd.read_csv(commonDir+"/"+countryData[countr_],sep='\t',index_col=['Date'],
                                           parse_dates=['Date'])
            
            countryDataFrame = countryDataFrame[:cut_off[countr_]]
            print(countryDataFrame)
    #         exit(0)
            len_data = len(countryDataFrame)
            columnUsed = activity_
        #     ewm_com_val = np.arange(0.1,1.2,0.1)
        #     print(ewm_com_val)
    #         exit(0)
        #     countryDataFrame['log_Confirmed'] = np.log(countryDataFrame[columnUsed]+1)
        #     countryDataFrame['log_Confirmed'] = countryDataFrame[columnUsed].ewm(com=0.8).mean()
            
            res_list = []
            
            for com_val in [1.0,3,5,7]:
        #         print(com_val)
                if com_val == 1.1:
                    countryDataFrame['log_Confirmed'] = np.log(countryDataFrame[columnUsed]+1)
                elif com_val > 1.1:
                    countryDataFrame['log_Confirmed'] = countryDataFrame[columnUsed].rolling(center
                                                                                     =True,
                                                                                      window=com_val).mean().fillna(countryDataFrame[columnUsed])
                else:
                    countryDataFrame['log_Confirmed'] = countryDataFrame[columnUsed].ewm(alpha=com_val).mean()
                for season_Flag in [False,True]:
                    for diff_val in [0,1]:
                    
                        model = pmdarima.auto_arima(
                        countryDataFrame['log_Confirmed'][:len_data],
                        start_p=1,
                        test='adf',
                        start_q=1,
                        seasonal=season_Flag,
                        max_p=8,
                        max_q=8,m=7,d=diff_val,D=1,start_P=0,start_Q=0,
                        max_P=6,max_Q=6,
                        trace=True,error_action='ignore',
                        trend='c',
                        enforce_stationarity=False,
                        suppress_warnings=True,stepwise=False,
                        n_jobs=7,
                        out_of_sample_size= 0,
                        scoring='mse'
                        )
                        print(model.summary())    
                        pred = model.predict_in_sample(start=0, dynamic=False)
                        pred_out_sample = model.predict(n_periods=6)
                        if com_val == 1.1:
                            pred = np.exp(pred)-1.0
                            pred_out_sample = np.exp(pred_out_sample)-1.0
                    
                        mse_ = mean_squared_error(countryDataFrame[columnUsed].values.squeeze()[3:len_data],pred[3:])
#                         vali_mse = mean_squared_error(countryDataFrame[columnUsed].values.squeeze()[len_data-6:],pred_out_sample)
                        vali_mse = mse_
                        res_list.append([model,mse_,model.aic(),vali_mse,com_val,season_Flag,diff_val])
                        print("------In--------",com_val)
            res_list = np.array(res_list)
            aic_best = np.argmin(res_list[:,2])
            mse_best = np.argmin(res_list[:,1])
            test_mse_best = np.argmin(res_list[:,3])    
            print("AIC Best Index ",aic_best,"....",res_list[aic_best])
            print("MSE Best Index ",mse_best,"....",res_list[mse_best])
            print("TEST MSE Best Index ",mse_best,"....",res_list[test_mse_best])
            
            mse_params = res_list[mse_best][0].get_params()
            with open(param_file, "a") as curFile:
                    curFile.write(str(countr_)+"\t"+activity_+
                                 "\t"+str(res_list[mse_best][4])+"\t"+
                                 str(mse_params['order'])+"\t"+
                                 str(mse_params['seasonal_order'])+"\n")
            
#             rmse_params = res_list[test_mse_best][0].get_params()
#             with open(param_file_test, "a") as curFile:
#                     curFile.write(str(countr_)+"\t"+activity_+
#                                  "\t"+str(res_list[test_mse_best][4])+"\t"+
#                                  str(rmse_params['order'])+"\t"+
#                                  str(rmse_params['seasonal_order'])+"\n")
            
            
            with open(outDataFile, "a") as myfile:
                    myfile.write(str(countr_)+"\t AIC BEST "+str(aic_best)+"\t"+str(res_list[aic_best]) +"\t MSE BEST  "
                                 +str(mse_best)+"\t"+str(res_list[mse_best])+"\t TEST BEST "
                                 +str(test_mse_best)+"\t"+str(res_list[test_mse_best])+"\n")
                    myfile.write("---------------------------------------"+"\n")
        
        