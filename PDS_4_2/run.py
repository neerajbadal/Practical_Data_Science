'''
Created on 27-Apr-2020

@author: Neeraj Badal
'''
import numpy as np
import sys
import pickle
import datetime

if __name__ == "__main__":
    
    if len (sys.argv) == 4:
    
        print("working on predictions")
        commonDir = "D:/Mtech/FY/SEM2/PDS/neeraj/data_pds_ml/"
    
        start_date = sys.argv[1]
        end_date = sys.argv[2]
        outDataFolder = sys.argv[3]
        
        
        model_trained_till = "2020-04-28"
        
        
        ind_hist = ["2020-04-21 20080    3975    645",
                    "2020-04-22    21370    4370    681",
                    "2020-04-23    23077    5012    721",
                    "2020-04-24    24530    5498    780",
                    "2020-04-25    26283    5939    825",
                    "2020-04-26    27890    6523    881",
                    "2020-04-27    29451    7137    939",
                    "2020-04-28    31324    7747    1008",
                    "2020-04-29    33062    8437    1079"
                    ]
        
        usa_hist = [ "2020-04-21    811865    75204    44444",
                    "2020-04-22    840351    77366    46622",
                    "2020-04-23    869170    80203    49954",
                    "2020-04-24    905358    99079    51949",
                    "2020-04-25    938154    100372    53755",
                    "2020-04-26    965785    106988    54881",
                    "2020-04-27    988197    111424    56259",
                    "2020-04-28    1012582    115936    58355",
                    "2020-04-29    1039909    120720    60967"
            ]
        
        belgium_hist = [
                    "2020-04-21    40956    9002    5998",
                    "2020-04-22    41889    9433    6262",
                    "2020-04-23    42797    9800    6490",
                    "2020-04-24    44293    10122    6679",
                    "2020-04-25    45325    10417    6917",
                    "2020-04-26    46134    10785    7094",
                    "2020-04-27    46687    10878    7207",
                    "2020-04-28    47334    10943    7331",
                    "2020-04-29    47859    11283    7501"
            ]
        
        italy_hist = [
            "2020-04-21    183957    51600    24648",
            "2020-04-22    187327    54543    25085",
            "2020-04-23    189973    57576    25549",
            "2020-04-24    192994    60498    25969",
            "2020-04-25    195351    63120    26384",
            "2020-04-26    197675    64928    26644",
            "2020-04-27    199414    66624    26977",
            "2020-04-28    201505    68941    27359",
            "2020-04-29    203591    71252    27682"
            ]
        

        country_hist = [
            usa_hist,
            belgium_hist,
            italy_hist,
            ind_hist
            ]
        
        model_trained_till = datetime.datetime.strptime(model_trained_till,
                                                         "%Y-%m-%d")
        start_date = datetime.datetime.strptime(start_date,
                                                         "%Y-%m-%d")
        
        end_date = datetime.datetime.strptime(end_date,
                                                         "%Y-%m-%d")
        
        countryData = ["US.csv","Belgium.csv","Italy.csv","India.csv"]
        
        confirmationModel = ["usa_confirmed.mod","belgium_confirmed.mod",
                             "italy_confirmed.mod","india_confirmed.mod"]
        
        recoveryModel = ["usa_recovery.mod","belgium_recovery.mod",
                             "italy_recovery.mod","india_recovery.mod"]
        
        deathModel = ["usa_death.mod","belgium_death.mod",
                             "italy_death.mod","india_death.mod"]
        
        
        step = datetime.timedelta(days=1)
        dates_forecasted = []
        date_start = start_date 
        while date_start <= end_date:
            range_elem = date_start.date()
            dates_forecasted.append(range_elem.strftime("%Y-%m-%d"))
            date_start += step
        
        countryPredictions = ["/US.pred","/Belgium.pred","/Italy.pred","/India.pred"]
        
        
        dates_forecasted = np.array(dates_forecasted)
        for country_counter in [0,1,2,3]:
            ''' Loading  Confirmation Model '''
            confirmatation_model = pickle.load(open(commonDir+confirmationModel[country_counter], 'rb'))
            ''' Loading Recovery Model '''
            recovery_model = pickle.load(open(commonDir+recoveryModel[country_counter], 'rb'))
            ''' Loading Death Model '''
            death_model = pickle.load(open(commonDir+deathModel[country_counter], 'rb'))
        
            date_diff_train_test = 0 
            greaterFlag = start_date > model_trained_till
            
            if greaterFlag:
                
                date_diff_train_test = (start_date - model_trained_till).days
                forecast_range = (end_date - start_date).days + 1
                if date_diff_train_test == 1:
                    print("case : date range next to training by 1")
                    confirmation_forecast = confirmatation_model.forecast(steps=forecast_range)
                    confirmation_forecast = np.array(confirmation_forecast)
                    
                    recovery_forecast = recovery_model.forecast(steps=forecast_range)
                    recovery_forecast = np.array(recovery_forecast)
                    
                    death_forecast = death_model.forecast(steps=forecast_range)
                    death_forecast = np.array(death_forecast)
                    
                    with open(outDataFolder+"/"+countryPredictions[country_counter], "w") as myfile:
                        myfile.write("Date\tConfirmed\tRecovered\tDeaths\n")
                        for res_counter in range(0,len(confirmation_forecast)):
                            if res_counter == len(confirmation_forecast) - 1:
                                myfile.write(dates_forecasted[res_counter]+"\t"+
                                         str(confirmation_forecast[res_counter])+"\t"+
                                         str(recovery_forecast[res_counter])+"\t"+
                                         str(death_forecast[res_counter]))
                            else:
                                
                                myfile.write(dates_forecasted[res_counter]+"\t"+
                                             str(confirmation_forecast[res_counter])+"\t"+
                                             str(recovery_forecast[res_counter])+"\t"+
                                             str(death_forecast[res_counter])+"\n"
                                    )
                            
                    
                if date_diff_train_test > 1:
                    forecast_range += date_diff_train_test - 1
                    print("gap between train and test more than 1 ",date_diff_train_test)
                    
                    confirmation_forecast = confirmatation_model.forecast(steps=forecast_range)
                    confirmation_forecast = confirmation_forecast[date_diff_train_test - 1:]
                    confirmation_forecast = np.array(confirmation_forecast)
                    
                    recovery_forecast = recovery_model.forecast(steps=forecast_range)
                    recovery_forecast = recovery_forecast[date_diff_train_test - 1:]
                    recovery_forecast = np.array(recovery_forecast)
                    
                    death_forecast = death_model.forecast(steps=forecast_range)
                    death_forecast = death_forecast[date_diff_train_test - 1:]
                    death_forecast = np.array(death_forecast)
                    
                    with open(outDataFolder+"/"+countryPredictions[country_counter], "w") as myfile:
                        myfile.write("Date\tConfirmed\tRecovered\tDeaths\n")
                        for res_counter in range(0,len(confirmation_forecast)):
                            if res_counter == len(confirmation_forecast) - 1:
                                myfile.write(dates_forecasted[res_counter]+"\t"+
                                         str(confirmation_forecast[res_counter])+"\t"+
                                         str(recovery_forecast[res_counter])+"\t"+
                                         str(death_forecast[res_counter]))
                            else:
                                myfile.write(dates_forecasted[res_counter]+"\t"+
                                             str(confirmation_forecast[res_counter])+"\t"+
                                             str(recovery_forecast[res_counter])+"\t"+
                                             str(death_forecast[res_counter])+"\n"
                                    )
                    
#                     forecasts_2 = loaded_model.forecast(steps=forecast_range)
#     #                 forecasts_2 = np.array(forecasts_2)
#                     print(forecasts_2[date_diff_train_test - 1:])
            else:
                date_diff_train_test = (start_date - model_trained_till).days
                print("No. of days overlap",date_diff_train_test)
                forecast_range = (end_date - start_date).days + 1
                forecast_val_size = (end_date - start_date).days + 1
                
                history_data_size = len(country_hist[country_counter])
                first_hist_date = str(country_hist[country_counter][0]).split()
                first_hist_date = datetime.datetime.strptime(first_hist_date[0],"%Y-%m-%d")
                
                
                if start_date < first_hist_date:
                    print("Date before history data")
                    days_in_overlap = (start_date - model_trained_till).days
                    
                    stored_vals = str(country_hist[country_counter][0]).split()
                    stored_vals = datetime.datetime.strptime(stored_vals[0],"%Y-%m-%d")
                    stored_vals -= datetime.timedelta(days=1)
                    
                    confirmation_forecast = confirmatation_model.get_prediction(
                        start=start_date,end=stored_vals,dynamic=False)    
                    
                    confirmation_forecast = np.array(confirmation_forecast.predicted_mean)    
                    
                    recovery_forecast = recovery_model.get_prediction(
                        start=start_date,end=stored_vals,dynamic=False)
                    
                    recovery_forecast = np.array(recovery_forecast.predicted_mean)
                    
                    death_forecast = death_model.get_prediction(
                        start=start_date,end=stored_vals,dynamic=False)
                    death_forecast = np.array(death_forecast.predicted_mean)
                    
                    traverse_overlap = stored_vals + datetime.timedelta(days=1)
                    step_l = datetime.timedelta(days=1)
                    
                    tem_confirm = []
                    tem_recover = []
                    tem_death = []
                    
                    if end_date < model_trained_till:
                        copy_till = end_date 
                    else:
                        copy_till = model_trained_till
                    
                    while traverse_overlap <= copy_till:
                        searchDate = traverse_overlap.strftime("%Y-%m-%d")
                        
                        stored_vals = [hist_dat for hist_dat in country_hist[country_counter] if searchDate in hist_dat]
                        stored_vals = str(stored_vals[0]).split()
                        
                        tem_confirm.append(stored_vals[1])
                        tem_recover.append(stored_vals[2])
                        tem_death.append(stored_vals[3])
                        
                        traverse_overlap += step_l
                    tem_confirm = np.array(tem_confirm)
                    tem_recover = np.array(tem_recover)
                    tem_death = np.array(tem_death)
                    
                    confirmation_forecast = np.append(confirmation_forecast,tem_confirm,0)
                    recovery_forecast = np.append(recovery_forecast,tem_recover,0)
                    death_forecast = np.append(death_forecast,tem_death,0)
                    if len(confirmation_forecast) < forecast_val_size: 
                        forecast_range += days_in_overlap - 1
                        
                        confirmation_tmp = confirmatation_model.forecast(steps=forecast_range)
                        confirmation_tmp = np.array(confirmation_tmp)
                        confirmation_forecast = np.append(confirmation_forecast,confirmation_tmp,0)
                        
                        recovery_tmp = recovery_model.forecast(steps=forecast_range)
                        recovery_tmp = np.array(recovery_tmp)
                        recovery_forecast = np.append(recovery_forecast,recovery_tmp,0)
                        
                        death_tmp = death_model.forecast(steps=forecast_range)
                        death_tmp = np.array(death_tmp)
                        death_forecast = np.append(death_forecast,death_tmp,0)
                    
                    with open(outDataFolder+"/"+countryPredictions[country_counter], "w") as myfile:
                        myfile.write("Date\tConfirmed\tRecovered\tDeaths\n")
                        for res_counter in range(0,len(confirmation_forecast)):
                            if res_counter == len(confirmation_forecast) - 1:
                                myfile.write(dates_forecasted[res_counter]+"\t"+
                                         str(confirmation_forecast[res_counter])+"\t"+
                                         str(recovery_forecast[res_counter])+"\t"+
                                         str(death_forecast[res_counter]))
                            else:
                                myfile.write(dates_forecasted[res_counter]+"\t"+
                                             str(confirmation_forecast[res_counter])+"\t"+
                                             str(recovery_forecast[res_counter])+"\t"+
                                             str(death_forecast[res_counter])+"\n"
                                    )
                    
                     
                else:
                    
                    days_in_overlap = (start_date - model_trained_till).days
                    traverse_overlap = start_date
                    step_l = datetime.timedelta(days=1)
                    
                    confirmation_forecast = []
                    recovery_forecast = []
                    death_forecast=[]
                    
                    while traverse_overlap <= model_trained_till:
                        searchDate = traverse_overlap.strftime("%Y-%m-%d")
                        
                        stored_vals = [hist_dat for hist_dat in country_hist[country_counter] if searchDate in hist_dat]
                        stored_vals = str(stored_vals[0]).split()
                        
                        confirmation_forecast.append(stored_vals[1])
                        recovery_forecast.append(stored_vals[2])
                        death_forecast.append(stored_vals[3])
                        
                        traverse_overlap += step_l
                    
                    confirmation_forecast = np.array(confirmation_forecast)
                    recovery_forecast = np.array(recovery_forecast)
                    death_forecast =  np.array(death_forecast)
                    
                    if len(confirmation_forecast) < forecast_val_size: 
                        forecast_range += days_in_overlap - 1
                        
                        
                        confirmation_tmp = confirmatation_model.forecast(steps=forecast_range)
        #                 confirmation_tmp = confirmation_tmp[date_diff_train_test - 1:]
                        confirmation_tmp = np.array(confirmation_tmp)
                        confirmation_forecast = np.append(confirmation_forecast,confirmation_tmp,0)
                        
                        
                        recovery_tmp = recovery_model.forecast(steps=forecast_range)
        #                 recovery_tmp = recovery_tmp[date_diff_train_test - 1:]
                        recovery_tmp = np.array(recovery_tmp)
                        recovery_forecast = np.append(recovery_forecast,recovery_tmp,0)
                        
                        death_tmp = death_model.forecast(steps=forecast_range)
        #                 death_tmp = death_tmp[date_diff_train_test - 1:]
                        death_tmp = np.array(death_tmp)
                        death_forecast = np.append(death_forecast,death_tmp,0)
                    
                    with open(outDataFolder+"/"+countryPredictions[country_counter], "w") as myfile:
                        myfile.write("Date\tConfirmed\tRecovered\tDeaths\n")
                        for res_counter in range(0,len(confirmation_forecast)):
                            if res_counter == len(confirmation_forecast) - 1:
                                myfile.write(dates_forecasted[res_counter]+"\t"+
                                         str(confirmation_forecast[res_counter])+"\t"+
                                         str(recovery_forecast[res_counter])+"\t"+
                                         str(death_forecast[res_counter]))
                            else:
                                myfile.write(dates_forecasted[res_counter]+"\t"+
                                             str(confirmation_forecast[res_counter])+"\t"+
                                             str(recovery_forecast[res_counter])+"\t"+
                                             str(death_forecast[res_counter])+"\n"
                                    )
                
                