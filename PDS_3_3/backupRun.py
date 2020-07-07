'''
Created on 19-Apr-2020

@author: Neeraj Badal
'''
'''
Created on 16-Mar-2020

@author: Neeraj Badal
'''
import sys
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error
if len (sys.argv) == 3:
        
        testDataFile = sys.argv[1]
        outDataFile = sys.argv[2]
        
        commonDir = "D:/Mtech/FY/SEM2/PDS/neeraj/data_pds_ml/"
#         teest_case_file = commonDir+"a3.testdat"
        
        modelFile = commonDir+'sklgbmII.sav'
        
        predFile = commonDir+"a3.preddat"
        
        
        ''' preparing test data file to matrix'''
        with open(testDataFile) as f:
            lines = f.readlines()
        
        num_of_test = lines[0]
        
        lines = lines[1:]    
        lines = [line.split(',') for line in lines]
        lines = [ [np.float(x) for x in line] for line in lines ]
        
        test_data_mat = np.array(lines)
        print("test file data read ")
        print("number of tests = ",len(test_data_mat))
        
        
#         data_std = test_data_mat - np.mean(test_data_mat,axis=0)
#         data_std = data_std / np.std(test_data_mat,axis=0)
    
#         test_std_dat = data_std
        
#         print(test_std_dat.shape)
        
        loaded_model = pickle.load(open(modelFile, 'rb'))
        print("model loaded")
        
        pred_model = loaded_model.predict(test_data_mat)
        
        print(pred_model)
        
        with open(predFile) as f:
            lines = f.readlines()
        
        lines = [ np.float(line) for line in lines ]
        
        pred_data_mat = np.array(lines)
        
        print(pred_data_mat)
        mse_train = mean_squared_error(pred_data_mat,pred_model)
        print(mse_train)
        np.savetxt(outDataFile,pred_model,fmt='%1.4f')
        print("output file written")
        exit(0)
        