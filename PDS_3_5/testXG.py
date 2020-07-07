'''
Created on 15-Apr-2020

@author: Neeraj Badal
'''
'''
Created on 13-Apr-2020

@author: Neeraj Badal
'''
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from PDS_3_5 import run
if __name__ == '__main__':
        commonDir = "D:/Mtech/FY/SEM2/PDS/neeraj/data_pds_ml/"
        trainFile = commonDir+"/years.train"
        ''' preparing data file to matrix'''
        with open(trainFile) as f:
            lines = f.readlines()
            
        lines = [line.split(',') for line in lines]
        lines = [ [float(x) for x in line] for line in lines ]
        train_mat = np.array(lines)
        print(train_mat.shape)
        
        
#         x_data = train_mat[:400000,1:]
#         y_data = train_mat[:400000,0]
        
        x_data = train_mat[:,1:]
        y_data = train_mat[:,0]
        
#         x_test = train_mat[400000:,1:]
#         y_test = train_mat[400000:,0]
        
        #lamba = 1.5 78.9,
        #min_child_weight = 200, 500
        reg_lamda = [600,800,1000]
        
        test_err = []
        train_err = []
        
        print("start fitting")
        
        
        k_fold = KFold(5)
        
        x_data, y_data = shuffle(x_data, y_data, random_state=10)
        mse_k_fold_train = []
        mse_k_fold_test = []
        fold_iter = 0
        for k, (train, test) in enumerate(k_fold.split(x_data, y_data)):
            run.fit_xg(x_data[train],y_data[train],x_data[test],y_data[test])
            
            print("evaluation ended",k,"-------------------")
              