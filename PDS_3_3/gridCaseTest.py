'''
Created on 11-Mar-2020

@author: Neeraj Badal
'''
import numpy as np
from sklearn import ensemble,tree
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
import multiprocessing
from functools import partial
from sklearn.preprocessing import StandardScaler
import time
from sklearn.model_selection import GridSearchCV
import sys
if __name__ == '__main__':
        commonDir = sys.argv[1]#"D:/Mtech/FY/SEM2/PDS/neeraj/data_pds_ml/"
        
        trainFile = commonDir+"/years.train"
        ''' preparing data file to matrix'''
        with open(trainFile) as f:
            lines = f.readlines()
            
        lines = [line.split(',') for line in lines]
        lines = [ [float(x) for x in line] for line in lines ]
        train_mat = np.array(lines)
        print(train_mat.shape)
        
        x_data = train_mat[:,1:]
        y_data = train_mat[:,0]
       
       
        sc = StandardScaler()
        x_data = sc.fit_transform(x_data)
          
          

      
        x_data, y_data = shuffle(x_data, y_data, random_state=10)
        mse_k_fold = []
         
        params_grid = {'max_iter':[40,80,100,150,200,250,300,350],'max_depth': [4,6,8,10,12,16]
                 ,'min_samples_leaf':[200,400,600]}
        
#         'validation_fraction':0.1 
        params = {'learning_rate':0.1,
                  'l2_regularization':0.5,'n_iter_no_change':6, 
                  'loss': 'least_squares','scoring':'neg_mean_squared_error'}
        est = HistGradientBoostingRegressor(**params)
        grid = GridSearchCV(estimator=est, param_grid=params_grid,n_jobs=2,cv=5)
         
         
#         est = HistGradientBoostingRegressor(**params)
        est = grid.fit(x_data, y_data)
        print(grid.best_score_)
        print(grid.best_params_)
         
#         0.352292980473917
# {'max_depth': 10, 'max_iter': 350, 'min_samples_leaf': 200}

          
          
        exit(0) 
        
