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
import pickle
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
        
        np.random.shuffle(train_mat)
        
        x_data = train_mat[:,1:]
        y_data = train_mat[:,0]
        
        
#         params_ = {'objective':'reg:squarederror', 'n_estimators':350,
#                        'tree_method':'approx','max_depth':6,
#                        'n_jobs':2,'min_child_weight':600}
#              
#         xgModel = xgb.XGBModel(**params_)
#          
#         print("start fitting")
#         xgModel.fit(x_data,y_data,eval_set=[(x_data,y_data)],
#                             eval_metric='rmse',verbose=True)
#          
#         filename = commonDir+'sklgbmIII.sav'
#         pickle.dump(xgModel, open(filename, 'wb'))
#         exit(0)
        
        
#         x_test = train_mat[400000:,1:]
#         y_test = train_mat[400000:,0]
        
        #lamba = 1.5 78.9,
        #min_child_weight = 200, 500
        reg_lamda = [500]
        
        test_err = []
        train_err = []
        
        for depth_ in reg_lamda:
        
            params_ = {'objective':'reg:squarederror', 'n_estimators':depth_,
                       'tree_method':'approx','max_depth':6,
                       'n_jobs':3,'min_child_weight':600}
            
            xgModel = xgb.XGBModel(**params_)
            
            print("start fitting")
            
            
            k_fold = KFold(5)
            
#             x_data, y_data = shuffle(x_data, y_data, random_state=10)
            mse_k_fold_train = []
            mse_k_fold_test = []
            fold_iter = 0
            
            test_bucket = []
            train_bucket = []
            
            for k, (train, test) in enumerate(k_fold.split(x_data, y_data)):
                
                xgModel.fit(x_data[train],y_data[train],eval_set=[(x_data[train],y_data[train]),(x_data[test],y_data[test])],
                            eval_metric='rmse',verbose=True)
                
                print("evaluation ended")
                result_ = xgModel.evals_result()
                fold_iter += 1
                print(fold_iter,"....",result_)
                print(result_['validation_0']['rmse'])
                print(result_['validation_1']['rmse'])
                train_bucket.append(result_['validation_0']['rmse'])
                test_bucket.append(result_['validation_1']['rmse'])
                
#                 mse_ = mean_squared_error(y_data[test], xgModel.predict(x_data[test]))
#                 mse_train = mean_squared_error(y_data[train], xgModel.predict(x_data[train]))
#                 print("train..........",mse_train)
#                 print("test..........",mse_)
#                 print("tree depth = ",depth_)
#                 mse_k_fold_test.append(mse_)
#                 mse_k_fold_train.append(mse_train)

                
#             filename = commonDir+'sklgbmII.sav'
#             pickle.dump(xgModel, open(filename, 'wb'))

#             test_err.append(np.mean(mse_k_fold_test))
#             train_err.append(np.mean(mse_k_fold_train))


#         xgModel.fit(x_data,y_data,eval_set=[(x_data,y_data),(x_test,y_test)],
#                     eval_metric='rmse',verbose=True)
        
#         print("evaluation ended")
#         result_ = xgModel.evals_result()
#         print(result_)
#         
#         pred_model = xgModel.predict(x_test)
#         mse_train = mean_squared_error(y_test,pred_model)
#         print(mse_train)
        
        
        
        train_bucket = np.array(train_bucket)
        test_bucket = np.array(test_bucket)
        
        train_bucket = train_bucket**2
        test_bucket = test_bucket**2
        
        train_mean = np.mean(train_bucket,axis=0)
        test_mean = np.mean(test_bucket,axis=0)
        
        plt.plot(range(0,len(train_mean)),train_mean,label='train error',marker='o')
        plt.plot(range(0,len(test_mean)),test_mean,label='test error',marker='o')
        
#         plt.plot(range(0,len(train_err)),train_err,label='train error',marker='o')
#         plt.plot(range(0,len(test_err)),test_err,label='test error',marker='o')
        
        
        plt.legend()
        plt.show()
        
        
#         params_ = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic'}
# 
#         watchlist = [(dtest, 'eval'), (dtrain, 'train')]
# 
#         num_round = 2
# 
#         bst = xgb.train(param, dtrain, num_round, watchlist)
        
        
        
#         filename = commonDir+'sklgbm.sav'
#         pickle.dump(est, open(filename, 'wb'))
#         
#         
#         mse_train = mean_squared_error(y_data, est.predict(x_data))
#         print(mse_train)