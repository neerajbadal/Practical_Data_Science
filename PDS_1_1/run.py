'''
Created on 19-Jan-2020

@author: Neeraj Badal
'''
# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import time
import multiprocessing
from functools import partial
def readTrainData(trainDataFile):
#     data_ = open(trainDataFile, 'r').readlines()
    data_ = [line.strip('\n') for line in open(trainDataFile, 'r')]
    train_data = data_[1:]
    train_data = [user_rating.split('\t') for user_rating in train_data]
#     print(train_data)
    train_data = np.array(train_data).astype(np.float)
    print(train_data.shape)
#     print(train_data[0])
#     print("no. of users ",np.max(train_data[:,0]))
#     print("no. of movies ",np.max(train_data[:,1]))
    no_users = np.max(train_data[:,0])
    no_movies = np.max(train_data[:,1])
    '''
    each column user and movie starts with index 0 , so plus 1 while creating
    mat in main function
    '''
    
    return [train_data,int(no_users),int(no_movies)]


def partitionList(list_,partition_size):
    return [list_[i:i+partition_size] for i in range(0, len(list_), partition_size)]


def getCVForFoldIndex(foldIndex,omega_sample_folds,list_index_train):
    folds_index_train = list(list_index_train - set([foldIndex]))
    train_data = [item for fold_id in folds_index_train for item in omega_sample_folds[fold_id]]
    test_data = omega_sample_folds[foldIndex]
    return [train_data,test_data]


def SGD_Factorization(user_movie_mat,kFoldCrossVal_params,rank_k,beta_,iters=300,cycles=5):
    
    print("---------------------new rank-------------------------")
    print(" for rank_k = ",rank_k, " beta = ",beta_)
    data_not_null_mask = user_movie_mat > 0.0
    
    start = time.time()
    epsilon_ = 1e-9
    
    
    
    omega_sample_folds = kFoldCrossVal_params[0]
    list_index_train = kFoldCrossVal_params[1]
    noOfFolds = kFoldCrossVal_params[2]
    
    m_ = user_movie_mat.shape[0]
    n_ = user_movie_mat.shape[1]
    
    tolerance = 1e-9
    
    k_fold_res = []
    
#     for fold_ in range(noOfFolds):
#         print(" in fold index : ",fold_ )
        
        
        
#         k_fold_cv_data = getCVForFoldIndex(fold_, omega_sample_folds, list_index_train)
#         train_data_ = k_fold_cv_data[0]
#         test_data_ = k_fold_cv_data[1]
#         
#         for cycle_ind in range(cycles):
#             print(" in fold index : ",fold_," cycle ind : ",cycle_ind)
#             P = np.random.rand(m_, rank_k)
#             Q = np.random.rand(n_,rank_k)
#             P = np.maximum(P, 0.0)
#             Q = np.maximum(Q, 0.0)
#             
#             
#             R_approx_prev = None
#             R_approx = np.dot(P,np.transpose(Q))
#             
#             rel_error = 999.9
#             iter_ = 0
#             learning_rate = (0.001)*2.0
#             
#             while True:#np.fabs(rel_error) > tolerance:
# #                 learning_rate = learning_rate / float(iter_+1)
#                 train_error = 0.0
#                 for i,j,r_sample in train_data_:
#                     r_pred = np.dot(P[i],Q[j]) 
#                     error = r_sample - r_pred
#                     train_error += (error**2)
#                     
#                     error_mul = learning_rate*error
#                      
#                     P[i] = P[i] + (error_mul)*Q[j]
#                  
#     #                 P[i][P[i] < 0.0] = 0.0
#     #                 Q[j] = Q[j] + (error_mul)*P[i]
#     #                 Q[j][Q[j] < 0.0] = 0.0
#     
#                     P[i] = np.maximum(P[i], 0.0)
#                     Q[j] = Q[j] + (error_mul)*P[i]
#                     Q[j] = np.maximum(Q[j], 0.0)
#                 
#                 train_error = np.sqrt(train_error)
#                 R_approx_prev = R_approx
#                 R_approx = np.dot(P,np.transpose(Q))
#                 rel_error = data_not_null_mask * (R_approx_prev - R_approx)
#                 rel_error = np.sqrt(np.sum(rel_error**2))
#                 iter_ = iter_ + 1
#                 
#                 
#                 
#                 
#                 
#                 print("iter completed = ",iter_,"...",np.fabs(rel_error))
#                 if iter_ == iters:
#                     break
#                 
#             test_error = 0.0
#             print("computing test error ...")
#             for i,j,r_sample in test_data_:
#                 test_error += ((r_sample - R_approx[i,j])**2)
#             test_error = np.sqrt(test_error)
#         k_fold_res.append([train_error,test_error,P,Q])
   
    fold_data = [omega_sample_folds,list_index_train,cycles,m_,n_,rank_k,iters,data_not_null_mask,beta_]
    
    pool = multiprocessing.Pool(5)
    k_fold_res = pool.map(partial(performFoldWork,fold_data), np.arange(0,noOfFolds))
    pool.close()
    pool.join()
    
    mean_train_error = np.mean([x[0] for x in k_fold_res])

    mean_test_error = np.mean([x[1] for x in k_fold_res])
    
    P_s = [x[2] for x in k_fold_res]
    Q_s = [x[3] for x in k_fold_res]
    
                
    endtime = time.time()
    print("time taken : ",endtime-start)
    return [mean_train_error,mean_test_error,P_s,Q_s]
    



def performFoldWork(fold_data,fold_):
    omega_sample_folds = fold_data[0]
    list_index_train = fold_data[1]
    cycles = fold_data[2]
    m_ = fold_data[3]
    n_ = fold_data[4]
    rank_k = fold_data[5]
    iters = fold_data[6]
    data_not_null_mask = fold_data[7]
    beta_ = fold_data[8]
    
    k_fold_cv_data = getCVForFoldIndex(fold_, omega_sample_folds, list_index_train)
    train_data_ = k_fold_cv_data[0]
    test_data_ = k_fold_cv_data[1]
    
#     b_mean = np.mean(np.array(train_data_)[:,2])
     
        
    for cycle_ind in range(cycles):
        print(" in fold index : ",fold_," cycle ind : ",cycle_ind)
        P = np.random.rand(m_, rank_k)
        Q = np.random.rand(n_,rank_k)
        
#         P = np.random.uniform(0.5,5.0,[m_, rank_k])
#         Q = np.random.uniform(0.5,5.0,[n_,rank_k])
        
        P = np.maximum(P, 0.0)
        Q = np.maximum(Q, 0.0)
        
        
#         B_u = np.zeros(m_)
#         B_m = np.zeros(n_)
        
        
        R_approx_prev = None
        R_approx = np.dot(P,np.transpose(Q))
        
        rel_error = 999.9
        iter_ = 0
        learning_rate = (0.001)*2.0
        
        tolerance = 1e-3
        
        test_err_list = []
        train_err_list = []
        prev_test_error = 0.0
        test_error_increase_count = 0
        while np.fabs(rel_error) > tolerance:
#                 learning_rate = learning_rate / float(iter_+1)
#             np.random.shuffle(train_data_)
            train_error = 0.0
            for i,j,r_sample in train_data_:
                r_pred = np.dot(P[i],Q[j]) 
                error = r_sample - r_pred
                train_error += (error**2)
                
                error_mul = learning_rate*error
                 
                
#                 B_u[i] += learning_rate * (error - b_mean * B_u[i])
#                 B_m[j] += learning_rate * (error - b_mean * B_m[j])
                
#                 P[i] = P[i] + (error_mul)*Q[j]
                P[i] = P[i] + learning_rate*(error*Q[j] - beta_ * P[i])
             
                P[i][P[i] < 0.0] = 0.0
#                 Q[j] = Q[j] + (error_mul)*P[i]
                Q[j] = Q[j] + learning_rate*(error*P[i] - beta_ * Q[j])
                Q[j][Q[j] < 0.0] = 0.0

                
                


#                 P[i] = np.maximum(P[i], 0.0)
#                 Q[j] = Q[j] + (error_mul)*P[i]
#                 Q[j] = np.maximum(Q[j], 0.0)
            
            train_error = train_error / float(len(train_data_))
            train_error = np.sqrt(train_error)
            R_approx_prev = R_approx
#             R_approx = b_mean + B_u[:,np.newaxis] + B_m[np.newaxis:,] + np.dot(P,np.transpose(Q))
            R_approx = np.dot(P,np.transpose(Q))
            rel_error = 0.0
            for i,j,r_sample in train_data_:
#                 r_pred_ = R_approx[i,j] 
                rel_error += (R_approx[i,j] - R_approx_prev[i,j])**2
                
            
            
#             rel_error = data_not_null_mask * (R_approx_prev - R_approx)
#             rel_error = np.sqrt(np.sum(rel_error**2))
            rel_error = np.sqrt(rel_error)
            
            test_error = 0.0
            print("computing test error ...")
            for i,j,r_sample in test_data_:
                test_error += ((r_sample - R_approx[i,j])**2)
            test_error = test_error / float(len(test_data_))
            test_error = np.sqrt(test_error)
            
            test_err_list.append(test_error)
            train_err_list.append(train_error)
            
#             if iter_ > 2:
            diff_test_error = prev_test_error - test_error
            if  diff_test_error < 0 :
                test_error_increase_count += 1
#                 if  test_error_increase_count > 5:
#                     print("iter completed = ",iter_,"...",np.fabs(rel_error),"....",train_error,"....",prev_test_error,"...",test_error)
#                     break    
                
            prev_test_error = test_error
            iter_ = iter_ + 1
            
            
            
            
            
            print("iter completed = ",iter_,"...",np.fabs(rel_error),"....",train_error,"....",test_error)
            
            
            if iter_ == iters or test_error_increase_count > 6:
                break
            
#         test_error = 0.0
#         print("computing test error ...")
#         for i,j,r_sample in test_data_:
#             test_error += ((r_sample - R_approx[i,j])**2)
#         test_error = np.sqrt(test_error)

#     pred_list = []
#     train_list = []
#     for i,j,r_sample in train_data_:
#         pred_list.append(R_approx[i,j])
#         train_list.append(r_sample)
# #         print(r_sample,"...",R_approx[i,j])

#     plt.plot(train_err_list,marker='o',label='train_error')
#     plt.plot(test_err_list,marker='o',label='test_error')
#     plt.legend()
#     plt.show()
    return [train_error,test_error,P,Q]
    
    
def trainEntireData(user_movie_mat,kFoldCrossVal_params,rank_k,beta_,iters=300):
    omega_sample_folds = kFoldCrossVal_params[0]
    list_index_train = kFoldCrossVal_params[1]
    noOfFolds = kFoldCrossVal_params[2]
    
    m_ = user_movie_mat.shape[0]
    n_ = user_movie_mat.shape[1]
    
    P = np.random.rand(m_, rank_k)
    Q = np.random.rand(n_,rank_k)
        
#         P = np.random.uniform(0.5,5.0,[m_, rank_k])
#         Q = np.random.uniform(0.5,5.0,[n_,rank_k])
        
    P = np.maximum(P, 0.0)
    Q = np.maximum(Q, 0.0)
        
    train_data_ = omega_sample_folds    
    R_approx_prev = None
    R_approx = np.dot(P,np.transpose(Q))
    
    rel_error = 999.9
    iter_ = 0
#     learning_rate = (0.001)*2.0
    learning_rate = (0.001)
    
    tolerance = 1e-3
    
    test_err_list = []
    train_err_list = []
    prev_test_error = 0.0
    test_error_increase_count = 0
    while np.fabs(rel_error) > tolerance:
        train_error = 0.0
        for i,j,r_sample in train_data_:
            r_pred = np.dot(P[i],Q[j]) 
            error = r_sample - r_pred
            train_error += (error**2)
            
            error_mul = learning_rate*error
             
            P[i] = P[i] + learning_rate*(error*Q[j] - beta_ * P[i])
         
            P[i][P[i] < 0.0] = 0.0
#                 Q[j] = Q[j] + (error_mul)*P[i]
            Q[j] = Q[j] + learning_rate*(error*P[i] - beta_ * Q[j])
            Q[j][Q[j] < 0.0] = 0.0

                    
        train_error = train_error / float(len(train_data_))
        train_error = np.sqrt(train_error)
        R_approx_prev = R_approx
#             R_approx = b_mean + B_u[:,np.newaxis] + B_m[np.newaxis:,] + np.dot(P,np.transpose(Q))
        R_approx = np.dot(P,np.transpose(Q))
        rel_error = 0.0
        for i,j,r_sample in train_data_:
#                 r_pred_ = R_approx[i,j] 
            rel_error += (R_approx[i,j] - R_approx_prev[i,j])**2
             
         
         
#             rel_error = data_not_null_mask * (R_approx_prev - R_approx)
#             rel_error = np.sqrt(np.sum(rel_error**2))
        rel_error = np.sqrt(rel_error)
#         
#         test_error = 0.0
#         print("computing test error ...")
#         for i,j,r_sample in test_data_:
#             test_error += ((r_sample - R_approx[i,j])**2)
#         test_error = test_error / float(len(test_data_))
#         test_error = np.sqrt(test_error)
#         
#         test_err_list.append(test_error)
        train_err_list.append(train_error)
        
#             if iter_ > 2:
#         diff_test_error = prev_test_error - test_error
#         if  diff_test_error < 0 :
#             test_error_increase_count += 1
#                 if  test_error_increase_count > 5:
#                     print("iter completed = ",iter_,"...",np.fabs(rel_error),"....",train_error,"....",prev_test_error,"...",test_error)
#                     break    
            
#         prev_test_error = test_error
        iter_ = iter_ + 1
        
        
        print("iter completed = ",iter_,"...",np.fabs(rel_error),"....",train_error)
        
        if iter_ == iters:
            break
    
    return [P,Q]


def performSGD(rank_k,user_movie_mat):
    print('--------------------')
    print("k = ",rank_k)
    iters = 20
    tolerance = 1e-2
    m_ = user_movie_mat.shape[0]
    n_ = user_movie_mat.shape[1]
    print(tolerance)
    learning_rate = (0.001)*2.0
#     P = np.random.normal(scale=1./rank_k, size=(m_,rank_k)).astype(np.float32)
#     Q = np.random.normal(scale=1./rank_k, size=(n_,rank_k)).astype(np.float32) 
    
    P = np.random.rand(m_, rank_k)
    Q = np.random.rand(n_,rank_k)
    #list of omegas
    omega_samples = [
            (i, j, user_movie_mat[i, j])
            for i in range(m_) for j in range(n_) if user_movie_mat[i, j] > 0 ]
    
    
    partition_list = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]
    
    folds_size = 5
    noOfSamplesInFold = int(np.ceil(len(omega_samples) / float(folds_size)))
    
    
    omega_sample_folds = partition_list(omega_samples,noOfSamplesInFold) 
    prev_error = 0.0
    
    list_index_train = set(np.arange(0,folds_size))
    
    
    for iter_ in range(0,iters):
        print("iter : ",iter_)
        for fold_iter in range(0,folds_size):
            folds_index_train = list(list_index_train - set([fold_iter]))
            train_data = [item for fold_id in folds_index_train for item in omega_sample_folds[fold_id]]
            np.random.shuffle(train_data)
            learning_rate = learning_rate / (iter_+1)
            for i,j,r_sample in train_data:
                r_pred = np.dot(P[i],Q[j]) 
                error = r_sample - r_pred
                error_mul = learning_rate*error
                 
                P[i] = P[i] + (error_mul)*Q[j]
                P[i][P[i] < 0.0] = 0.0
             
                Q[j] = Q[j] + (error_mul)*P[i]
                Q[j][Q[j] < 0.0] = 0.0
             
             
         
#         R_approx = np.matmul(P,np.transpose(Q))
#         xs, ys = user_movie_mat.nonzero()
#         error = 0
#         for x, y in zip(xs, ys):
#             error += pow(user_movie_mat[x, y] - R_approx[x, y], 2)
#           
#         error = np.sqrt(error)
# #         print("iter ",iter_)
#         if iter_ % 10 == 0:
#             print("iter ",iter_,"  error ",error)
#         if np.fabs(error-prev_error) <= tolerance:
#             print("iter ",iter_,"  error ",error)
#             return error
#         prev_error = error
#     print("iter ",iter_,"  error ",error)
    return 0.0
    #19371
if __name__ == '__main__':
#     trainDataFile = "D:/Mtech/SY/PDS/a1/ratings.train"
    
#     homeDir = "/home2/e0268-9/"
    homeDir = "D:/Mtech/SY/PDS/a1/"
    
    
    trainDataFile = homeDir+"/ratings.train"
    
#     trainDataFile = "/home2/e0268-9/ratings.train"


    import pandas as pd
    train_error = "D:/Mtech/SY/PDS/a1/train_err_rank_wise - Copy.dat"
    test_error = "D:/Mtech/SY/PDS/a1/test_err_rank_wise - Copy.dat"
     
    r_wise_pred = "D:/Mtech/SY/PDS/a1/r_wise_pred.dat"
     
    train_err = pd.read_csv(train_error,header=None,delimiter='\n')
    test_err = pd.read_csv(test_error,header=None,delimiter='\n')
      
    pred_ =  pd.read_csv(r_wise_pred,header=None,delimiter='\t')
      
    print(len(test_err),"...",test_err) 
      
#     plt.scatter(np.arange(2,len(train_err)+2),train_err,label="train")
#     plt.plot(np.arange(2,30),test_err[:28],marker='o',label="test_1")
#     plt.plot(np.arange(2,30),test_err[28:56],marker='o',label="test_2")
#     plt.plot(np.arange(2,30),test_err[56:84],marker='o',label="test_3")
#     plt.plot(np.arange(2,30),test_err[84:],marker='o',label="test_4")
#      
#      
#     plt.plot(np.arange(2,30),train_err[:28],marker='o',label="train_1")
#     plt.plot(np.arange(2,30),train_err[28:56],marker='o',label="train_2")
#     plt.plot(np.arange(2,30),train_err[56:84],marker='o',label="train_3")
#     plt.plot(np.arange(2,30),train_err[84:],marker='o',label="train_4")
#      
#     plt.legend()
#     plt.show()
# #     print(train_err)
#     exit(0)
 
 
 
 
    testDataFile = homeDir+"/ratings.test"
    test_data_prac = [line.strip('\n') for line in open(testDataFile, 'r')]
     
    predictFile = homeDir+"/ratings.pred"
    pred_data_prac = [line.strip('\n') for line in open(predictFile, 'r')]
 
 
    test_data_prac = test_data_prac[1:]
    test_data_prac = [test_data.split('\t') for test_data in test_data_prac]
#     print(train_data)
    test_data_prac = np.array(test_data_prac).astype(np.int)
     
    pred_data_prac = np.array(pred_data_prac).astype(np.float)
#     
# #     print(test_data_prac)
#     
# #     print(pred_data_prac)
#     
#     pred_ = np.array(pred_.values)
#     
# #     for i_ in range(0,pred_.shape[1]):
#     i_ = 8
#     plt.plot(pred_[:,i_],marker='x',label=str(i_+1))
#     
#     plt.plot(pred_data_prac,marker='o',label="preds_")
#     plt.legend()
#     plt.show()
#     print(pred_)
#     
# 
#     exit(0)

    user_rating_pack = readTrainData(trainDataFile)
    no_users = user_rating_pack[1]
    no_movies = user_rating_pack[2]
    user_rating_data = user_rating_pack[0]
    user_movie_mat = np.zeros((no_users+1,no_movies+1),np.float)
    for user_dat in user_rating_data:
        user_movie_mat[int(user_dat[0]),int(user_dat[1])] = user_dat[2]
    
    
    
    
    print(user_movie_mat.shape)
#     user_movie_mat = np.array([
#     [5, 3, 0, 1],
#     [4, 0, 0, 1],
#     [1, 1, 0, 5],
#     [1, 0, 0, 4],
#     [0, 1, 5, 4],
#     ])
    
    omega_samples = [
            (i, j, user_movie_mat[i, j])
            for i in range(user_movie_mat.shape[0]) for j in range(user_movie_mat.shape[1]) if user_movie_mat[i, j] > 0 ]
    
    
    np.random.shuffle(omega_samples)
    
    folds_size = 5
    noOfSamplesInFold = int(np.ceil(len(omega_samples) / float(folds_size)))
    omega_sample_folds = partitionList(omega_samples,noOfSamplesInFold)
    
#     print(len(omega_sample_folds))
#     print(len(omega_sample_folds[0]),"....",len(omega_sample_folds[4]))
    
    noOfFolds = folds_size
    list_index_train = set(np.arange(0,folds_size))
    kFoldCrossVal_params = [omega_sample_folds,list_index_train,noOfFolds]
    
#     kFoldCrossVal_params_1 = [omega_samples,list_index_train,noOfFolds]
#     
#     errors_ = trainEntireData(user_movie_mat,kFoldCrossVal_params_1, 12,0.2, iters=100)
#     
#     R_approx_rank_wise = np.dot(errors_[0],np.transpose(errors_[1]))
#     R_approx_rank_wise = np.array(R_approx_rank_wise)
    
    
    
    
    
    rank_k_lim = 13
#     [0.01,0.02,0.08,.1]
#     errors_ = [SGD_Factorization(user_movie_mat,kFoldCrossVal_params, k_rank,beta_val, iters=100, cycles=1 ) for beta_val in [0.01,0.02,0.04,0.1] for k_rank in range(2,rank_k_lim)]
    errors_ = [SGD_Factorization(user_movie_mat,kFoldCrossVal_params, k_rank,0.1, iters=100, cycles=1 ) for k_rank in range(12,rank_k_lim)]
     
    R_approx_rank_wise = [np.dot(x[2][0],np.transpose(x[3][0])) for x in errors_]
#     
    R_approx_rank_wise = np.array(R_approx_rank_wise)


#     
# #     print(test_data_prac[0,0])
#     
#     print(R_approx_rank_wise[0])
#     print(user_movie_mat)
#     exit(0)
#     print(R_approx_rank_wise[0][test_data_prac[0,0]][test_data_prac[0,1]])
    
    predicted_by_model = [ R_approx_rank_wise[test_data_prac[i_,0]][test_data_prac[i_,1]] for i_ in range(len(test_data_prac))]
     
     
#     predicted_by_model = [ [model_[test_data_prac[i_,0]][test_data_prac[i_,1]] for i_ in range(len(test_data_prac))] for model_ in R_approx_rank_wise 
#                         ] 
    predicted_by_model = np.array(predicted_by_model)
#     
#     np.savetxt(homeDir+"/r_wise_pred.dat",predicted_by_model.T, delimiter='\t')
# #     print(R_approx_rank_wise[:][test_data_prac[:,0],test_data_prac[:,1]],"...",pred_data_prac[:])
#     
    print(predicted_by_model)
    
    pred_e = (predicted_by_model[0] - pred_data_prac)**2
    pred_e = np.sqrt(np.sum(pred_e))
    print(pred_e)
    
    exit(0)
#     plt.plot(predicted_by_model[0],marker='x',label="pred")
#     plt.plot(pred_data_prac,marker='o',label="known")
#     plt.legend()
#     plt.show()
    
    
    train_err_rank_wise = [x[0] for x in errors_]
    test_err_rank_wise = [x[1] for x in errors_]
    
#     np.savetxt(homeDir+"/train_err_rank_wise.dat",train_err_rank_wise)
#     np.savetxt(homeDir+"/test_err_rank_wise.dat",test_err_rank_wise)
    
    
#     plt.scatter(np.arange(10,rank_k_lim),train_err_rank_wise,label="train-loss")
#     plt.scatter(np.arange(10,rank_k_lim),test_err_rank_wise,label="validation-loss")
#     plt.legend()
#     plt.show()
#     np.savetxt("/home2/e0268-9/train_err_rank_wise.dat",train_err_rank_wise)
#     np.savetxt("/home2/e0268-9/test_err_rank_wise.dat",test_err_rank_wise)
    
    
#     for fold_ in range(noOfFolds):
#         k_fold_cv_data = getCVForFoldIndex(fold_, omega_sample_folds, list_index_train)
#         print("len ",len(k_fold_cv_data[0]),"....",len(k_fold_cv_data[1]))
        
    
    
#     performSGD(50,user_movie_mat)
#     error_ = [performSGD(k,user_movie_mat) for k in range(1,21)]
#     plt.scatter(np.arange(1,21),error_)
#     plt.show()
    
    
    
    
    
    
# '''
# Created on 19-Jan-2020
# 
# @author: Neeraj Badal
# '''
# # import pandas as pd
# import numpy as np
# import sys
# import matplotlib.pyplot as plt
# import matplotlib.pylab as pylab
# import time
# import multiprocessing
# from functools import partial
# def readTrainData(trainDataFile):
#     '''
#     utility function to read training data.
#     returns : 
#     a list containing train data array, no. of users and no. of movies
#     '''
#     
#     data_ = [line.strip('\n') for line in open(trainDataFile, 'r')]
#     train_data = data_[1:]
#     train_data = [user_rating.split('\t') for user_rating in train_data]
#     train_data = np.array(train_data).astype(np.float)
#     print(train_data.shape)
#     no_users = np.max(train_data[:,0])
#     no_movies = np.max(train_data[:,1])
#     '''
#     each column user and movie starts with index 0 , so plus 1 while creating
#     mat in main function
#     '''
#     
#     return [train_data,int(no_users),int(no_movies)]
# 
# 
# def readTestData(testDataFile):
#     '''
#     utility function to read test data.
#     returns : test data array
#     '''
#     data_ = [line.strip('\n') for line in open(testDataFile, 'r')]
#     test_data_ = data_[1:]
#     test_data_ = [user_rating.split('\t') for user_rating in test_data_]
#     test_data_ = np.array(test_data_)[:,:2].astype(np.int)
# 
#     print("no. of test samples = ",test_data_.shape)
#     
#     return test_data_
# 
# 
# 
# 
# def partitionList(list_,partition_size):
#     '''
#     partitions the list on based on provided partition_size
#     returns : list containing partitions of equal size except may be the last one
#     '''
#     return [list_[i:i+partition_size] for i in range(0, len(list_), partition_size)]
# 
# 
# def getCVForFoldIndex(foldIndex,omega_sample_folds,list_index_train):
#     '''
#     prepares the hold out block and training block for performing
#     K-Fold Cross Validation and returns the same.
#     '''
#     folds_index_train = list(list_index_train - set([foldIndex]))
#     train_data = [item for fold_id in folds_index_train for item in omega_sample_folds[fold_id]]
#     test_data = omega_sample_folds[foldIndex]
#     return [train_data,test_data]
# 
# 
# def SGD_Factorization(user_movie_mat,kFoldCrossVal_params,rank_k,beta_,iters=300,cycles=5):
#     '''
#     Perform SGD iterations based on K-Fold data
#     '''
#     print("---------------------new rank-------------------------")
#     print(" for rank_k = ",rank_k, " beta = ",beta_)
#     data_not_null_mask = user_movie_mat > 0.0
#     
#     start = time.time()
#     omega_sample_folds = kFoldCrossVal_params[0]
#     list_index_train = kFoldCrossVal_params[1]
#     noOfFolds = kFoldCrossVal_params[2]
#     
#     m_ = user_movie_mat.shape[0]
#     n_ = user_movie_mat.shape[1]
#     
#     fold_data = [omega_sample_folds,list_index_train,cycles,m_,n_,rank_k,iters,data_not_null_mask,beta_]
#     
# #     performFoldWork(fold_data,0)
#     
# #     exit(0)
#     pool = multiprocessing.Pool(5)
#     k_fold_res = pool.map(partial(performFoldWork,fold_data), np.arange(0,noOfFolds))
#     pool.close()
#     pool.join()
#     
#     mean_train_error = np.mean([x[0] for x in k_fold_res])
# 
#     mean_test_error = np.mean([x[1] for x in k_fold_res])
#     
#     mean_rel_error = np.mean([x[4] for x in k_fold_res])
#     
#     
#     P_s = [x[2] for x in k_fold_res]
#     Q_s = [x[3] for x in k_fold_res]
#     
#     
#     f=open("D:/Mtech/SY/PDS/a1/"+"/err_dile_ent.dat",'a+')
#      
#     f.write("%f\t%f\t%f\t%f\t%f\n"%(beta_,rank_k,mean_train_error,mean_test_error,mean_rel_error))
#      
# #     np.savetxt(f,mean_train_error)
# #     np.savetxt(f,mean_train_error)
# #     np.savetxt(f,mean_train_error)
#      
# #     np.savetxt("/home2/e0268-9/"+"/test_err_"+str(beta_)+"_"+str(rank_k)+".dat",mean_test_error)
# #     np.savetxt("/home2/e0268-9/"+"/rel_err_"+str(beta_)+"_"+str(rank_k)+".dat",mean_rel_error)
#      
#     f.close()
#     
#                 
#     endtime = time.time()
#     print("time taken : ",endtime-start)
#     return [mean_train_error,mean_test_error,P_s,Q_s,mean_rel_error]
#     
# 
# 
# 
# def performFoldWork(fold_data,fold_):
#     '''
#     performs actual SGD work for each fold iterations.
#     '''
#     omega_sample_folds = fold_data[0]
#     list_index_train = fold_data[1]
#     cycles = fold_data[2]
#     m_ = fold_data[3]
#     n_ = fold_data[4]
#     rank_k = fold_data[5]
#     iters = fold_data[6]
#     data_not_null_mask = fold_data[7]
#     beta_ = fold_data[8]
#     
#     k_fold_cv_data = getCVForFoldIndex(fold_, omega_sample_folds, list_index_train)
#     train_data_ = k_fold_cv_data[0]
#     test_data_ = k_fold_cv_data[1]
#     
#         
#     for cycle_ind in range(cycles):
#         print(" in fold index : ",fold_," cycle ind : ",cycle_ind)
#         P = np.random.rand(m_, rank_k)
#         Q = np.random.rand(n_,rank_k)
#         
# #         P = np.random.uniform(0.5,5.0,[m_, rank_k])
# #         Q = np.random.uniform(0.5,5.0,[n_,rank_k])
#         
#         P = np.maximum(P, 0.0)
#         Q = np.maximum(Q, 0.0)
#         
#         
#         R_approx_prev = None
#         R_approx = np.dot(P,np.transpose(Q))
#         
#         rel_error = 999.9
#         iter_ = 0
#         learning_rate = (0.001)*2.0
#         
#         tolerance = 1e-3
#         
#         test_err_list = []
#         train_err_list = []
#         prev_test_error = 0.0
#         test_error_increase_count = 0
#         while np.fabs(rel_error) > tolerance:
# #                 learning_rate = learning_rate / float(iter_+1)
#             np.random.shuffle(train_data_)
#             train_error = 0.0
#             for i,j,r_sample in train_data_:
#                 r_pred = np.dot(P[i],Q[j]) 
#                 error = r_sample - r_pred
#                 train_error += (error**2)
#                 
#                 error_mul = learning_rate*error
#                  
#                 
# #                 B_u[i] += learning_rate * (error - b_mean * B_u[i])
# #                 B_m[j] += learning_rate * (error - b_mean * B_m[j])
#                 
# #                 P[i] = P[i] + (error_mul)*Q[j]
#                 P[i] = P[i] + learning_rate*(error*Q[j] - beta_ * P[i])
#              
#                 P[i][P[i] < 0.0] = 0.0
# #                 Q[j] = Q[j] + (error_mul)*P[i]
#                 Q[j] = Q[j] + learning_rate*(error*P[i] - beta_ * Q[j])
#                 Q[j][Q[j] < 0.0] = 0.0
# 
#                 
#                 
# 
# 
# #                 P[i] = np.maximum(P[i], 0.0)
# #                 Q[j] = Q[j] + (error_mul)*P[i]
# #                 Q[j] = np.maximum(Q[j], 0.0)
#             
#             train_error = train_error / float(len(train_data_))
#             train_error = np.sqrt(train_error)
#             R_approx_prev = R_approx
# #             R_approx = b_mean + B_u[:,np.newaxis] + B_m[np.newaxis:,] + np.dot(P,np.transpose(Q))
#             R_approx = np.dot(P,np.transpose(Q))
#             rel_error = 0.0
#             for i,j,r_sample in train_data_:
# #                 r_pred_ = R_approx[i,j] 
#                 rel_error += (R_approx[i,j] - R_approx_prev[i,j])**2
#                 
#             
#             
# #             rel_error = data_not_null_mask * (R_approx_prev - R_approx)
# #             rel_error = np.sqrt(np.sum(rel_error**2))
# #             rel_error = np.sqrt(rel_error)
#             
#             test_error = 0.0
#             print("computing test error ...")
#             for i,j,r_sample in test_data_:
#                 test_error += ((r_sample - R_approx[i,j])**2)
#             test_error = test_error / float(len(test_data_))
#             test_error = np.sqrt(test_error)
#             
#             test_err_list.append(test_error)
#             train_err_list.append(train_error)
#             
# #             if iter_ > 2:
#             diff_test_error = prev_test_error - test_error
#             if  diff_test_error < 0 :
#                 test_error_increase_count += 1
# #                 if  test_error_increase_count > 5:
# #                     print("iter completed = ",iter_,"...",np.fabs(rel_error),"....",train_error,"....",prev_test_error,"...",test_error)
# #                     break    
#                 
#             prev_test_error = test_error
#             iter_ = iter_ + 1
#             
#             
#             
#             
#             
#             print("iter completed = ",iter_,"...",np.fabs(rel_error),"....",train_error,"....",test_error)
#             
#             
#             if iter_ == iters or test_error_increase_count > 6:
#                 break
#             
#     return [train_error,test_error,P,Q,rel_error]
#     
#     
# def trainEntireData(user_movie_mat,kFoldCrossVal_params,rank_k,beta_,iters=300):
#     '''
#     Trains the model using the entire data set available
#     '''
#     
#     omega_sample_folds = kFoldCrossVal_params[0]
#     list_index_train = kFoldCrossVal_params[1]
#     noOfFolds = kFoldCrossVal_params[2]
#     
#     m_ = user_movie_mat.shape[0]
#     n_ = user_movie_mat.shape[1]
#     
#     P = np.random.rand(m_, rank_k)
#     Q = np.random.rand(n_,rank_k)
#         
# #         P = np.random.uniform(0.5,5.0,[m_, rank_k])
# #         Q = np.random.uniform(0.5,5.0,[n_,rank_k])
#         
#     P = np.maximum(P, 0.0)
#     Q = np.maximum(Q, 0.0)
#         
#     train_data_ = omega_sample_folds    
#     R_approx_prev = None
#     R_approx = np.dot(P,np.transpose(Q))
#     
#     rel_error = 999.9
#     iter_ = 0
# #     learning_rate = (0.001)*2.0
#     learning_rate = (0.001)
#     
#     tolerance = 1e-3
#     
#     test_err_list = []
#     train_err_list = []
#     prev_test_error = 0.0
#     test_error_increase_count = 0
#     while np.fabs(rel_error) > tolerance:
#         train_error = 0.0
#         np.random.shuffle(train_data_)
#         for i,j,r_sample in train_data_:
#             r_pred = np.dot(P[i],Q[j]) 
#             error = r_sample - r_pred
#             train_error += (error**2)
#             
#             error_mul = learning_rate*error
#              
#             P[i] = P[i] + learning_rate*(error*Q[j] - beta_ * P[i])
#          
#             P[i][P[i] < 0.0] = 0.0
# #                 Q[j] = Q[j] + (error_mul)*P[i]
#             Q[j] = Q[j] + learning_rate*(error*P[i] - beta_ * Q[j])
#             Q[j][Q[j] < 0.0] = 0.0
# 
#                     
#         train_error = train_error / float(len(train_data_))
#         train_error = np.sqrt(train_error)
#         R_approx_prev = R_approx
# #             R_approx = b_mean + B_u[:,np.newaxis] + B_m[np.newaxis:,] + np.dot(P,np.transpose(Q))
#         R_approx = np.dot(P,np.transpose(Q))
#         rel_error = 0.0
#         for i,j,r_sample in train_data_:
# #                 r_pred_ = R_approx[i,j] 
#             rel_error += (R_approx[i,j] - R_approx_prev[i,j])**2
#              
#          
#          
# #             rel_error = data_not_null_mask * (R_approx_prev - R_approx)
# #             rel_error = np.sqrt(np.sum(rel_error**2))
#         rel_error = np.sqrt(rel_error)
# #         
# #         test_error = 0.0
# #         print("computing test error ...")
# #         for i,j,r_sample in test_data_:
# #             test_error += ((r_sample - R_approx[i,j])**2)
# #         test_error = test_error / float(len(test_data_))
# #         test_error = np.sqrt(test_error)
# #         
# #         test_err_list.append(test_error)
#         train_err_list.append(train_error)
#         
# #             if iter_ > 2:
# #         diff_test_error = prev_test_error - test_error
# #         if  diff_test_error < 0 :
# #             test_error_increase_count += 1
# #                 if  test_error_increase_count > 5:
# #                     print("iter completed = ",iter_,"...",np.fabs(rel_error),"....",train_error,"....",prev_test_error,"...",test_error)
# #                     break    
#             
# #         prev_test_error = test_error
#         iter_ = iter_ + 1
#         
#         
#         print("iter completed = ",iter_,"...",np.fabs(rel_error),"....",train_error)
#         
#         if iter_ == iters:
#             break
#     
#     return [P,Q]
# 
# 
# def performSGD(rank_k,user_movie_mat):
#     print('--------------------')
#     print("k = ",rank_k)
#     iters = 20
#     tolerance = 1e-2
#     m_ = user_movie_mat.shape[0]
#     n_ = user_movie_mat.shape[1]
#     print(tolerance)
#     learning_rate = (0.001)*2.0
# #     P = np.random.normal(scale=1./rank_k, size=(m_,rank_k)).astype(np.float32)
# #     Q = np.random.normal(scale=1./rank_k, size=(n_,rank_k)).astype(np.float32) 
#     
#     P = np.random.rand(m_, rank_k)
#     Q = np.random.rand(n_,rank_k)
#     #list of omegas
#     omega_samples = [
#             (i, j, user_movie_mat[i, j])
#             for i in range(m_) for j in range(n_) if user_movie_mat[i, j] > 0 ]
#     
#     
#     partition_list = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]
#     
#     folds_size = 5
#     noOfSamplesInFold = int(np.ceil(len(omega_samples) / float(folds_size)))
#     
#     
#     omega_sample_folds = partition_list(omega_samples,noOfSamplesInFold) 
#     prev_error = 0.0
#     
#     list_index_train = set(np.arange(0,folds_size))
#     
#     
#     for iter_ in range(0,iters):
#         print("iter : ",iter_)
#         for fold_iter in range(0,folds_size):
#             folds_index_train = list(list_index_train - set([fold_iter]))
#             train_data = [item for fold_id in folds_index_train for item in omega_sample_folds[fold_id]]
#             np.random.shuffle(train_data)
#             learning_rate = learning_rate / (iter_+1)
#             for i,j,r_sample in train_data:
#                 r_pred = np.dot(P[i],Q[j]) 
#                 error = r_sample - r_pred
#                 error_mul = learning_rate*error
#                  
#                 P[i] = P[i] + (error_mul)*Q[j]
#                 P[i][P[i] < 0.0] = 0.0
#              
#                 Q[j] = Q[j] + (error_mul)*P[i]
#                 Q[j][Q[j] < 0.0] = 0.0
#              
#              
#          
# #         R_approx = np.matmul(P,np.transpose(Q))
# #         xs, ys = user_movie_mat.nonzero()
# #         error = 0
# #         for x, y in zip(xs, ys):
# #             error += pow(user_movie_mat[x, y] - R_approx[x, y], 2)
# #           
# #         error = np.sqrt(error)
# # #         print("iter ",iter_)
# #         if iter_ % 10 == 0:
# #             print("iter ",iter_,"  error ",error)
# #         if np.fabs(error-prev_error) <= tolerance:
# #             print("iter ",iter_,"  error ",error)
# #             return error
# #         prev_error = error
# #     print("iter ",iter_,"  error ",error)
#     return 0.0
#     #19371
# if __name__ == '__main__':
#     
# #     homeDir = "D:/Mtech/SY/PDS/a1/"
# #     import pandas as pd
# #     errFile = homeDir + "/err_dile_ent.dat"
# #     err_dat = pd.read_csv(errFile,header=None,delimiter='\t')
# #     print(err_dat)
# #     err_dat = err_dat.values.tolist()
# #     err_folds = partitionList(err_dat,15)
# #     print(len(err_folds))
# #     x_range_ = np.arange(5,20)
# #     err_folds = np.array(err_folds)
# #     for ind in range(0,len(err_folds)):
# # #         plt.plot(x_range_,err_folds[ind,:,2],marker='o',label="train_error"+str(ind))
# # #         plt.plot(x_range_,err_folds[ind,:,3],marker='o',label="test_error"+str(ind))
# #         plt.plot(x_range_,err_folds[ind,:,4],marker='o',label="rel_error"+str(ind))
# #           
# #     plt.legend()
# #     plt.show()
# #       
# #     exit(0)
#     
#     
#     if len (sys.argv) == 2:
#         start_time_ = time.time()
#         homeDir = "D:/Mtech/SY/PDS/a1/"
# #         homeDir = "/home2/e0268-9/"
#         hyper_param_train_flag = False#False
#         entire_data_train_flag = True#True
#         trainDataFile = sys.argv[1]
#         user_rating_pack = readTrainData(trainDataFile)
#         no_users = user_rating_pack[1]
#         no_movies = user_rating_pack[2]
#         user_rating_data = user_rating_pack[0]
#         user_movie_mat = np.zeros((no_users+1,no_movies+1),np.float)
#         for user_dat in user_rating_data:
#             user_movie_mat[int(user_dat[0]),int(user_dat[1])] = user_dat[2]
#         print(user_movie_mat.shape)
#         
#         omega_samples = [
#             (i, j, user_movie_mat[i, j])
#             for i in range(user_movie_mat.shape[0]) for j in range(user_movie_mat.shape[1]) if user_movie_mat[i, j] > 0 ]
#     
#     
#         np.random.shuffle(omega_samples)
#         
#         if hyper_param_train_flag == True:
#             folds_size = 5
#             noOfSamplesInFold = int(np.ceil(len(omega_samples) / float(folds_size)))
#             omega_sample_folds = partitionList(omega_samples,noOfSamplesInFold)
#     
#     
#             noOfFolds = folds_size
#             list_index_train = set(np.arange(0,folds_size))
#             kFoldCrossVal_params = [omega_sample_folds,list_index_train,noOfFolds]
#     
#             rank_k_lim = 20
#             #     [0.01,0.02,0.08,.1]
#             start_rank = 5
# #             beta_params = [0.02,0.06,0.1,0.3]
#             beta_params = [0.06,0.1,0.3]
#             errors_ = [SGD_Factorization(user_movie_mat,kFoldCrossVal_params, k_rank,beta_val, iters=220, cycles=1 ) for beta_val in beta_params for k_rank in range(start_rank,rank_k_lim)]
# #             errors_ = [SGD_Factorization(user_movie_mat,kFoldCrossVal_params, k_rank,0.1, iters=100, cycles=1 ) for k_rank in range(12,rank_k_lim)]
#             
#             train_err_rank_wise = [x[0] for x in errors_]
#             test_err_rank_wise = [x[1] for x in errors_]
#             rel_err_rank_wise = [x[4] for x in errors_]
#             
#             noOfSamplesInList = rank_k_lim - start_rank 
#             train_err = partitionList(train_err_rank_wise,noOfSamplesInList)
#             
#             test_err = partitionList(test_err_rank_wise,noOfSamplesInList)
#             
#             rel_err = partitionList(rel_err_rank_wise,noOfSamplesInList)
#             
#             train_err = np.array(train_err)
#             test_err = np.array(test_err)
#             rel_err = np.array(rel_err)
#             
#             rank_index = np.arange(start_rank,rank_k_lim)
#             
#             print("time of execution = ",time.time() - start_time_)
#             
#             beta_ind = 0
#             
#             np.savetxt(homeDir+"/train_err_rank_wise.dat",train_err.T)
#             np.savetxt(homeDir+"/test_err_rank_wise.dat",test_err.T)
#             np.savetxt(homeDir+"/rel_err_rank_wise.dat",rel_err.T)
#             
#             
# #             for beta_val in beta_params:
# #                 
# #                 plt.plot(rank_index,test_err[beta_ind],marker='o',label="test_"+str(beta_val))
# #                 plt.plot(rank_index,train_err[beta_ind],marker='o',label="train_"+str(beta_val))
# #                 plt.plot(rank_index,rel_err[beta_ind],marker='o',label="rel_"+str(beta_val))
# #                 beta_ind += 1
# #             
# #             plt.legend()
# #             plt.show()
#             
#         elif entire_data_train_flag == True:
#             folds_size = 5
#             noOfFolds = folds_size
#             list_index_train = set(np.arange(0,folds_size))
#             
#             np.random.shuffle(omega_samples)
#             testDataFile = homeDir+"/ratings.test"
#             test_data_prac = [line.strip('\n') for line in open(testDataFile, 'r')]
#      
#             predictFile = homeDir+"/ratings.pred"
#             pred_data_prac = [line.strip('\n') for line in open(predictFile, 'r')]
#  
#  
#             test_data_prac = test_data_prac[1:]
#             test_data_prac = [test_data.split('\t') for test_data in test_data_prac]
#             #     print(train_data)
#             test_data_prac = np.array(test_data_prac).astype(np.int)
#      
#             pred_data_prac = np.array(pred_data_prac).astype(np.float)
# #             test_data = test_data_prac
#             
#             
#             test_data = omega_samples[55000:]
#             test_data = np.array(test_data)
#             omega_samples = omega_samples[:54000]
#             
#             kFoldCrossVal_params_1 = [omega_samples,list_index_train,noOfFolds]
#             
#             
#             
#             errors_ = trainEntireData(user_movie_mat,kFoldCrossVal_params_1, 14,0.1, iters=400)
#      
# #             errors_[0].tofile(homeDir+"/P_mat.dat")
# #             errors_[1].tofile(homeDir+"/Q_mat.dat")
#             
# #             print(errors_[0].dtype)
#             
#             pred_value = [np.dot(errors_[0][int(test_data[i,0])],errors_[1][int(test_data[i,1])]) for i in range(0,len(test_data))]
#         
# #             print(pred_value)
#             pred_value = np.array(pred_value)
#             
#             rmse_err = (pred_value[:] - test_data[:,2])**2
# #             rmse_err = (pred_value[:] - pred_data_prac[:])**2
#             rmse_err = np.sum(rmse_err)
#             rmse_err /= float(len(pred_value))
#             rmse_err = np.sqrt(rmse_err)
#             print(rmse_err)
#             
#             
#             
#             
# #             R_approx_rank_wise = np.dot(errors_[0],np.transpose(errors_[1]))
# #             R_approx_rank_wise = np.array(R_approx_rank_wise)
#             
#             
#             
#             
#             exit(0)
#             
#     if len (sys.argv) == 3:
#         homeDir = "D:/Mtech/SY/PDS/a1/"
#         testDataFile = sys.argv[1]
#         outDataFile = sys.argv[2]
#         print(testDataFile)
#         m_ = 610
#         n_ = 9724
#         rank_sel = 12
#         
#         P_mat = np.fromfile(homeDir+"/P_mat.dat",dtype=np.float64)
#         Q_mat = np.fromfile(homeDir+"/Q_mat.dat",dtype=np.float64)
#         
#         P_mat = P_mat.reshape(m_,rank_sel)
#         Q_mat = Q_mat.reshape(n_,rank_sel)
#         
#         test_data_ = readTestData(testDataFile)
#         print(test_data_)
#         
#         pred_value = [np.dot(P_mat[test_data_[i,0]],Q_mat[test_data_[i,1]]) for i in range(0,len(test_data_))]
#         
#         print(pred_value)
#         
# #         np.set_printoptions(suppress=True)
#         
# #         pred_value = np.array(pred_value)
#         np.savetxt(outDataFile,pred_value,fmt='%1.15f')
#         
#         
#     
#     
#     exit(0)
#     
# #     trainDataFile = "D:/Mtech/SY/PDS/a1/ratings.train"
#     
# #     homeDir = "/home2/e0268-9/"
#     homeDir = "D:/Mtech/SY/PDS/a1/"
#     
#     
#     trainDataFile = homeDir+"/ratings.train"
#     
# #     trainDataFile = "/home2/e0268-9/ratings.train"
# 
# 
#     import pandas as pd
#     train_error = "D:/Mtech/SY/PDS/a1/train_err_rank_wise - Copy.dat"
#     test_error = "D:/Mtech/SY/PDS/a1/test_err_rank_wise - Copy.dat"
#      
#     r_wise_pred = "D:/Mtech/SY/PDS/a1/r_wise_pred.dat"
#      
#     train_err = pd.read_csv(train_error,header=None,delimiter='\n')
#     test_err = pd.read_csv(test_error,header=None,delimiter='\n')
#       
#     pred_ =  pd.read_csv(r_wise_pred,header=None,delimiter='\t')
#       
#     print(len(test_err),"...",test_err) 
#       
# #     plt.scatter(np.arange(2,len(train_err)+2),train_err,label="train")
# #     plt.plot(np.arange(2,30),test_err[:28],marker='o',label="test_1")
# #     plt.plot(np.arange(2,30),test_err[28:56],marker='o',label="test_2")
# #     plt.plot(np.arange(2,30),test_err[56:84],marker='o',label="test_3")
# #     plt.plot(np.arange(2,30),test_err[84:],marker='o',label="test_4")
# #      
# #      
# #     plt.plot(np.arange(2,30),train_err[:28],marker='o',label="train_1")
# #     plt.plot(np.arange(2,30),train_err[28:56],marker='o',label="train_2")
# #     plt.plot(np.arange(2,30),train_err[56:84],marker='o',label="train_3")
# #     plt.plot(np.arange(2,30),train_err[84:],marker='o',label="train_4")
# #      
# #     plt.legend()
# #     plt.show()
# # #     print(train_err)
# #     exit(0)
#  
#  
#  
#  
#     testDataFile = homeDir+"/ratings.test"
#     test_data_prac = [line.strip('\n') for line in open(testDataFile, 'r')]
#      
#     predictFile = homeDir+"/ratings.pred"
#     pred_data_prac = [line.strip('\n') for line in open(predictFile, 'r')]
#  
#  
#     test_data_prac = test_data_prac[1:]
#     test_data_prac = [test_data.split('\t') for test_data in test_data_prac]
# #     print(train_data)
#     test_data_prac = np.array(test_data_prac).astype(np.int)
#      
#     pred_data_prac = np.array(pred_data_prac).astype(np.float)
# #     
# # #     print(test_data_prac)
# #     
# # #     print(pred_data_prac)
# #     
# #     pred_ = np.array(pred_.values)
# #     
# # #     for i_ in range(0,pred_.shape[1]):
# #     i_ = 8
# #     plt.plot(pred_[:,i_],marker='x',label=str(i_+1))
# #     
# #     plt.plot(pred_data_prac,marker='o',label="preds_")
# #     plt.legend()
# #     plt.show()
# #     print(pred_)
# #     
# # 
# #     exit(0)
# 
#     user_rating_pack = readTrainData(trainDataFile)
#     no_users = user_rating_pack[1]
#     no_movies = user_rating_pack[2]
#     user_rating_data = user_rating_pack[0]
#     user_movie_mat = np.zeros((no_users+1,no_movies+1),np.float)
#     for user_dat in user_rating_data:
#         user_movie_mat[int(user_dat[0]),int(user_dat[1])] = user_dat[2]
#     
#     
#     
#     
#     print(user_movie_mat.shape)
# #     user_movie_mat = np.array([
# #     [5, 3, 0, 1],
# #     [4, 0, 0, 1],
# #     [1, 1, 0, 5],
# #     [1, 0, 0, 4],
# #     [0, 1, 5, 4],
# #     ])
#     
#     omega_samples = [
#             (i, j, user_movie_mat[i, j])
#             for i in range(user_movie_mat.shape[0]) for j in range(user_movie_mat.shape[1]) if user_movie_mat[i, j] > 0 ]
#     
#     
#     np.random.shuffle(omega_samples)
#     
#     folds_size = 5
#     noOfSamplesInFold = int(np.ceil(len(omega_samples) / float(folds_size)))
#     omega_sample_folds = partitionList(omega_samples,noOfSamplesInFold)
#     
# #     print(len(omega_sample_folds))
# #     print(len(omega_sample_folds[0]),"....",len(omega_sample_folds[4]))
#     
#     noOfFolds = folds_size
#     list_index_train = set(np.arange(0,folds_size))
#     kFoldCrossVal_params = [omega_sample_folds,list_index_train,noOfFolds]
#     
# #     kFoldCrossVal_params_1 = [omega_samples,list_index_train,noOfFolds]
# #     
# #     errors_ = trainEntireData(user_movie_mat,kFoldCrossVal_params_1, 12,0.2, iters=100)
# #     
# #     R_approx_rank_wise = np.dot(errors_[0],np.transpose(errors_[1]))
# #     R_approx_rank_wise = np.array(R_approx_rank_wise)
#     
#     
#     
#     
#     
#     rank_k_lim = 13
# #     [0.01,0.02,0.08,.1]
# #     errors_ = [SGD_Factorization(user_movie_mat,kFoldCrossVal_params, k_rank,beta_val, iters=100, cycles=1 ) for beta_val in [0.01,0.02,0.04,0.1] for k_rank in range(2,rank_k_lim)]
#     errors_ = [SGD_Factorization(user_movie_mat,kFoldCrossVal_params, k_rank,0.1, iters=100, cycles=1 ) for k_rank in range(12,rank_k_lim)]
#      
#     R_approx_rank_wise = [np.dot(x[2][0],np.transpose(x[3][0])) for x in errors_]
# #     
#     R_approx_rank_wise = np.array(R_approx_rank_wise)
# 
# 
# #     
# # #     print(test_data_prac[0,0])
# #     
# #     print(R_approx_rank_wise[0])
# #     print(user_movie_mat)
# #     exit(0)
# #     print(R_approx_rank_wise[0][test_data_prac[0,0]][test_data_prac[0,1]])
#     
#     predicted_by_model = [ R_approx_rank_wise[test_data_prac[i_,0]][test_data_prac[i_,1]] for i_ in range(len(test_data_prac))]
#      
#      
# #     predicted_by_model = [ [model_[test_data_prac[i_,0]][test_data_prac[i_,1]] for i_ in range(len(test_data_prac))] for model_ in R_approx_rank_wise 
# #                         ] 
#     predicted_by_model = np.array(predicted_by_model)
# #     
# #     np.savetxt(homeDir+"/r_wise_pred.dat",predicted_by_model.T, delimiter='\t')
# # #     print(R_approx_rank_wise[:][test_data_prac[:,0],test_data_prac[:,1]],"...",pred_data_prac[:])
# #     
#     print(predicted_by_model)
#     
#     pred_e = (predicted_by_model[0] - pred_data_prac)**2
#     pred_e = np.sqrt(np.sum(pred_e))
#     print(pred_e)
#     
#     exit(0)
# #     plt.plot(predicted_by_model[0],marker='x',label="pred")
# #     plt.plot(pred_data_prac,marker='o',label="known")
# #     plt.legend()
# #     plt.show()
#     
#     
#     train_err_rank_wise = [x[0] for x in errors_]
#     test_err_rank_wise = [x[1] for x in errors_]
#     
# #     np.savetxt(homeDir+"/train_err_rank_wise.dat",train_err_rank_wise)
# #     np.savetxt(homeDir+"/test_err_rank_wise.dat",test_err_rank_wise)
#     
#     
# #     plt.scatter(np.arange(10,rank_k_lim),train_err_rank_wise,label="train-loss")
# #     plt.scatter(np.arange(10,rank_k_lim),test_err_rank_wise,label="validation-loss")
# #     plt.legend()
# #     plt.show()
# #     np.savetxt("/home2/e0268-9/train_err_rank_wise.dat",train_err_rank_wise)
# #     np.savetxt("/home2/e0268-9/test_err_rank_wise.dat",test_err_rank_wise)
#     
#     
# #     for fold_ in range(noOfFolds):
# #         k_fold_cv_data = getCVForFoldIndex(fold_, omega_sample_folds, list_index_train)
# #         print("len ",len(k_fold_cv_data[0]),"....",len(k_fold_cv_data[1]))
#         
#     
#     
# #     performSGD(50,user_movie_mat)
# #     error_ = [performSGD(k,user_movie_mat) for k in range(1,21)]
# #     plt.scatter(np.arange(1,21),error_)
# #     plt.show()   
    
    
    
    
    