'''
Created on 14-Apr-2020

@author: Neeraj Badal
'''
import numpy as np
from sklearn.metrics import mean_squared_error
from multiprocessing import Pool
from functools import partial

# table = np.random.rand(1000000, 1)
# qua = np.quantile(table,[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],axis=0)
# table[table[:,9].argsort()]
# print("completed",np.squeeze(qua))
from scipy.sparse import csr_matrix


class T_Leaf:
    noOfLevels = 0
    noOfLeaves = 0
    leafList = []
    decisionDict = dict()
    
    def compute_M(self,data):
        cols = np.arange(data.size)
        return csr_matrix((cols, (data.ravel(), cols)),
                      shape=(data.max() + 1, data.size))

    def get_indices_sparse(self,data):
        M = self.compute_M(data)
        return [np.unravel_index(row.data, data.shape) for row in M]
    
    
    def computeGradient(self,y_pred,y_actual):
        return (y_pred-y_actual)*2.0
     
    def computeHessian(self,y_pred):
        return np.ones((len(y_pred)))*2.0
     
     
    def giveBestSplit(self,split_val,colIndex):
    
#         print(lhs_indices)
        lhs_indices = np.nonzero(self.xvals[:,colIndex] <= split_val)[0]
  
        rhs_indices = np.nonzero(self.xvals[:,colIndex] > split_val)[0]
    
        
#         sortedXVals = self.xvals[np.argsort(self.xvals[:,colIndex])[0],colIndex]
#         
#         lhs_indices = np.searchsorted(sortedXVals,split_val,'right')
        
#         lhs_indices = np.nonzero(self.xvals[:,colIndex] <= split_val)[0]
#  
#         rhs_indices = np.nonzero(self.xvals[:,colIndex] > split_val)[0]
        
            
        if len(lhs_indices) < self.minimumVals or len(rhs_indices) < self.minimumVals:
            return [float('-inf'),[],[]]
        
        gain_left = np.sum(self.gradient_[lhs_indices])
        gain_right = np.sum(self.gradient_[rhs_indices])
         
        hessian_left = np.sum(self.hessian_[lhs_indices])
        hessian_right = np.sum(self.hessian_[rhs_indices])
         
        gain_ = (gain_left**2) / (hessian_left + self.lambda_)
        gain_ += (gain_right**2) / (hessian_right + self.lambda_)
        gain_ -= (gain_left + gain_right)**2 / (hessian_left+hessian_right+self.lambda_)
        gain_ -= self.gamma_
        gain_ = gain_ / 2.0
        
        return [gain_,lhs_indices,rhs_indices]
     
    def spliLeaf(self,colIndex):
#         tempData = self.xvals[self.xvals[:,colIndex].argsort()]
        quantiles_ = np.quantile(self.xvals[:,colIndex],[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],axis=0)
        quantiles_ = np.squeeze(quantiles_)
        max_gain = float('-inf')
        probable_split = [-999,[],[]]
#         for split_val in quantiles_:
#             
#             lhs_indices = np.nonzero(self.xvals[:,colIndex] <= split_val)[0]
#  
#             rhs_indices = np.nonzero(self.xvals[:,colIndex] > split_val)[0]
#             
# #             print(np.nonzero(self.xvals[:,colIndex] <= split_val)[0])
# #             print(np.where(self.xvals[:,colIndex] <= split_val)[0])
#             
# #             lhs_indices = np.where(self.xvals[:,colIndex] <= split_val)[0]
# # 
# #             rhs_indices = np.where(self.xvals[:,colIndex] > split_val)[0]
#             
#             if len(lhs_indices) < self.minimumVals or len(rhs_indices) < self.minimumVals:
#                 continue
#             
#             gain_left = np.sum(self.gradient_[lhs_indices])
#             gain_right = np.sum(self.gradient_[rhs_indices])
#             
#             hessian_left = np.sum(self.hessian_[lhs_indices])
#             hessian_right = np.sum(self.hessian_[rhs_indices])
#             
#             gain_ = (gain_left**2) / (hessian_left + self.lambda_)
#             gain_ += (gain_right**2) / (hessian_right + self.lambda_)
#             gain_ -= (gain_left + gain_right)**2 / (hessian_left+hessian_right+self.lambda_)
#             gain_ -= self.gamma_
#             gain_ = gain_ / 2.0
#             if gain_ > max_gain :
#                 max_gain = gain_
#                 probable_split[0] = split_val
#                 probable_split[1] = lhs_indices
#                 probable_split[2] = rhs_indices
        
        
        
        
#         lhs_indices_bucket = [sortedXVals[:lhs_cut_quant,-1].astype(np.int) for lhs_cut_quant in lhs_cut ]
#         rhs_indices_bucket = [sortedXVals[lhs_cut_quant:,-1].astype(np.int) for lhs_cut_quant in lhs_cut ]
#         
        gainResults = [self.giveBestSplit(split_val,colIndex) for split_val in quantiles_]
        
        gainResults = np.array(gainResults)
        max_gain_index = np.argmax(gainResults[:,0])
        max_gain = gainResults[max_gain_index,0]
        probable_split[0] = quantiles_[max_gain_index]
        probable_split[1] = gainResults[max_gain_index,1]
        probable_split[2] = gainResults[max_gain_index,2]
        
                
        return [max_gain,probable_split]
    def prepareForSplitting(self):
        
        noOfColumns = self.xvals.shape[1] - 1
        max_gain_col = float('-inf')
        probable_split_col = [-1,-999,[],[]]
        
        
        splitReults = [ self.spliLeaf(col_)  for col_ in range(0,noOfColumns)]
        
#         pool = Pool(3)
#         splitReults = pool.map(self.spliLeaf,range(0,noOfColumns))
#         pool.close()
#         pool.join()
        
        
        splitReults = np.array(splitReults)
        
        max_col = np.argmax(splitReults[:,0])
        max_gain_col = splitReults[max_col,0]
        probable_split_col[0] = max_col
        probable_split_col[1] = splitReults[max_col,1][0]
        probable_split_col[2] = splitReults[max_col,1][1]
        probable_split_col[3] = splitReults[max_col,1][2]
        
#         print("found best column")
        
#         for col_ in range(0,noOfColumns):
# #             print("in col index ",col_)
#             gain_,split_leaves = self.spliLeaf(col_)
#             if gain_ > max_gain_col:
#                 max_gain_col = gain_
#                 probable_split_col[0] = col_
#                 probable_split_col[1] = split_leaves[0]
#                 probable_split_col[2] = split_leaves[1]
#                 probable_split_col[3] = split_leaves[2]
    
        
        leftIndices = probable_split_col[2]
        rightIndices = probable_split_col[3]
        
        if len(leftIndices) <self.minimumVals or len(rightIndices) < self.minimumVals:
#             print("I m a leaf",T_Leaf.noOfLevels)
            T_Leaf.noOfLeaves += 1
            if self.isLeftChild:
                T_Leaf.decisionDict[self.parent+str(self.parent_level+1)+"_l"] = ['T',
                                                                                self.xvals[:,-1]  ]
            else:
                T_Leaf.decisionDict[self.parent+str(self.parent_level+1)+"_r"] = ['T',
                                                                                  self.xvals[:,-1]]
            
            self.isLeaf = True
#             T_Leaf.leafList.append([T_Leaf.noOfLevels,self.xvals,
#                                     self.y_preds,self.gainSplit,
#                                     self.gradient_,self.hessian_])
            
            return
        T_Leaf.noOfLevels += 1
        if self.parent_level + 2 > self.maxLevels or T_Leaf.noOfLeaves >= self.maxLeaves:
#             print("reached max levels",T_Leaf.noOfLevels)
            T_Leaf.noOfLeaves += 1
            if self.isLeftChild:
                T_Leaf.decisionDict[self.parent+str(self.parent_level+1)+"_l"] = ['T',self.xvals[:,-1]]
            else:
                T_Leaf.decisionDict[self.parent+str(self.parent_level+1)+"_r"] = ['T',self.xvals[:,-1]]
            
            
#             T_Leaf.leafList.append([T_Leaf.noOfLevels,self.xvals,
#                                     self.y_preds,self.gainSplit,
#                                     self.gradient_,self.hessian_])
            
#             initial_pred = np.ones((len(self.y_preds[leftIndices])))* np.mean(self.y_preds[leftIndices])
#             T_Leaf.leafList.append([T_Leaf.noOfLevels,self.xvals[leftIndices],
#                                     self.y_preds[leftIndices],max_gain_col,
#                                     self.computeGradient(initial_pred,self.y_preds[leftIndices]),
#                                     self.computeHessian(self.y_preds[leftIndices])])
#             initial_pred = np.ones((len(self.y_preds[rightIndices])))* np.mean(self.y_preds[rightIndices])
#             
#             T_Leaf.leafList.append([T_Leaf.noOfLevels,self.xvals[rightIndices],
#                                     self.y_preds[rightIndices],max_gain_col,
#                                     self.computeGradient(initial_pred,self.y_preds[rightIndices]),
#                                     self.computeHessian(self.y_preds[rightIndices])])
            self.isLeaf = True
            return
        if self.parent_level == -1:
            
            T_Leaf.decisionDict[str(self.parent_level+1)] = ['I',probable_split_col[0],
                                                           probable_split_col[1],
                                                           str(self.parent_level+1)+"_"+str(self.parent_level+2)+"_l",
                                                           str(self.parent_level+1)+"_"+str(self.parent_level+2)+"_r"]
        else:
            if self.isLeftChild:
                nextProbeStartIndex = self.parent+str(self.parent_level+1)+"_l_"
                T_Leaf.decisionDict[self.parent+str(self.parent_level+1)+"_l"] = ['I',probable_split_col[0],
                                                                      probable_split_col[1],
                                                           nextProbeStartIndex+str(self.parent_level+2)+"_l",
                                                           nextProbeStartIndex+str(self.parent_level+2)+"_r"]
            else:
                nextProbeStartIndex = self.parent+str(self.parent_level+1)+"_r_"
                T_Leaf.decisionDict[self.parent+str(self.parent_level+1)+"_r"] = ['I',probable_split_col[0],
                                                                      probable_split_col[1],
                                                           nextProbeStartIndex+str(self.parent_level+2)+"_l",
                                                           nextProbeStartIndex+str(self.parent_level+2)+"_r"]
        
        if self.parent_level == -1:
            leftChild = T_Leaf(self.xvals[leftIndices],self.y_preds[leftIndices],True,
                              str(self.parent_level+1)+"_",self.parent_level+1,max_gain_col,self.originalY[leftIndices])
            rightChild = T_Leaf(self.xvals[rightIndices],self.y_preds[rightIndices],False,
                               str(self.parent_level+1)+"_",self.parent_level+1,max_gain_col,self.originalY[rightIndices])
        
        else:
            if self.isLeftChild:
                currentName = self.parent+str(self.parent_level+1)+"_l_"
            else:
                currentName = self.parent+str(self.parent_level+1)+"_r_"
            
            leftChild = T_Leaf(self.xvals[leftIndices],self.y_preds[leftIndices],True,
                              currentName,self.parent_level+1,max_gain_col,self.originalY[leftIndices])
            rightChild = T_Leaf(self.xvals[rightIndices],self.y_preds[rightIndices],False,
                               currentName,self.parent_level+1,max_gain_col,self.originalY[rightIndices])
        
#         print("finished level",T_Leaf.noOfLevels-1)
    def __init__(self,xvals,y_preds,childFlag=False,parent_="",parent_level=-1,
                 gain_split = 0,original_y=None,lambda_=1,gamma_=0,maxLevels=6,
                 estimator_=0,maxLeaves=12):
        self.xvals = xvals
        self.y_preds = y_preds
        self.originalY = original_y
        initial_pred = np.ones((len(self.y_preds)))* np.mean(self.y_preds)
        if estimator_ == 0:
            self.gradient_ = self.computeGradient(initial_pred,self.originalY)
        else:
            self.gradient_ = self.computeGradient(self.y_preds,self.originalY)
        self.hessian_ = self.computeHessian(self.y_preds)
        
        self.lambda_ = lambda_
        self.gamma_ = gamma_
        self.minimumVals = 500
        self.maxLevels = maxLevels
#         quantizeLevels = [0,1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
#         self.quantiles_ = np.quantile(self.xvals[:,:self.xvals.shape[1]-1],quantizeLevels,axis=0)
# #         print(self.quantiles_.shape)
        
        self.isLeaf = False
        self.gainSplit = gain_split
        self.isLeftChild = childFlag
        self.parent = parent_
        self.parent_level = parent_level
        self.maxLeaves = maxLeaves
#         print("start preparing")
        self.prepareForSplitting()
    


def fit_xg(x_train,y_train,x_test,y_test,n_estimators=100):
    original_y = np.copy(y_train)
    
    initial_pred = np.ones((len(y_train)))* np.mean(y_train)
    
    base_val = initial_pred[0]
    lambda_ = 1.0
    b = np.zeros((x_train.shape[0],x_train.shape[1]+1))
    b[:,:-1] = x_train
    x_train = b
    x_train[:,-1] = np.arange(0,len(x_train))
    
    
    
#     print(len(T_Leaf.leafList))
#     print(T_Leaf.leafList)
    
    learning_rate = 0.4
    
    for esti_in in range(0,n_estimators):
        leaf_ = T_Leaf(x_train,y_train,original_y=original_y,estimator_=esti_in)
        lenSoFar = 0
        print("completed tree strucutring",len(y_train))
        for k,v in T_Leaf.decisionDict.items():
            if v[0] == 'T':
                leaf_data_index = v[1].astype(np.int)
                lenSoFar += 1
                G_j = np.sum(leaf_.gradient_[leaf_data_index])
                H_j = np.sum(leaf_.hessian_[leaf_data_index])
                w_j = -(G_j)/(H_j+lambda_)
                T_Leaf.decisionDict[k].extend([w_j])
                if esti_in == 0:
                    y_train[leaf_data_index] = initial_pred[leaf_data_index] + (learning_rate * w_j)
                else:
                    y_train[leaf_data_index] = y_train[leaf_data_index] + (learning_rate * w_j)
        
        
        
        print("len so far ",lenSoFar)
        nodes_test_set = [predict_(T_Leaf.decisionDict, x_test_ins)
                          for x_test_ins in x_test] 
        
        if esti_in == 0:
            y_test_pred = [base_val + learning_rate * T_Leaf.decisionDict[node_key][2] 
                        for node_key in nodes_test_set]
        else:
            y_test_pred = [y_test_pred[ind_] + learning_rate * T_Leaf.decisionDict[nodes_test_set[ind_]][2] 
                        for ind_ in range(0,len(nodes_test_set))]
    
        mse_ = mean_squared_error(y_test,y_test_pred)
        mse_train = mean_squared_error(original_y,y_train)
        T_Leaf.decisionDict.clear()
        T_Leaf.noOfLevels = 0
        T_Leaf.noOfLeaves = 0
        initial_pred = y_train
        print(esti_in,"------train error ",mse_train)
        print(esti_in,"------test error ",mse_)
    

def predict_(treeDict,key):
    isInternal = treeDict['0'][0] == 'I'
    nodeName = '0'
    while isInternal:
        if key[treeDict[nodeName][1]] <= treeDict[nodeName][2]:
            nodeName = treeDict[nodeName][3]
        else:
            nodeName = treeDict[nodeName][4]
        isInternal = treeDict[nodeName][0] == 'I'
#     print(nodeName)
#     print(treeDict[nodeName])
    return nodeName
    
if __name__ == '__main__':
    commonDir = "D:/Mtech/FY/SEM2/PDS/neeraj/data_pds_ml/"
    trainFile = commonDir+"/years.train"
    ''' preparing data file to matrix'''
    with open(trainFile) as f:
        lines = f.readlines()
        
    lines = [line.split(',') for line in lines]
    lines = [ [float(x) for x in line] for line in lines ]
    train_mat = np.array(lines)
    lambda_ = 1.0
    print(train_mat.shape)
    
    
#         x_data = train_mat[:400000,1:]
#         y_data = train_mat[:400000,0]
    
    x_data = train_mat[:20000,1:]
    b = np.zeros((x_data.shape[0],x_data.shape[1]+1))
    b[:,:-1] = x_data
    x_data = b
    x_data[:,-1] = np.arange(0,len(x_data)) 
    
    y_data = train_mat[:20000,0]
    
    original_y = np.copy(y_data)
    
    initial_pred = np.ones((len(y_data)))* np.mean(y_data)
    
    
    
    
    
    
    leaf_ = T_Leaf(x_data,y_data)
    print(len(T_Leaf.leafList))
#     print(T_Leaf.leafList)
    
    learning_rate = 0.4
    
    
    
    print(T_Leaf.decisionDict)
    
    print(len(leaf_.gradient_))
    
#     {k: v for k, v in points.items() if v[0] < 5 and v[1] < 5}
    lenSoFar = 0
    for k,v in T_Leaf.decisionDict.items():
        if v[0] == 'T':
            lenSoFar += 1
            leaf_data_index = v[1].astype(np.int)
            
            G_j = np.sum(leaf_.gradient_[leaf_data_index])
            H_j = np.sum(leaf_.hessian_[leaf_data_index])
            w_j = -(G_j)/(H_j+lambda_)
            y_data[leaf_data_index] = initial_pred[leaf_data_index] + (learning_rate * w_j)
    
    print("len so far ",lenSoFar)
    
    print(predict_(T_Leaf.decisionDict, x_data[598]))
    print(y_data)
    exit(0)
    
    leafList = np.array(T_Leaf.leafList)
    
    l_weights = []
    
    for l_index in range(0,len(leafList)):
        G_j = np.sum(leafList[l_index,4])
        h_j = np.sum(leafList[l_index,5])
        l_weights.append(-(G_j)/(h_j+lambda_))
        leaf_data_index = np.array(leafList[l_index,1])
        leaf_data_index = leaf_data_index[:,-1].astype(np.int)
        
        y_data[leaf_data_index] = initial_pred[leaf_data_index] + (learning_rate * l_weights[-1])
    
    
    initial_pred = y_data
    print(y_data)
    
#     print("prepare for splitting : ",leaf_.prepareForSplitting())
    