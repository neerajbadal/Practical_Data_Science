'''
Created on 16-Mar-2020

@author: Neeraj Badal
'''
import sys
import numpy as np
import pickle
# from sklearn.metrics import mean_squared_error

def mse_local(y_orig,y_pred):
    mse_ = np.mean((y_orig - y_pred)**2)
#     mse_ = mse_ / len(y_orig)
    return mse_ 

class T_Leaf:
    noOfLevels = 0
    noOfLeaves = 0
    leafList = []
    decisionDict = dict()
    
    def computeGradient(self,y_pred,y_actual):
        return (y_pred-y_actual)*2.0
     
    def computeHessian(self,y_pred):
        return np.ones((len(y_pred)))*2.0
     
     
    def giveBestSplit(self,split_val,colIndex):
    
        lhs_indices = np.nonzero(self.xvals[:,colIndex] <= split_val)[0]
  
        rhs_indices = np.nonzero(self.xvals[:,colIndex] > split_val)[0]
        
            
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
        
        splitReults = np.array(splitReults)
        
        max_col = np.argmax(splitReults[:,0])
        max_gain_col = splitReults[max_col,0]
        probable_split_col[0] = max_col
        probable_split_col[1] = splitReults[max_col,1][0]
        probable_split_col[2] = splitReults[max_col,1][1]
        probable_split_col[3] = splitReults[max_col,1][2]
        
    
        
        leftIndices = probable_split_col[2]
        rightIndices = probable_split_col[3]
        
        if len(leftIndices) <self.minimumVals or len(rightIndices) < self.minimumVals:
            T_Leaf.noOfLeaves += 1
            if self.isLeftChild:
                T_Leaf.decisionDict[self.parent+str(self.parent_level+1)+"_l"] = ['T',
                                                                                self.xvals[:,-1]  ]
            else:
                T_Leaf.decisionDict[self.parent+str(self.parent_level+1)+"_r"] = ['T',
                                                                                  self.xvals[:,-1]]
            
            self.isLeaf = True
            
            return
        T_Leaf.noOfLevels += 1
        if self.parent_level + 2 > self.maxLevels or T_Leaf.noOfLeaves >= self.maxLeaves:
            T_Leaf.noOfLeaves += 1
            if self.isLeftChild:
                T_Leaf.decisionDict[self.parent+str(self.parent_level+1)+"_l"] = ['T',self.xvals[:,-1]]
            else:
                T_Leaf.decisionDict[self.parent+str(self.parent_level+1)+"_r"] = ['T',self.xvals[:,-1]]
            
            
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
        
        self.isLeaf = False
        self.gainSplit = gain_split
        self.isLeftChild = childFlag
        self.parent = parent_
        self.parent_level = parent_level
        self.maxLeaves = maxLeaves
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
        
        nodes_test_set = [predict_(T_Leaf.decisionDict, x_test_ins)
                          for x_test_ins in x_test] 
        
        if esti_in == 0:
            y_test_pred = [base_val + learning_rate * T_Leaf.decisionDict[node_key][2] 
                        for node_key in nodes_test_set]
        else:
            y_test_pred = [y_test_pred[ind_] + learning_rate * T_Leaf.decisionDict[nodes_test_set[ind_]][2] 
                        for ind_ in range(0,len(nodes_test_set))]
    
        mse_ = mse_local(y_test,y_test_pred)
        mse_train = mse_local(original_y,y_train)
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

if len (sys.argv) == 3:
        
        testDataFile = sys.argv[1]
        outDataFile = sys.argv[2]
        
        commonDir = "D:/Mtech/FY/SEM2/PDS/neeraj/data_pds_ml/"
#         teest_case_file = commonDir+"a3.testdat"
        
        modelFile = commonDir+'sklgbmIII.sav'
        
        
        
        
        ''' preparing test data file to matrix'''
        with open(testDataFile) as f:
            testDatalines = f.readlines()
        
        num_of_test = testDatalines[0]
        
        testDatalines = testDatalines[1:]    
        testDatalines = [line.split(',') for line in testDatalines]
        testDatalines = [ [np.float(x) for x in line] for line in testDatalines ]
        
        test_data_mat = np.array(testDatalines)
        print("test file data read ")
        print("number of tests = ",len(test_data_mat))
        
        loaded_model = pickle.load(open(modelFile, 'rb'))
        print("model loaded")
        
        pred_model = loaded_model.predict(test_data_mat)
        
        print(pred_model)
        
        
        predFile = commonDir+"a3.preddat"
        
        with open(predFile) as f:
            lines = f.readlines()
        
        lines = [ np.float(line) for line in lines ]
        
        pred_data_mat = np.array(lines)
        
        print(pred_data_mat)
        mse_train = mse_local(pred_data_mat,pred_model)
        print(mse_train)
        
        
        np.savetxt(outDataFile,pred_model,fmt='%1.4f')
        print("output file written")
        exit(0)
        