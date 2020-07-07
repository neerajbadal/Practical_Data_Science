'''
Created on 15-Jun-2020

@author: Neeraj Badal
'''

import numpy as np
from sklearn.metrics import accuracy_score
# with open("../PDS_5_1/test_sample_out") as f:
#     orig = f.readlines()



with open("../PDS_5_1/out.dat") as f:
    preds = f.readlines()
    
# orig = [line.rstrip('\n') for line in orig]

orig_label = np.load("../PDS_5_1/train_label_2.npy")[:]
preds = [line.rstrip('\n') for line in preds]

mapDict = dict()
mapDict['NT'] = 4
mapDict['S'] = 0
mapDict['H'] = 1
mapDict['D'] = 2
mapDict['C'] = 3
# orig_label = [mapDict[char_] for char_ in orig]
pred_label = [mapDict[char_] for char_ in preds]
y_preds, y_actuals = np.vstack(pred_label), np.vstack(orig_label)
test_acc = accuracy_score(y_actuals, y_preds)
print(test_acc)
