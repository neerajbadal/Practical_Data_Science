'''
Created on 24-Feb-2020

@author: Neeraj Badal
'''
import numpy as np
from scipy.spatial.distance import cdist

def doDistanceE(pointList_1,pointList_2):
#     distance_cal = [[np.linalg.norm(point_1-point_2) for point_2 in pointList_2] for point_1 in pointList_1]
    distance_cal = cdist(pointList_1,pointList_2, 'euclidean')
    
    return distance_cal
import sys
if __name__ == '__main__':
    if len (sys.argv) == 2:
        pass
    if len (sys.argv) == 3:
        testDataFile = sys.argv[1]
        outDataFile = sys.argv[2]
        kMeansFile = "D:/Mtech/FY/SEM2/PDS/neeraj/pds_corr_p2/data_8.model"
        labelMapFile = "D:/Mtech/FY/SEM2/PDS/neeraj/pds_corr_p2/data_8.label_map"
        
        
        ''' preparing test data file to matrix'''
        with open(testDataFile) as f:
            lines = f.readlines()
        
        print(len(lines))
        num_of_test = lines[0]
        
        lines = lines[1:]    
        lines = [line.split(' ') for line in lines]
        lines = [ [np.float(x) for x in line] for line in lines ]
        
        
     
     
     
        test_data_mat = np.array(lines)
        
        data_std = test_data_mat - np.mean(test_data_mat,axis=0)
        data_std = data_std / np.std(test_data_mat,axis=0)
    
        test_std_dat = data_std
        
        
        print(test_data_mat.shape)
        
        ''' reading model params'''
        with open(kMeansFile) as f:
            lines = f.readlines()
        
        lines = [line.split('\t') for line in lines]
        lines = [ [np.float(x) for x in line] for line in lines ]
        
        
     
        kmeans_centroid_mat = np.array(lines)
        
        print(kmeans_centroid_mat.shape)
        
        ''' reading label maps '''
        with open(labelMapFile) as f:
            lines = f.readlines()
        
        lines = [ int(line) for line in lines ]
        
        label_map = np.array(lines)
        
        print(label_map)
        
        label_map = dict(enumerate(label_map, 0))
        print(label_map)
        
#         exit(0) 
         
        distance_clust = []
        clust_close_index = []
        
        output_label_ = []
        for i_ in range(0,len(test_std_dat)):
            dist_ = doDistanceE(kmeans_centroid_mat,[test_std_dat[i_]])
#             print(kmeans_centroid_mat)
#             print(len(test_std_dat[i_]))
            
            dist_ = np.squeeze(dist_)
#             print(dist_)
            class_label = np.argmin(dist_)
            print(class_label,label_map[class_label])
            output_label_.append(label_map[class_label])
            
#             exit(0)
        output_label_ = np.array(output_label_)
        
#         label_file = "D:/Mtech/FY/SEM2/PDS/neeraj/pds_corr_p2/labels_8.csv"
#         with open(label_file) as f:
#                 lines = f.readlines()
#                  
#         lines = [ int(line) for line in lines ]
#         label_mat = np.array(lines)
#         
#         sum_ = 0
#         for i_ in range(0,len(label_mat)):
#             if label_mat[i_] == output_label_[i_]:
#                 var_ = 1
#             else:
#                 var_ = 0
#             sum_ = sum_ + var_
#         
#         sum_ = sum_ / float(len(label_mat))
#         print("accuracy = ",sum_)
        
        np.savetxt(outDataFile,output_label_,fmt='%i')
        
#         with open(outDataFile, 'w') as fileId:
#                 for i_ in range(0,len(output_label_)):
#                     fileId.write(str(output_label_[i_]))
#                     if i_ < len(output_label_):
#                         fileId.write('\n')
        
        
        
        