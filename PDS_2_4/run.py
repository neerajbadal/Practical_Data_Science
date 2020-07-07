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

def getLabelBucket(label_arr,no_of_labels):
    label_bucket = []
    for i_ in range(0,no_of_labels):
        label_bucket.append([])
    
    for l_ind in range(0,len(label_arr)):
        label_bucket[label_arr[l_ind]].append(l_ind)
    
    label_bucket = np.array(label_bucket)
    
    return label_bucket


def performKMeansPlus(data_,n_clusters_):
    centroid_1 = np.random.randint(0,len(data_))
    k_centroids = []
    data_index = np.arange(0,len(data_))
    k_centroids.append(data_[centroid_1])
    for k_i in range(1,n_clusters_):
        distance_list = doDistanceE(data_, k_centroids)
        distance_list = np.array(distance_list)
        distance_list = np.min(distance_list,axis=1)
        distance_list = distance_list**2
        sum_dist = np.sum(distance_list)
        probs_n = distance_list / sum_dist
        choice_n = np.random.choice(a=data_index,p=probs_n)
        k_centroids.append(data_[choice_n])
    
    k_centroids = np.array(k_centroids)
    return k_centroids
    


def performKmeans(data_,n_clusters_,tolerance=1e-3,max_iter=300):
    
    data_index = np.arange(0,len(data_))
    
    min_dist_across = []
    model_for_min_dist = []
    for random_i in range(0,20):
    
        k_centroids = performKMeansPlus(data_, n_clusters_)  
        
        for i_ in range(0,max_iter):
            distance_list = doDistanceE(data_, k_centroids)
            distance_list = np.array(distance_list)
            
            min_distance_centroid = np.argmin(distance_list,axis=1)
            
            cluster_wise_index = getLabelBucket(min_distance_centroid,n_clusters_)
            cluster_wise_data = []
            for j_ in range(0,n_clusters_):
                cluster_wise_data.append(data_[cluster_wise_index[j_]])
            
            cluster_wise_data = np.array(cluster_wise_data)
            old_centroids = k_centroids
            k_centroids = [np.mean(cluster_wise_data[k_i],axis=0) for k_i in range(0,len(cluster_wise_data))]
            
            
            hasConverged = True
            
            for j_ in range(0,n_clusters_):
                dist_error = np.linalg.norm(k_centroids[j_] - old_centroids[j_])
                if dist_error > tolerance:
                    hasConverged = False
            
            if hasConverged:
                print("iteration i : ",i_)
                distance_list = doDistanceE(data_, k_centroids)
                distance_list = np.array(distance_list)
                min_distance_centroid = np.argmin(distance_list,axis=1)
                min_distance_sum = 0#np.min(distance_list,axis=1)
                
                cluster_wise_index = getLabelBucket(min_distance_centroid,n_clusters_)
                cluster_wise_data = []
                for j_ in range(0,n_clusters_):
                    cluster_wise_data.append(data_[cluster_wise_index[j_]])
                    distance_list = doDistanceE(cluster_wise_data[-1], [k_centroids[j_]])
                    min_distance_sum += np.mean(distance_list)
                    
                print("min dis sum = ",min_distance_sum)
                min_dist_across.append(min_distance_sum)
                model_for_min_dist.append(k_centroids)
                break
         
    
    min_dist_across = np.array(min_dist_across)
    best_model_index = np.argmin(min_dist_across)
    best_model = model_for_min_dist[best_model_index]
    
    
    distance_list = doDistanceE(data_, best_model)
    distance_list = np.array(distance_list)
         
    min_distance_centroid = np.argmin(distance_list,axis=1)
    
    return [min_distance_centroid,best_model]

def performPCAAnalysis(data_,cut_dim):
    
    data_std = data_
    data_cov_ = np.cov(data_std.T)
    e_vals, e_vecs = np.linalg.eig(data_cov_)
    total_var = np.sum(e_vals)
    var_val_wise = [(i / total_var) for i in sorted(e_vals, reverse=True)]
    cum_var = np.cumsum(var_val_wise)

    eig_pair_vec = [(np.abs(e_vals[i_]), e_vecs[:, i_]) for i_ in range(len(e_vals))]

    eig_pair_vec.sort(key=lambda k: k[0], reverse=True)
    
    eig_pair_vec = np.array(eig_pair_vec)
    
    proj_matrix = []
    for i_ in range(0,len(eig_pair_vec)):
        if i_ < cut_dim:
            proj_matrix.append(eig_pair_vec[i_,1])
    
    proj_matrix = np.array(proj_matrix)
    proj_matrix = proj_matrix.T
    print(proj_matrix.shape)
    
    return proj_matrix
    
def performLabelMatching(cluster_label,given_label,ind_mat,no_of_clusters):

        bucket_giv = getLabelBucket(given_label, no_of_clusters)
        bucket_clust = getLabelBucket(cluster_label, no_of_clusters)
         
        
        for i_ in range(0,len(bucket_giv)):
            print("label = ",i_,"...",len(bucket_giv[i_]),"....",len(bucket_clust[i_]))
            
         
        card_mat = [[ len(set(cus).intersection(set(giv))) for giv in bucket_giv] for cus in bucket_clust]
        card_mat = np.array(card_mat)

        card_mat_max = np.max(card_mat,axis=1)
        
        max_list = []
        max_list_length = []
        
        for i_ in range(0,len(card_mat_max)):
            max_list.append(np.where(card_mat[i_] == card_mat_max[i_])[0])
            max_list_length.append(len(max_list[-1]))
        
        
        card_mat_sum = np.sum(card_mat_max)
        print("K-Means total matches = ",card_mat_sum)
        
        card_mat_strength = [[ len(set(cus).intersection(set(giv)))/(float(len(giv))) for giv in bucket_giv] for cus in bucket_clust]
        card_mat_strength = np.array(card_mat_strength)
         
        card_mat_max = np.argmax(card_mat,axis=1)
        print("K-Means labels = ",card_mat_max)
        
        label_assigned = []
        for i_ in range(0,len(card_mat_max)):
            if card_mat_max[i_] not in label_assigned and max_list_length[i_]==1:
                label_assigned.append(card_mat_max[i_])
            else:
                if max_list_length[i_]==1:
                    label_assigned.append(-1)
                else:
                    label_assigned.append(-2)
                
        
        
        availablle_label = np.arange(-2,20)
        
#         print(availablle_label)
#         print(label_assigned)
#         print(set(availablle_label)-set(label_assigned))
#         print(np.squeeze(np.argwhere(np.array(label_assigned) >= 0)))

        label_assigned[8] = 19
        label_assigned[3] = 0
        label_assigned[9] = 2
        label_assigned[10] = 18
        label_assigned[17] = 10
        label_assigned[12] = 4
        
        print(label_assigned)
        
        
        return label_assigned



import sys
if __name__ == '__main__':
    if len (sys.argv) == 1:
        
        commonDir = "D:/Mtech/FY/SEM2/PDS/neeraj/pds_corr_p2/"
        
        mnist_file = commonDir+"/data_8.csv"
        label_file = commonDir+"/labels_8.csv"
        
        ind_probe_file = mnist_file.split(sep='.')[0]+".probe_ind"
        centroid_model = mnist_file.split(sep='.')[0]+".model"
        random_ind_probe_file = mnist_file.split(sep='.')[0]+".random_probe_ind"
        
        label_prob_flag = True
#         label_prob_flag = False
#         label_gen_flag = True
        label_gen_flag = False
        
        ''' preparing data file to matrix'''
        with open(mnist_file) as f:
            lines = f.readlines()
            
        lines = [line.split(',') for line in lines]
        lines = [ [float(x) for x in line] for line in lines ]
        mnist_mat = np.array(lines)
     
        if label_prob_flag:
        
            ''' loading label file for training '''
            with open(label_file) as f:
                lines = f.readlines()
                 
    #         lines = [line.split(',') for line in lines]
            lines = [ int(line) for line in lines ]
            label_mat = np.array(lines)
        
        if label_prob_flag :
            ''' loading index probe file'''
            with open(ind_probe_file) as f:
                lines = f.readlines()
                       
            lines = [ int(line) for line in lines ]
            ind_mat = np.array(lines)
        
        
        ''' given number of clusters '''
        no_of_clusters = 20
        
        ''' reducing dimension using pca'''
        data_std = mnist_mat - np.mean(mnist_mat,axis=0)
        data_std = data_std / np.std(mnist_mat,axis=0)
    
        mnist_mat = data_std
        match_list = []
        
        probe_dim =[150]
        
        for k_ in probe_dim:
        
            print("------------------k= ",k_," --------")
            mnist_mat_proj = mnist_mat
            
            '''  training using k-menas clustering with k-means++ and random restart initialization  '''
            
            labels_params = performKmeans(mnist_mat,no_of_clusters)
            labels_cus = labels_params[0]
            if label_prob_flag:
                cluster_label = labels_cus[ind_mat]
                match_count = performLabelMatching(cluster_label,label_mat, ind_mat,no_of_clusters)
                match_list.append(match_count)    
        
        '''  assigning clustered according to label buckets from k-means  '''
        bucket_cus = getLabelBucket(labels_cus, no_of_clusters)
        

        ''' choose 5-points from each bucket closed to the centroid'''
         
        distance_clust = []
        clust_close_index = []
        for i in range(0,len(bucket_cus)):
            points_ = mnist_mat_proj[bucket_cus[i]]
            dist_ = doDistanceE(points_,[labels_params[1][i]])
             
            print("point index in this cluster ",len(bucket_cus[i]))
             
            dist_ = np.squeeze(dist_)
             
            dist_sorted = np.argsort(dist_)[:5]
             
            dist_sorted = np.array(dist_sorted).astype(int)
            print(len(dist_sorted))
            
             
            close_index = np.array(bucket_cus[i])[dist_sorted]
             
            clust_close_index.append(close_index)
         
        print("-------------------------")
        
        
        if label_gen_flag:
            with open(ind_probe_file, 'w') as fileId:
                for clust_i in range(0,len(clust_close_index)):
                    for close_i in range(0,len(clust_close_index[clust_i])):
                        print(clust_close_index[clust_i][close_i])
                        fileId.write(str(clust_close_index[clust_i][close_i])+'\n')
            
            
            
            with open(centroid_model, 'w') as fileId:
                for clust_i in range(0,len(labels_params[1])):
                    for j_ in range(0,len(labels_params[1][clust_i])):
                        fileId.write(str(labels_params[1][clust_i][j_]))
                        if j_ < len(labels_params[1][clust_i])-1:
                            fileId.write('\t')
                    if clust_i < len(labels_params[1]):
                        fileId.write('\n')
        
        exit(0)
    
    if len (sys.argv) == 3:
        
        testDataFile = sys.argv[1]
        outDataFile = sys.argv[2]
        
        commonDir = "D:/Mtech/FY/SEM2/PDS/neeraj/pds_corr_p2/"
        
        kMeansFile = commonDir+"/data_8.model"
        labelMapFile = commonDir+"/data_8.label_map"
        
        
        ''' preparing test data file to matrix'''
        with open(testDataFile) as f:
            lines = f.readlines()
        
        num_of_test = lines[0]
        
        lines = lines[1:]    
        lines = [line.split(' ') for line in lines]
        lines = [ [np.float(x) for x in line] for line in lines ]
        
        test_data_mat = np.array(lines)
        print("test file data read ")
        print("number of tests = ",len(test_data_mat))
        
        
        data_std = test_data_mat - np.mean(test_data_mat,axis=0)
        data_std = data_std / np.std(test_data_mat,axis=0)
    
        test_std_dat = data_std
        
        
        ''' reading model params'''
        with open(kMeansFile) as f:
            lines = f.readlines()
        
        lines = [line.split('\t') for line in lines]
        lines = [ [np.float(x) for x in line] for line in lines ]
     
        kmeans_centroid_mat = np.array(lines)
     
        print("reading model params ")
        
        print("dimensions of model params = ",kmeans_centroid_mat.shape)
        
        ''' reading label maps '''
        with open(labelMapFile) as f:
            lines = f.readlines()
        
        lines = [ int(line) for line in lines ]
        
        label_map = np.array(lines)
        label_map = dict(enumerate(label_map, 0))
        print("label maps loaded")
        
        
        print("processing test records")
        output_label_ = []
        for i_ in range(0,len(test_std_dat)):
            dist_ = doDistanceE(kmeans_centroid_mat,[test_std_dat[i_]])
            dist_ = np.squeeze(dist_)
            class_label = np.argmin(dist_)
            output_label_.append(label_map[class_label])
            
        output_label_ = np.array(output_label_)
        
        
        np.savetxt(outDataFile,output_label_,fmt='%i')
        print("output file written")
        exit(0)
                
        
        
        