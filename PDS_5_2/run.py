'''
Created on 14-Jun-2020

@author: Neeraj Badal
'''
import numpy as np
import sys
import pandas as pd

def preparePlayerHand(player_dat):
    
    comb_hand = []
    
    for game_ind in range(0,len(player_dat)):
        player_hand = np.zeros(52)
        count_ = 0
        
        
        for  char_ in player_dat[game_ind]:
            if char_ != '.':
                card_ind = count_+mapDict[char_]
                player_hand[card_ind] = 1
            elif char_ == '.':
                count_ += 13
        
        comb_hand.append(player_hand)
        
    comb_hand = np.array(comb_hand)    
    return comb_hand

def preparePlayerFeaturesII(player_hand_dat):
    high_card_sum = 0
    high_card_dim_vec = np.zeros((len(player_hand_dat),6))
    length_dim_vec = np.zeros((len(player_hand_dat),4))
    solid_seq_vec = np.zeros((len(player_hand_dat),4))
    
    singleton_vec = np.zeros((len(player_hand_dat),4))
    balanced_vec = np.zeros((len(player_hand_dat),1))
    singleton_hc_vec = np.zeros((len(player_hand_dat),4))
    
    aces_vec = np.zeros((len(player_hand_dat),4))
   
    high_card_sum += 4*(player_hand_dat[:,9] + player_hand_dat[:,22] +
                        player_hand_dat[:,35] + player_hand_dat[:,48]) 
    
    high_card_sum += 3*(player_hand_dat[:,10] + player_hand_dat[:,23] +
                        player_hand_dat[:,36] + player_hand_dat[:,49]) 

    high_card_sum += 2*(player_hand_dat[:,11] + player_hand_dat[:,24] +
                        player_hand_dat[:,37] + player_hand_dat[:,50]) 

    high_card_sum += 1*(player_hand_dat[:,12] + player_hand_dat[:,25] +
                        player_hand_dat[:,38] + player_hand_dat[:,51]) 

    
    
    print(high_card_sum.shape)
    print(" high card dim vec filling completed ")

    suit_vals_S = np.sum(player_hand_dat[:,:13],axis=1)
    suit_vals_H = np.sum(player_hand_dat[:,13:26],axis=1)
    suit_vals_D = np.sum(player_hand_dat[:,26:39],axis=1)
    suit_vals_C = np.sum(player_hand_dat[:,39:52],axis=1)

    print(" Suit card count completed")
    
    solid_seq_S_1 = np.sum(player_hand_dat[:,9:12],axis=1)//3 
    solid_seq_S_2 = np.sum(player_hand_dat[:,10:13],axis=1)//3
    solid_seq_S_3 = np.sum(player_hand_dat[:,11:13],axis=1)
    solid_seq_S_3 = (solid_seq_S_3 + player_hand_dat[:,8])//3
    
    solid_seq_S = solid_seq_S_1 + solid_seq_S_2 + solid_seq_S_3
    spot_seq_S = np.sum(player_hand_dat[:,:8],axis=1)
    print(solid_seq_S[0],spot_seq_S[0])
    
    
    solid_seq_H_1 = np.sum(player_hand_dat[:,22:25],axis=1)//3 
    solid_seq_H_2 = np.sum(player_hand_dat[:,23:26],axis=1)//3
    solid_seq_H_3 = np.sum(player_hand_dat[:,24:26],axis=1)
    solid_seq_H_3 = (solid_seq_H_3 + player_hand_dat[:,21])//3
    
    solid_seq_H = solid_seq_H_1 + solid_seq_H_2 + solid_seq_H_3
    spot_seq_H = np.sum(player_hand_dat[:,13:21],axis=1)
    
    solid_seq_D_1 = np.sum(player_hand_dat[:,35:38],axis=1)//3 
    solid_seq_D_2 = np.sum(player_hand_dat[:,36:39],axis=1)//3
    solid_seq_D_3 = np.sum(player_hand_dat[:,37:39],axis=1)
    solid_seq_D_3 = (solid_seq_D_3 + player_hand_dat[:,34])//3
    
    solid_seq_D = solid_seq_D_1 + solid_seq_D_2 + solid_seq_D_3
    spot_seq_D = np.sum(player_hand_dat[:,26:34],axis=1)
    
    solid_seq_C_1 = np.sum(player_hand_dat[:,35:38],axis=1)//3 
    solid_seq_C_2 = np.sum(player_hand_dat[:,36:39],axis=1)//3
    solid_seq_C_3 = np.sum(player_hand_dat[:,37:39],axis=1)
    solid_seq_C_3 = (solid_seq_C_3 + player_hand_dat[:,34])//3
    
    solid_seq_C = solid_seq_C_1 + solid_seq_C_2 + solid_seq_C_3
    spot_seq_C = np.sum(player_hand_dat[:,26:34],axis=1)
    
    print("solid seq completed ")
    
    card_count_S = np.sum(player_hand_dat[:,:13],axis=1)
    card_count_H = np.sum(player_hand_dat[:,13:26],axis=1)
    card_count_D = np.sum(player_hand_dat[:,26:39],axis=1)
    card_count_C = np.sum(player_hand_dat[:,39:52],axis=1)

    single_hc_S = np.sum(player_hand_dat[:,10:13],axis=1)
    single_hc_H = np.sum(player_hand_dat[:,23:26],axis=1)
    single_hc_D = np.sum(player_hand_dat[:,36:39],axis=1)
    single_hc_C = np.sum(player_hand_dat[:,49:52],axis=1)

    
    zero_spot_S = np.sum(player_hand_dat[:,0:10],axis=1)
    zero_spot_H = np.sum(player_hand_dat[:,13:23],axis=1)
    zero_spot_D = np.sum(player_hand_dat[:,26:36],axis=1)
    zero_spot_C = np.sum(player_hand_dat[:,39:49],axis=1)
    
    
    for game_ind in range(0,len(player_hand_dat)):    
        if high_card_sum[game_ind] <= 11:
            high_card_dim_vec[game_ind,0] = 1
        elif high_card_sum[game_ind] >= 12 and high_card_sum[game_ind] <= 14:
            high_card_dim_vec[game_ind,1] = 1
        elif high_card_sum[game_ind] >= 15 and high_card_sum[game_ind] <= 17:
            high_card_dim_vec[game_ind,2] = 1
        elif high_card_sum[game_ind] >= 18 and high_card_sum[game_ind] <= 19:
            high_card_dim_vec[game_ind,3] = 1
        elif high_card_sum[game_ind] >= 20 and high_card_sum[game_ind] <= 22:
            high_card_dim_vec[game_ind,4] = 1
        elif high_card_sum[game_ind] >= 23:
            high_card_dim_vec[game_ind,5] = 1
    
             
        length_dim_vec[game_ind,0] = max(suit_vals_S[game_ind] - 5,0)
        length_dim_vec[game_ind,1] = max(suit_vals_H[game_ind] - 5,0)
        length_dim_vec[game_ind,2] = max(suit_vals_D[game_ind] - 5,0)
        length_dim_vec[game_ind,3] = max(suit_vals_C[game_ind] - 5,0)
         
         
        if solid_seq_S[game_ind] >= 1 and (spot_seq_S[game_ind] >= 1):
            solid_seq_vec[game_ind,0] = 1
         
         
        if solid_seq_H[game_ind] >= 1 and (spot_seq_H[game_ind] >= 1):
            solid_seq_vec[game_ind,1] = 1
         
         
        if solid_seq_D[game_ind] >= 1 and ((spot_seq_D[game_ind] >= 1)):
            solid_seq_vec[game_ind,2] = 1
         
         
        if solid_seq_C[game_ind] >= 1 and ((spot_seq_C[game_ind] >= 1)):
            solid_seq_vec[game_ind,3] = 1
         
         
         
         
        if card_count_S[game_ind] == 1:
            singleton_vec[game_ind,0] = 1
        if card_count_H[game_ind] == 1:
            singleton_vec[game_ind,1] = 1
        if card_count_D[game_ind] == 1:
            singleton_vec[game_ind,2] = 1
        if card_count_C[game_ind] == 1:
            singleton_vec[game_ind,3] = 1                
                     
        doubleton_c = 0            
        if card_count_S[game_ind] == 2:
            doubleton_c += 1
        if card_count_H[game_ind] == 2:
            doubleton_c += 1
        if card_count_D[game_ind] == 2:
            doubleton_c += 1
        if card_count_C[game_ind] == 2:
            doubleton_c += 1
         
        if doubleton_c <= 1:
            balanced_vec[game_ind] = 1
         
         
         
        if single_hc_S[game_ind] == 1 and zero_spot_S[game_ind]==0:
            singleton_hc_vec[game_ind,0] = 1
        if single_hc_H[game_ind] == 1 and zero_spot_H[game_ind]==0:
            singleton_hc_vec[game_ind,1] = 1
        if single_hc_D[game_ind] == 1 and zero_spot_D[game_ind]==0:
            singleton_hc_vec[game_ind,2] = 1
        if single_hc_C[game_ind] == 1 and zero_spot_C[game_ind]==0:
            singleton_hc_vec[game_ind,3] = 1
         
        aces_vec[game_ind,0] = player_hand_dat[game_ind,9]
        aces_vec[game_ind,1] = player_hand_dat[game_ind,22]
        aces_vec[game_ind,2] = player_hand_dat[game_ind,35]
        aces_vec[game_ind,3] = player_hand_dat[game_ind,48]
         
     
         
    feature_vec = [high_card_dim_vec,length_dim_vec,
                        solid_seq_vec,singleton_vec,balanced_vec,
                        singleton_hc_vec,aces_vec]
     
         
    return feature_vec 

if __name__ == '__main__':
    if len (sys.argv) == 3:
        print("working on predictions")
        commonDir = "D:/Mtech/FY/SEM2/PDS/neeraj/data_pds_ml/"
    
        ipFileName = sys.argv[1]
        opFileName = sys.argv[2]
        
#         with open(ipFileName) as f:
#             lines = f.readlines()
        
        lines = pd.read_csv(ipFileName,header=None)
        lines = lines[0].to_list()
        
        noOfSamples = int(lines[0])
        lines = lines[1:]
        print("no. of test cases : ",noOfSamples)
        lines = lines[:noOfSamples]
        
        mapDict = dict()
        for i_ in range(0,8):
            mapDict[str(i_+2)] = i_
          
        mapDict['T'] = 8
        mapDict['A'] = 9
        mapDict['K'] = 10
        mapDict['Q'] = 11
        mapDict['J'] = 12
          
        mapDict['NT'] = 4
        mapDict['S'] = 0
        mapDict['H'] = 1
        mapDict['D'] = 2
        mapDict['C'] = 3
        
        
#         revMapDict = dict()
#         revMapDict[4] = 'NT'
#         revMapDict[0] = 'S'
#         revMapDict[1] = 'H'
#         revMapDict[2] = 'D'
#         revMapDict[3] = 'C'
        
        
        
#         print(lines[0])
# #         lines = lines.rstrip('\n')
#         lines = [line.rstrip('\n') for line in lines]
# #         lines = [line.split('\n') for line in lines]
#         lines = np.array(lines)
#         print(lines[0])
#         print(lines.shape)
#         if lines.ndim == 1:
#             print(lines[0],"  squezzing")
#             lines = np.squeeze(lines)
#             print(lines[0])
#         else:
#             lines = lines[:,0]
        
        
        
#         lines = lines[:,0]
        
        lines = [line.split(' ') for line in lines]
        lines = np.array(lines)
        
        players_ = lines
        
        player_hands = [preparePlayerHand(players_[:,pl_ind]) for pl_ind in range(players_.shape[1])]
        player_hands = np.array(player_hands)
        
        n_s_players_S = np.sum(player_hands[1,:,:13],axis=1) + np.sum(player_hands[3,:,:13],axis=1)  
      
        n_s_players_H = np.sum(player_hands[1,:,13:26],axis=1) + np.sum(player_hands[3,:,13:26],axis=1)
          
        n_s_players_D = np.sum(player_hands[1,:,26:39],axis=1) + np.sum(player_hands[3,:,26:39],axis=1)
          
        n_s_players_C = np.sum(player_hands[1,:,39:52],axis=1) + np.sum(player_hands[3,:,39:52],axis=1)
          
          
        n_s_players_S[n_s_players_S < 8] = 0
        n_s_players_H[n_s_players_H < 8] = 0
        n_s_players_D[n_s_players_D < 8] = 0
        n_s_players_C[n_s_players_C < 8] = 0
          
        n_s_players_S[n_s_players_S >= 8] = 1
        n_s_players_H[n_s_players_H >= 8] = 1
        n_s_players_D[n_s_players_D >= 8] = 1
        n_s_players_C[n_s_players_C >= 8] = 1
          
          
        e_w_players_S = np.sum(player_hands[0,:,:13],axis=1) + np.sum(player_hands[2,:,:13],axis=1)  
          
        e_w_players_H = np.sum(player_hands[0,:,13:26],axis=1) + np.sum(player_hands[2,:,13:26],axis=1)
          
        e_w_players_D = np.sum(player_hands[0,:,26:39],axis=1) + np.sum(player_hands[2,:,26:39],axis=1)
          
        e_w_players_C = np.sum(player_hands[0,:,39:52],axis=1) + np.sum(player_hands[2,:,39:52],axis=1)
          
          
        e_w_players_S[e_w_players_S < 8] = 0
        e_w_players_H[e_w_players_H < 8] = 0
        e_w_players_D[e_w_players_D < 8] = 0
        e_w_players_C[e_w_players_C < 8] = 0
          
          
        e_w_players_S[e_w_players_S >= 8] = 1
        e_w_players_H[e_w_players_H >= 8] = 1
        e_w_players_D[e_w_players_D >= 8] = 1
        e_w_players_C[e_w_players_C >= 8] = 1
          
          
          
          
        n_s_trump_fits = np.c_[n_s_players_S,n_s_players_H,
                               n_s_players_D,n_s_players_C]
          
        e_w_trump_fits = np.c_[e_w_players_S,e_w_players_H,
                               e_w_players_D,e_w_players_C]
            
        
        print("trump fits created ")
        
        pl_feature_vec = []
      
        for pl_index in range(0,len(player_hands)):
            pl_feature_vec.append(preparePlayerFeaturesII(player_hands[pl_index]))
          
          
          
        train_feat = np.c_[pl_feature_vec[0][0],
                           pl_feature_vec[1][0],
                           pl_feature_vec[2][0],
                           pl_feature_vec[3][0]]
           
           
        for dim_rem in range(1,len(pl_feature_vec[3])):
            train_feat = np.c_[train_feat,pl_feature_vec[0][dim_rem],
                           pl_feature_vec[1][dim_rem],
                           pl_feature_vec[2][dim_rem],
                           pl_feature_vec[3][dim_rem]]
           
          
          
        train_feat = np.c_[train_feat,n_s_trump_fits,e_w_trump_fits,player_hands[0],player_hands[1],
                            player_hands[2],player_hands[3]]
        
        
        print(train_feat.shape)
