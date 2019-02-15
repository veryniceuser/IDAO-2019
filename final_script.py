import os
from itertools import repeat
import numpy as np
import pandas as pd
import gc
import csv
from tqdm import tqdm as tqdm
from multiprocessing import Pool

####################### CHANGE FILE NAME ##########################
file_name =  'test_private_v2_track_1'#'train_v2' # 'test_public_v2'
file_path = './data/{}.csv'.format(file_name)
# Number od processes to create in Pool
num_workers = 4
# Number of part to divide dataset file (jobs to do for Pool)
num_works = 8
# The value to use for stations with missing hits
# when computing FOI features
EMPTY_FILLER = 10**9

def encode_MatchedHit_columns(df):
    isMatchedHit2 = (df['MatchedHit_X[2]'] != -9999).astype(int)
    isMatchedHit3 = (df['MatchedHit_X[3]'] != -9999).astype(int)
    
    #is_through = isMatchedHit2 + isMatchedHit3
    #df['is_through'] = (is_through == 2).astype(int)
        
    delta_x = df['MatchedHit_X[0]'] - (df['MatchedHit_X[3]'] * isMatchedHit3 + 
                                       df['MatchedHit_X[2]'] * (1 - isMatchedHit3) * isMatchedHit2 +
                                       df['MatchedHit_X[1]'] * (1 - isMatchedHit3) * (1 - isMatchedHit2))
    delta_y = df['MatchedHit_Y[0]'] - (df['MatchedHit_Y[3]'] * isMatchedHit3 + 
                                       df['MatchedHit_Y[2]'] * (1 - isMatchedHit3) * isMatchedHit2 +
                                       df['MatchedHit_Y[1]'] * (1 - isMatchedHit3) * (1 - isMatchedHit2))
    delta_z = df['MatchedHit_Z[0]'] - (df['MatchedHit_Z[3]'] * isMatchedHit3 + 
                                       df['MatchedHit_Z[2]'] * (1 - isMatchedHit3) * isMatchedHit2 +
                                       df['MatchedHit_Z[1]'] * (1 - isMatchedHit3) * (1 - isMatchedHit2))
    df['square_deviation'] = delta_x * delta_x + delta_y * delta_y
    df['deviation'] = np.sqrt(df['square_deviation'])
    df['length_MHit'] = np.sqrt(df['square_deviation'] + delta_z * delta_z)

def add_FOI_vector_feature(df):
    delta_x = df['closest_x[0]'] - (df['closest_x[3]'] * (df['closest_x[3]'] != EMPTY_FILLER).astype(int) +
                                    df['closest_x[2]'] * (1 - (df['closest_x[3]'] != EMPTY_FILLER).astype(int)) * 
                                                         (df['closest_x[2]'] != EMPTY_FILLER).astype(int) + 
                                    df['closest_x[1]'] * (1 - (df['closest_x[3]'] != EMPTY_FILLER).astype(int)) * 
                                                         (1 - (df['closest_x[2]'] != EMPTY_FILLER).astype(int)))
    delta_y = df['closest_y[0]'] -(df['closest_y[3]'] * (df['closest_y[3]'] != EMPTY_FILLER).astype(int) +
                                    df['closest_y[2]'] * (1 - (df['closest_y[3]'] != EMPTY_FILLER).astype(int)) * 
                                                         (df['closest_y[2]'] != EMPTY_FILLER).astype(int) + 
                                    df['closest_y[1]'] * (1 - (df['closest_y[3]'] != EMPTY_FILLER).astype(int)) * 
                                                         (1 - (df['closest_y[2]'] != EMPTY_FILLER).astype(int)))
    delta_z = df['closest_z[0]'] -(df['closest_z[3]'] * (df['closest_z[3]'] != EMPTY_FILLER).astype(int) +
                                    df['closest_z[2]'] * (1 - (df['closest_z[3]'] != EMPTY_FILLER).astype(int)) * 
                                                         (df['closest_z[2]'] != EMPTY_FILLER).astype(int) + 
                                    df['closest_z[1]'] * (1 - (df['closest_z[3]'] != EMPTY_FILLER).astype(int)) * 
                                                         (1 - (df['closest_z[2]'] != EMPTY_FILLER).astype(int)))
    df['square_deviation_FOI'] = delta_x * delta_x + delta_y * delta_y
    df['deviation_FOI'] = np.sqrt(df['square_deviation_FOI'])
    df['length_FOI'] = np.sqrt(df['square_deviation_FOI'] + delta_z * delta_z)



def add_diff_of_SD_feature(df):
    df['diff_of_square_deviations'] = (df['square_deviation_FOI'] - df['square_deviation']).apply(abs)
    df['diff_of_deviations'] = (df['deviation_FOI'] - df['deviation']).apply(abs)
    df['diff_of_lengths'] = (df['length_MHit'] - df['length_FOI']).apply(abs)



def add_dist_between_FOI_and_hit_XY(df):
    for station_num in range(4):
        delta_x = df['closest_x[%d]' % station_num] - df['MatchedHit_X[%d]' % station_num]
        delta_y = df['closest_y[%d]' % station_num] - df['MatchedHit_Y[%d]' % station_num]
        df.loc[:, 'dist_FOI_hit_XY[%d]' % station_num] = delta_x * delta_x + delta_y * delta_y
        df['dist_FOI_hit_T[%d]' % station_num] = np.abs(df['closest_T[%d]' % station_num] - df['MatchedHit_T[%d]' % station_num])

def add_dist_between_FOI_and_hit_XYZ(df):
    for station_num in range(4):
        delta_x = df['closest_x[%d]' % station_num] - df['MatchedHit_X[%d]' % station_num]
        delta_y = df['closest_y[%d]' % station_num] - df['MatchedHit_Y[%d]' % station_num]
        delta_z = df['closest_z[%d]' % station_num] - df['MatchedHit_Z[%d]' % station_num]
        df['dist_FOI_hit_XYZ[%d]' % station_num] = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z


def add_angles(df):
    df['momentum_angle'] = df['PT'] / df['P']
    df['FOI_vect_angle'] = df['deviation_FOI'] / df['length_FOI']
    df['mhit_vect_angle'] = df['deviation'] / df['length_MHit']
    df['diff_FOI_angle'] = np.abs(df['FOI_vect_angle'] - df['momentum_angle'])
    df['diff_mhit_angle'] = np.abs(df['mhit_vect_angle'] - df['momentum_angle'])

    isMatchedHit2 = (df['MatchedHit_X[2]'] != -9999).astype(int)
    isMatchedHit3 = (df['MatchedHit_X[3]'] != -9999).astype(int)
    df['time_estim'] = np.abs(df['MatchedHit_T[0]'] - (df['MatchedHit_T[3]'] * isMatchedHit3 + 
                                       df['MatchedHit_T[2]'] * (1 - isMatchedHit3) * isMatchedHit2 +
                                       df['MatchedHit_T[1]'] * (1 - isMatchedHit3) * (1 - isMatchedHit2)))
    df['time_estim_closest'] = np.abs(df['closest_T[0]'] - (df['closest_T[3]'] * (df['closest_T[3]'] != EMPTY_FILLER).astype(int) +
                                    df['closest_T[2]'] * (1 - (df['closest_T[3]'] != EMPTY_FILLER).astype(int)) * 
                                                         (df['closest_T[2]'] != EMPTY_FILLER).astype(int) + 
                                    df['closest_T[1]'] * (1 - (df['closest_T[3]'] != EMPTY_FILLER).astype(int)) * 
                                                         (1 - (df['closest_T[2]'] != EMPTY_FILLER).astype(int))))



def str_to_list(s):
    return [float(i) for i in s[1:-1].split()]

N_STATIONS = 4
FEATURES_PER_STATION = 6
N_FOI_FEATURES = N_STATIONS*FEATURES_PER_STATION

###
### Problem in implementation
### if df is small, there is a chance, that
### [np.array(hit_x)[hit] if np.array(hit_x)[hit].size != 0 else np.array([EMPTY_FILLER])
###            for hit_x, hit in zip(df["FOI_hits_X"].apply(str_to_list), hits)]
### will contain only arrays of length 1, and when we perform  df["Lextra_Y[%i]" % station] - *value few lines higher*
### we get in a numpy array not an array of arrays (shape = (n,)) but an array of floats, 
### because all arrays contain only 1 element
### this will likely happen if n is on the order of 10**3, but will larger n, chances are very tiny
###
### by n i mean number of rows
###
def find_closest_hit_per_station_vectorized(df):
    result = np.empty((df.shape[0], N_FOI_FEATURES), dtype=np.float32)
    closest_x_per_station = result[:,0:4]
    closest_y_per_station = result[:,4:8]
    closest_T_per_station = result[:,8:12]
    closest_z_per_station = result[:,12:16]
    closest_dx_per_station = result[:,16:20]
    closest_dy_per_station = result[:,20:24]
   
    for station in range(4):
        print("Calculating for station {}".format(station))
        hits = df['FOI_hits_S'].apply(str_to_list).apply(lambda x : np.array(x) == station)

        x_distances_2 = (df["Lextra_X[%i]" % station] -
         np.reshape([np.array(hit_x)[hit] if np.array(hit_x)[hit].size != 0 else np.array([EMPTY_FILLER])
            for hit_x, hit in zip(df["FOI_hits_X"].apply(str_to_list), hits)], df.shape[0]))**2
 
        y_distances_2 = (df["Lextra_Y[%i]" % station] -
         np.reshape([np.array(hit_y)[hit] if np.array(hit_y)[hit].size != 0 else np.array([EMPTY_FILLER])
            for hit_y, hit in zip(df["FOI_hits_Y"].apply(str_to_list), hits)], df.shape[0]))**2
       
        x_hits = np.reshape([np.array(hit_x)[hit] if np.array(hit_x)[hit].size != 0 else np.array([EMPTY_FILLER])
                            for hit_x, hit in zip(df["FOI_hits_X"].apply(str_to_list), hits)], df.shape[0])
        y_hits = np.reshape([np.array(hit_y)[hit] if np.array(hit_y)[hit].size != 0 else np.array([EMPTY_FILLER])
                            for hit_y, hit in zip(df["FOI_hits_Y"].apply(str_to_list), hits)], df.shape[0])

        distances_2 = x_distances_2 + y_distances_2
        closest_hit = distances_2.apply(np.argmin)
        df["closest_dist_x[%i]" % station] = pd.Series([x_distances_2[i][closest_hit[i]] for i in tqdm(x_distances_2.index)])
        df["closest_dist_y[%i]" % station] = pd.Series([y_distances_2[i][closest_hit[i]] for i in tqdm(y_distances_2.index)])
        closest_x_per_station[:, station] = np.array([x_hits[i][closest_hit[i]] for i in tqdm(x_distances_2.index)])
        closest_y_per_station[:, station] = np.array([y_hits[i][closest_hit[i]] for i in tqdm(y_distances_2.index)])
        closest_T_per_station[:, station] = np.array(
            [np.array(hit_t)[hit][cl_hit] if np.array(hit_t)[hit].size != 0 else np.array([EMPTY_FILLER])
             for hit_t, hit, cl_hit in tqdm(zip(df["FOI_hits_T"].apply(str_to_list), hits, closest_hit))])    
        closest_z_per_station[:, station] = np.array(
            [np.array(hit_z)[hit][cl_hit] if np.array(hit_z)[hit].size != 0 else np.array([EMPTY_FILLER])
             for hit_z, hit, cl_hit in tqdm(zip(df["FOI_hits_Z"].apply(str_to_list), hits, closest_hit))])
               
        closest_dx_per_station[:, station] = np.array(
            [np.array(hit_dx)[hit][cl_hit] if np.array(hit_dx)[hit].size != 0 else np.array([EMPTY_FILLER])
             for hit_dx, hit, cl_hit in tqdm(zip(df["FOI_hits_DX"].apply(str_to_list), hits, closest_hit))])
               
        closest_dy_per_station[:, station] = np.array(
            [np.array(hit_dy)[hit][cl_hit] if np.array(hit_dy)[hit].size != 0 else np.array([EMPTY_FILLER])
             for hit_dy, hit,cl_hit in tqdm(zip(df["FOI_hits_DY"].apply(str_to_list), hits, closest_hit))])
        del hits
        del x_distances_2
        del y_distances_2
        del distances_2
        gc.collect() 
               
    return result


def process_foi(start_idxs_and_count):
    start_idx, count = start_idxs_and_count[0], start_idxs_and_count[1]
    train = pd.read_csv(file_path, skiprows=range(1, start_idx), nrows=count)

    closest_hits_features = find_closest_hit_per_station_vectorized(train)

    cl_hits_df = pd.DataFrame(closest_hits_features,columns=["closest_{0}[{1}]".format(typ, station) 
                                                for typ in ['x', 'y', 'T', 'z', 'dx', 'dy'] for station in [0,1,2,3]])
    train[cl_hits_df.columns] = cl_hits_df.loc[:, cl_hits_df.columns]
    encode_MatchedHit_columns(train)
    add_FOI_vector_feature(train)
    add_diff_of_SD_feature(train)
    add_dist_between_FOI_and_hit_XY(train)
    add_dist_between_FOI_and_hit_XYZ(train)
    #add_is_Muon(train)
    add_angles(train)

    # lexas_features = ['square_deviation', 'square_deviation_FOI', 'diff_of_square_deviations',
    #               'dist_FOI_hit_XY[0]',
    #               'dist_FOI_hit_XY[1]', 'dist_FOI_hit_XY[2]', 'dist_FOI_hit_XY[3]',
    #               'dist_FOI_hit_XYZ[0]', 'dist_FOI_hit_XYZ[1]', 'dist_FOI_hit_XYZ[2]',
    #               'dist_FOI_hit_XYZ[3]', 'is_same_moment']
                  #'is_through',

    SIMPLE_PLUS_FOI_FEAT_COLS = (['{}[{}]'.format(name, station) 
                                            for name in (
                                                ['ncl', 'avg_cs', 'dist_FOI_hit_XY','dist_FOI_hit_XYZ', 'dist_FOI_hit_T'] + #
                                                ['MatchedHit_%s' % typ for typ in 
                                                    ['TYPE', 'X', 'Y', 'Z', 'DX', 'DY', 'DZ', 'T', 'DT']
                                                ] + 
                                                ['Mextra_D%s2' % typ for typ in ['X', 'Y']] + 
                                                ['Lextra_%s' % typ for typ in ['X', 'Y']] + 
                                                ['closest_%s' % typ for typ in ['x', 'y', 'z', 'T', 'dx', 'dy']] +
                                                ['closest_dist_%s' % typ for typ in ['x', 'y']]
                                            ) for station in range(4)
                                    ] + 
                                    ['NShared',
                                    'square_deviation', 'square_deviation_FOI', 
                                    'diff_of_square_deviations',
                                    'FOI_hits_N', 'PT', 'P'] + 
                                    [ 'deviation', 'length_MHit', 
                                      'deviation_FOI', 'length_FOI',
                                      'diff_of_deviations' ,'diff_of_lengths', #'isMuon'
                                    ] + 
                                    ['momentum_angle', 'FOI_vect_angle', 'mhit_vect_angle', 
                                    'diff_FOI_angle', 'diff_mhit_angle', 
                                    'time_estim', 'time_estim_closest'])

# 'is_through', 'ndof',

    train.loc[:, SIMPLE_PLUS_FOI_FEAT_COLS].to_csv("add_feats_from_idx_{}".format(start_idx, start_idx +count), index=False)

    #train.to_csv("add_feats_from_idx_{}".format(start_idx, start_idx +count), index=False)
    print("Done")
    return 0

if __name__ == '__main__':
    lines = sum(1 for line in csv.reader(open(file_path)))
    with Pool(num_workers) as p:
        p.map(process_foi, [(start, count) 
                               for start, count in zip(
                                    [0] + [idx * (lines // num_works) + 1 for idx in range(1, num_works)],
                                    ([lines // num_works]*(num_works-1)) + [lines]
                                                       )
                               ])
    feats = pd.read_csv("add_feats_from_idx_0")
    for idx in range(1, num_works):
        feats = feats.append(pd.read_csv('add_feats_from_idx_{}'.format(idx * (lines // num_works) + 1)))
    feats.reset_index(inplace=True)
    feats.iloc[:, feats.columns != 'index'].to_csv("additional_feats_for_{}.csv".format(file_name), index=False)



