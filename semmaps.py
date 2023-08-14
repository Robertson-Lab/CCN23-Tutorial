import numpy as np
import pandas as pd

from scipy.io import loadmat
from scipy import stats
from matplotlib import pyplot as plt

import cv2

from vrgaze.utils import degrees_to_pixels, scale_durations
# import utils
import os,sys

def read_sem_map(img_mat_file):
    annots = loadmat(img_mat_file)
    map_array = np.array([[element for element in upperElement] for upperElement in annots['thisThreshMap']])
    return map_array

def zscore_sem_map(map_array):
    row_num,col_num = map_array.shape
    row_len = row_num*col_num

    reshaped_array = map_array.reshape(row_len,1)
    zscore_vals = stats.zscore(reshaped_array, ddof = 1)
    zscore_array = zscore_vals.reshape(row_num,col_num)
    
    return zscore_array

def plot_sem_map(map_array,image_path,image_width,image_height,map_type = 'Unspecified', fig_size = (12,6)):
    image_name = os.path.basename(image_path)
    img = cv2.imread(image_path)
    res = cv2.resize(img, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)[..., ::-1]
    
    fig = plt.figure(figsize=fig_size)
    plt.axis('off')
    plt.imshow(res)
    plt.imshow(map_array, alpha=0.6)
    plt.title(f'{image_name} {map_type}')
    plt.colorbar(label='Semantic Value')
    
def calc_sem_map_comparison(first_fix_y,first_fix_x,first_fix_dur, map_path, map_type):
    sem_map = read_sem_map(map_path)
    z_sem_map = zscore_sem_map(sem_map)
    
    df = pd.DataFrame()
    
    # get sal map values
    df['sem_vals'] = z_sem_map[first_fix_y,first_fix_x]
    df['weighted_sem_vals'] = np.multiply(df['sem_vals'],first_fix_dur)
    df['over_max'] = np.divide(df['sem_vals'],z_sem_map.max())
    df['weighted_over_max'] = np.multiply(df['over_max'],first_fix_dur)
    df['map_type'] = map_type
    
    return df

def run_sem_map_comparison(trial, sem_map_dir,image_width,image_height,num_fixations):
    # get trial fixations
    fix_data = trial.get_fixations()
    
    # change to equirect coordinates & scale durations
    x_pix, y_pix = degrees_to_pixels(fix_data['fix_yaw'], fix_data['fix_pitch'], image_width,image_height)
    x_pix = np.array(x_pix)
    y_pix = np.array(y_pix)
    # scale fixation durations from 0.1 to 1
    normed_durations = np.array(scale_durations(fix_data['duration']))

    # exclude poles
    valid_idx = np.where(np.logical_and(y_pix>+100, y_pix <+ 900)) 
    # get valid fixations from first 15 fixations that occured (not first 15 valid fixations)
    valid_idx = np.where(valid_idx[0]<=14)
    
    valid_y_pix = y_pix[valid_idx]
    valid_x_pix = x_pix[valid_idx]
    valid_dur = normed_durations[valid_idx]

    # get first n fixations or all fixations if number fix < desired number
    if len(valid_y_pix)>= num_fixations:
      first_fix_x = valid_x_pix[range(num_fixations)]
      first_fix_y= valid_y_pix[range(num_fixations)]
      first_fix_dur = valid_dur[range(num_fixations)]
    elif len(valid_y_pix) < num_fixations:
      first_fix_x = valid_x_pix[range(len(valid_idx))]
      first_fix_y= valid_y_pix[range(len(valid_idx))]
      first_fix_dur = valid_dur[range(len(valid_idx))]
    
    # get sem maps
    who_path = os.path.join(sem_map_dir, trial.trial_name + '_who.mat')
    what_path = os.path.join(sem_map_dir, trial.trial_name + '_what.mat')
    where_path = os.path.join(sem_map_dir, trial.trial_name + '_where.mat')
    
    df_who = calc_sem_map_comparison(first_fix_y,first_fix_x,first_fix_dur, who_path, 'who')
    df_what = calc_sem_map_comparison(first_fix_y,first_fix_x,first_fix_dur, what_path, 'what')
    df_where = calc_sem_map_comparison(first_fix_y,first_fix_x,first_fix_dur, where_path, 'where')
    
    df = pd.concat([df_who, df_what, df_where])
    df['trial_name'] = trial.trial_name
    df['subject'] = trial.subject

    df = df.reindex(columns=['trial_name', 'subject', 'map_type','sem_vals','weighted_sem_vals','over_max','weighted_over_max'])

    return df