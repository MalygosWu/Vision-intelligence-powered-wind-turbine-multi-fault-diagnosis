# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 13:28:45 2024

@author: wcfda
"""
from DCGAN.utilize import collect_img_index, collect_img_data 

import os
import cv2
import copy
import numpy as np
from scipy import stats
from itertools import chain
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# ----------------------------------------------------------------------------- relevant functions
def get_SSE(image1, image2):
    error = image1.astype('int') - image2.astype('int')
    SSE = np.sum(error**2)
    
    return SSE

def get_freq_thres(images):
    avg_n_color_pixels = np.sum(images > 0)/len(images)
    main_color_freq = int(avg_n_color_pixels/30)
    freq_thres = max(main_color_freq, 10)
    
    return freq_thres

    
def find_main_color(image, freq_thres):
    color_freq = Counter(image.flatten())
    main_colors = [color for (color, freq) in color_freq.items() if freq > freq_thres]
    
    return main_colors

def gather_color(image, color_list):
    main_colors = np.sort(color_list)
    image_1d = image.flatten()
    gathered_img_1d = image_1d.copy()
    
    for i, color in enumerate(image_1d):
        # color and main_colors must be uint8 values
        dist = color - main_colors
        closest_color = main_colors[np.argmin(dist)]
        gathered_img_1d[i] = closest_color
    gathered_img = gathered_img_1d.reshape(image.shape)
    
    return gathered_img

def unify_color(images, base_images = None):
    unified_images = []
    main_color_list = []
    
    if base_images is None:
        freq_thres = get_freq_thres(images)
        for img in images:
            main_colors = find_main_color(img, freq_thres)
            main_color_list.append(main_colors)
            unified_img = gather_color(img, main_colors)
            unified_images.append(unified_img)
    else:
        freq_thres = get_freq_thres(base_images)
        for b_img in base_images:
            base_colors = find_main_color(b_img, freq_thres)
            main_color_list.append(base_colors)
        for img in images:
            SSE_list = [get_SSE(img, b_img) for b_img in base_images]
            match_idx = np.argmin(SSE_list)
            main_colors = main_color_list[match_idx]
            unified_img = gather_color(img, main_colors)
            unified_images.append(unified_img)  
            
    main_color_set = set(chain.from_iterable(main_color_list))
    print(f'Main color frequency > {freq_thres}; Main colors: {main_color_set}')

    return np.array(unified_images)

def remove_outlier(images, n_neighbors = 30, score_thres = 1.5):
    cleaned_images = copy.deepcopy(images)
    for img in cleaned_images:
        color_indices = np.where(img > 0)
        idx_data = np.asarray(color_indices).T
        
        n_idx = len(idx_data)
        if n_idx > 1:
            k = n_idx - 1 if n_neighbors + 1 > n_idx else n_neighbors
            neigh = NearestNeighbors(n_neighbors = k)
            neigh.fit(idx_data)
            neigh_dist = neigh.kneighbors()[0]
            mean_dist = np.mean(neigh_dist, axis = 1)
        
            z_scores = stats.zscore(mean_dist)
            outlier_loc = np.where(np.abs(z_scores) > score_thres)[0]
            outlier_indices = tuple(idx_arr[outlier_loc] for idx_arr in color_indices)
            img[outlier_indices] = 0

    return cleaned_images

def reinforce_outline(images):
    if len(images.shape) < 4:
        images = np.expand_dims(images, axis = 3)
    (_, h, w, _) = images.shape
    reinforced_images = []
    for img in images:
        r_img = np.zeros_like(img)
        color_indices = np.where(img > 0)
        idx_data = np.asarray(color_indices).T
        for idx in idx_data:
            i0, i1, i2 = idx[0], idx[1], idx[2]
            color = img[i0, i1, i2]
            r_img[min(i0 + 1, h - 1), i1, i2] = color
            r_img[max(i0 - 1, 0), i1, i2] = color
            r_img[i0, min(i1 + 1, w - 1), i2] = color
            r_img[i0, max(i1 - 1, 0), i2] = color
    
        r_img[color_indices] = img[color_indices]
        reinforced_images.append(r_img)
        
    return np.array(reinforced_images)

def refine_image(target_images, base_images = None, stages = [1, 2, 3]):
    images = copy.deepcopy(target_images)
    for stage in stages:
        if stage == 1: 
            images = unify_color(images, base_images)
        if stage == 2: 
            images = remove_outlier(images)
        if stage == 3: 
            images = reinforce_outline(images)
    
    return images
    
        
    
# ----------------------------------------------------------------------------- main process
if __name__ == '__main__':
    WT_idx = 230
    source = 'Ningxia'
    feature = 'Nace_Vib_X'
    stage_info = {
        'T_GBO_Visu': [1, 2, 3], 
        'PitchAngle2': [1, 2, 3], 
        'Nace_Vib_X': [1]
        }
    refine_stages = stage_info[feature]
    
    image_dir = 'C:/Users/wcfda/Desktop/Study/Projects/ViT_WTPM/codes/image generation/'\
        f'img_data/{source}_{WT_idx}/{feature}/' + '{} state/{} images/'
    real_F_image_dir = image_dir.format('faulty', 'real')
    real_H_image_dir = image_dir.format('healthy', 'real')
    generated_F_image_dir = image_dir.format('faulty', 'generated')
    train_H_image_dir = image_dir.format('healthy', 'train')
    train_F_image_dir = image_dir.format('faulty', 'train')

    if not os.path.exists(train_H_image_dir):
        os.makedirs(train_H_image_dir)
    if not os.path.exists(train_F_image_dir):
        os.makedirs(train_F_image_dir)
    
# ----------------------------------------------------------------------------- refine healthy images
    image_indices = collect_img_index(real_H_image_dir)
    real_H_images = collect_img_data(real_H_image_dir, p = 1, flag = 0)
    unified_H_images = unify_color(real_H_images)
    
    show = False
    for i, index in enumerate(image_indices):
        cv2.imwrite(train_H_image_dir + f'{index}.png', unified_H_images[i]) 
        if show:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(real_H_images[i])
            ax1.axis('off')
            ax2.imshow(unified_H_images[i])
            ax2.axis('off')
            plt.show()
            plt.close
    
# ----------------------------------------------------------------------------- refine faulty images
    real_F_images = collect_img_data(real_F_image_dir, p = 1, flag = 0)    
    generated_F_images = collect_img_data(generated_F_image_dir, p = 1, flag = 0)
    refined_F_images = refine_image(generated_F_images, base_images = real_F_images, stages = refine_stages)
    n_image = len(refined_F_images)
    
    show = False
    for i in range(n_image):
        cv2.imwrite(train_F_image_dir + f'{i}.png', refined_F_images[i]) 
        if show:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(generated_F_images[i])
            ax1.axis('off')
            ax2.imshow(refined_F_images[i])
            ax2.axis('off')
            plt.show()
            plt.close

    
    