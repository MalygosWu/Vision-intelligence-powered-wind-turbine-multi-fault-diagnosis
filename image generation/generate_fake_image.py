# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 18:36:23 2023

@author: wcfda
references:
https://youtu.be/xBX2VlDgd4I   #Introduction video
https://youtu.be/Mng57Tj18pc   #Keras implementation video. 
"""

import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model

WT_idx = 230
source = 'Ningxia'
feature = 'Nace_Vib_X'
epoch = 30
n_image = 1500
model_load_dir = 'C:/Users/wcfda/Desktop/Study/Projects/ViT_WTPM/codes/image generation/'\
    f'training_results/{source}_{WT_idx}/{feature}/models_epoch_{epoch}/generator.keras'
image_save_dir = 'C:/Users/wcfda/Desktop/Study/Projects/ViT_WTPM/codes/image generation/'\
    f'img_data/{source}_{WT_idx}/{feature}/faulty state/generated images/'
if not os.path.exists(image_save_dir):
    os.makedirs(image_save_dir)

# generate fake images
generator = load_model(model_load_dir)
latent_dim = generator.layers[0].input.shape[1]
noises = tf.random.normal([n_image, latent_dim])
generated_images = generator.predict(noises)

show = False
for i in range(n_image):
    scaled_image = (generated_images[i] + 1) / 2
    std_image = 255 * scaled_image
    cv2.imwrite(image_save_dir + f'{i}.png', std_image) 
    if show:
        plt.imshow(std_image)
        plt.axis('off')
        plt.show()
        plt.close
            
            
            

            
        

