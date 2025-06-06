# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 23:59:13 2024

@author: wcfda
"""
from Diff_Augment import DiffAugment

import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input, Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import set_random_seed
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.layers import (Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, 
                                     Flatten, Dropout, Dense, Reshape, ReLU)
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
WT_idx = 230
source = 'Ningxia'
feature = 'Nace_Vib_X'
image_load_dir = r'C:/Users/wcfda/Desktop/Study/Projects/ViT_WTPM/codes/'\
    f'image generation/img_data/{source}_{WT_idx}/{feature}/' + '{} state/{} images/'
save_dir = 'C:/Users/wcfda/Desktop/Study/Projects/ViT_WTPM/codes/'\
    f'image generation/training_results/{source}_{WT_idx}/{feature}/'
image_save_dir = save_dir + 'generated images/'
loss_save_dir = save_dir + 'loss/'
if not os.path.exists(image_save_dir):
    os.makedirs(image_save_dir)
if not os.path.exists(loss_save_dir):
    os.makedirs(loss_save_dir)
    
sample_size = 1000
latent_dim = 100
batch_size = 32
num_epoch = 30
gap_epoch = 5

# -----------------------------------------------------------------------------
def collect_img_index(image_dir, image_type = 'png'):
    suffix = f'.{image_type}'
    file_names = os.listdir(image_dir)
    image_names = [name for name in file_names if suffix in name]
    image_indices = [int(name.split(suffix)[0]) for name in image_names]

    return np.sort(image_indices)
        
        
def collect_img_data(image_dir, feature = None, size = None, p = None, flag = 1):

    if (feature is None) or (type(feature) == str): 
        feat_list = [feature]
    elif type(feature) == list: 
        feat_list = feature
    else: 
        raise TypeError('(feature) should be a string or list')
    
    size_list, percent_list = None, None
    if type(size) == int: 
        size_list = [size for name in feat_list]
    elif type(size) == list: 
        size_list = size
    if (type(p) == float) or (type(p) == int): 
        percent_list = [p for name in feat_list]
    elif type(p) == list: 
        percent_list = p
    if (size_list is None) and (percent_list is None):
        raise TypeError('(size) or (p) cannot be empty at the same time')
    
    image_list = []
    for i, feat_name in enumerate(feat_list):
        image_final_dir = image_dir if feat_name is None else image_dir.format(feat_name)
        image_indices = collect_img_index(image_final_dir)
        n_image = len(image_indices)
        sample_size1 = 0 if size_list is None else size_list[i]
        sample_size2 = 0 if percent_list is None else int(n_image*percent_list[i])
        sample_size = min(max(sample_size1, sample_size2), n_image)
        
        for i in range(sample_size):
            image_idx = image_indices[i]
            image_fp = image_final_dir + f'{image_idx}.png'
            image = cv2.imread(image_fp, flag)
            image_list.append(image)
    
    image_data = np.array(image_list)
    if len(image_data.shape) < 4: 
        image_data = np.expand_dims(image_data, axis = 3) 
    
    return image_data

def divide_batchs(image_data, batch_size):
    num_obs = image_data.shape[0]
    batch_data = []
    for i in range(0, num_obs, batch_size):  
        one_batch = image_data[i:i + batch_size]
        batch_data.append(tf.convert_to_tensor(one_batch, dtype = tf.float32))
    
    return batch_data

data_dir = image_load_dir.format('faulty', 'real')
image_data = collect_img_data(data_dir, p = 1)
image_data = (image_data - 127.5) / 127.5
np.random.seed(9)
sample_idx = np.random.choice(image_data.shape[0], sample_size)
aug_image_data = image_data[sample_idx]
dataset = divide_batchs(aug_image_data, batch_size)

for image_batch in dataset:
    images = (image_batch.numpy() + 1)/2
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow((images * 255).astype("int32")[0])
    ax2.imshow((DiffAugment(images, policy = 'translation,cutout') * 255)[0])
    plt.show()
    break

# -----------------------------------------------------------------------------
def build_discriminator():
    discriminator = Sequential(
        [   
         Input(shape = (128, 128, 3)),
         Conv2D(64, kernel_size = (5, 5), strides = (2, 2), padding = "same",
                kernel_initializer = RandomNormal(mean = 0.0, stddev = 0.02), 
                use_bias = False),
         LeakyReLU(0.2),
    
         Conv2D(128, kernel_size = (5, 5), strides = (2, 2), padding = "same",
                kernel_initializer = RandomNormal(mean = 0.0, stddev = 0.02), 
                use_bias = False),
         BatchNormalization(momentum = 0.1, epsilon = 0.8, center = 1.0, scale = 0.02),
         LeakyReLU(0.2),
    
         Conv2D(256, kernel_size = (5, 5), strides = (2, 2), padding = "same",
                kernel_initializer = RandomNormal(mean = 0.0, stddev = 0.02), 
                use_bias = False),
         BatchNormalization(momentum = 0.1, epsilon = 0.8, center = 1.0, scale = 0.02),
         LeakyReLU(0.2),
    
         Conv2D(512, kernel_size = (5, 5), strides = (2, 2), padding = "same",
                kernel_initializer = RandomNormal(mean = 0.0, stddev = 0.02), 
                use_bias = False),
         BatchNormalization(momentum = 0.1, epsilon = 0.8, center = 1.0, scale = 0.02),
         LeakyReLU(0.2),
    
         Conv2D(1024, kernel_size = (5, 5), strides = (2, 2), padding = "same",
                kernel_initializer = RandomNormal(mean = 0.0, stddev = 0.02), 
                use_bias = False),
         BatchNormalization(momentum = 0.1, epsilon = 0.8, center = 1.0, scale = 0.02),
         LeakyReLU(0.2),
    
         Flatten(),
         Dropout(0.2),
         Dense(1, activation = 'sigmoid')
        ],
        name = "discriminator",
    )
    
    return discriminator

# -----------------------------------------------------------------------------
def build_generator(latent_dim):
    generator = Sequential(
        [
         Input(shape = (latent_dim,)),
         Dense(8 * 8 * 1024),
         Reshape((8, 8, 1024)),
         Conv2DTranspose(512, kernel_size = (5, 5), strides = (2, 2), padding = 'same',
                         kernel_initializer = RandomNormal(mean = 0.0, stddev = 0.02),
                         use_bias = False),
         BatchNormalization(momentum = 0.1, epsilon = 0.8, center = 1.0, scale = 0.02),
         ReLU(),
        
         Conv2DTranspose(256, kernel_size = (5, 5), strides = (2, 2), padding = 'same',
                         kernel_initializer = RandomNormal(mean = 0.0, stddev = 0.02),
                         use_bias = False),
         BatchNormalization(momentum = 0.1, epsilon = 0.8, center = 1.0, scale = 0.02),
         ReLU(),
    
         Conv2DTranspose(128, kernel_size = (5, 5), strides = (2, 2), padding = 'same',
                         kernel_initializer = RandomNormal(mean = 0.0, stddev = 0.02),
                         use_bias = False),
         BatchNormalization(momentum = 0.1, epsilon = 0.8, center = 1.0, scale = 0.02),
         ReLU(),
    
         Conv2DTranspose(64, kernel_size = (5, 5), strides = (2, 2), padding = 'same',
                         kernel_initializer = RandomNormal(mean = 0.0, stddev = 0.02),
                         use_bias = False),
         BatchNormalization(momentum = 0.1, epsilon = 0.8, center = 1.0, scale = 0.02),
         ReLU(),
    
         Conv2D(3,  kernel_size = (5, 5), strides = (1, 1), padding = 'same',
                kernel_initializer = RandomNormal(mean = 0.0, stddev = 0.02),
                use_bias = False, activation = 'tanh')
        ],
        name = "generator",
    )
    
    return generator

# -----------------------------------------------------------------------------
saved_names = os.listdir(save_dir)
epoch_list = []
for name in saved_names:
    if 'models_epoch' in name:
        trained_epoch = int(name.split('_')[-1])
        epoch_list.append(trained_epoch)

if len(epoch_list) == 0:
    trained_epoch = 0
    discriminator = build_discriminator()
    generator = build_generator(latent_dim)
else:
    trained_epoch = np.max(epoch_list)
    model_load_dir = save_dir + f'models_epoch_{trained_epoch}/'
    discriminator = load_model(model_load_dir + 'discriminator.keras')
    generator = load_model(model_load_dir + 'generator.keras')

discriminator.summary()
generator.summary()

random_noise = tf.random.normal([1, latent_dim])
generated_image = generator(random_noise, training = False)
plt.imshow((generated_image[0] + 1) / 2)
plt.axis('off')
plt.show()

decision = discriminator(generated_image)
print('decision', decision)

# -----------------------------------------------------------------------------
binary_cross_entropy = BinaryCrossentropy()
generator_optimizer = Adam(0.0002, beta_1 = 0.5)
discriminator_optimizer = Adam(0.0002, beta_1 = 0.5)

def generator_loss(label, fake_output):
    gen_loss = binary_cross_entropy(label, fake_output)
    return gen_loss

def discriminator_loss(label, output):
    disc_loss = binary_cross_entropy(label, output)
    return disc_loss

# -----------------------------------------------------------------------------
@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size, latent_dim])
    # plt.subplot(1, 2, 1)
    # plt.imshow((images[0].numpy()*255).astype("int32"))
    images = DiffAugment(images, policy = 'translation,cutout')
    # plt.subplot(1, 2, 2)
    # plt.imshow((images[0].numpy()*255).astype("int32"))
    # plt.show()

    with tf.GradientTape() as disc_tape1:
        generated_images = generator(noise, training = True)
        generated_images = DiffAugment(generated_images,policy = 'translation,cutout')

        real_output = discriminator(images, training = True)
        real_targets = tf.ones_like(real_output)
        disc_loss1 = discriminator_loss(real_targets, real_output)

    gradients_disc1 = disc_tape1.gradient(disc_loss1, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_disc1, discriminator.trainable_variables))

    with tf.GradientTape() as disc_tape2:
        fake_output = discriminator(generated_images, training = True)
        fake_targets = tf.zeros_like(fake_output)
        disc_loss2 = discriminator_loss(fake_targets, fake_output)

    gradients_disc2 = disc_tape2.gradient(disc_loss2, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_disc2, discriminator.trainable_variables))

    with tf.GradientTape() as gen_tape:
        generated_images = generator(noise, training = True)
        generated_images = DiffAugment(generated_images, policy = 'translation,cutout')
        fake_output = discriminator(generated_images, training = True)
        real_targets = tf.ones_like(fake_output)
        gen_loss = generator_loss(real_targets, fake_output)

    gradients_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_gen, generator.trainable_variables))

    return disc_loss1 + disc_loss2, gen_loss

def generate_and_save_images(model, epoch, seed, dim  = (5, 5), figsize = (5, 5)):
    generated_images = model(seed)
    generated_images *=  255
    generated_images.numpy()
    plt.figure(figsize = figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        img = array_to_img(generated_images[i])
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig(image_save_dir + f'Epoch_{epoch}.png')
    plt.close()
    
def load_loss(loss_fp):
    if os.path.exists(loss_fp):
        with open(loss_fp, 'rb') as path:
            loss = pickle.load(path)
    else:
        loss = []

    return loss

def save_loss(loss, loss_fp):
    folder_dir = '/'.join(loss_fp.split('/')[:-1])
    if not os.path.exists(folder_dir):
        raise FileNotFoundError('Invalid folder directory')
    with open(loss_fp, 'wb') as path:
        pickle.dump(loss, path)
        
# -----------------------------------------------------------------------------
set_random_seed(9)
seed = tf.random.normal([25, latent_dim])
disc_losses = load_loss(loss_save_dir + 'discriminator')
gen_losses = load_loss(loss_save_dir + 'generator')

def save_results(epoch):
    model_save_dir = save_dir + f'models_epoch_{epoch}/'
    if not os.path.exists(model_save_dir): 
        os.makedirs(model_save_dir)
    discriminator.save(model_save_dir + 'discriminator.keras')
    generator.save(model_save_dir + 'generator.keras')
    save_loss(disc_losses, loss_save_dir + 'discriminator')
    save_loss(gen_losses, loss_save_dir + 'generator')
    
def train(dataset, num_epoch):
    if trained_epoch == 0:
        generate_and_save_images(generator, 0, seed)
    for i in range(1, num_epoch + 1):
        epoch = trained_epoch + i
        print(f'Epoch-{epoch} ({num_epoch - i} left)')
        
        disc_loss = gen_loss = 0
        for image_batch in tqdm(dataset):
            image_batch = dataset[0]
            losses = train_step(image_batch)
            disc_loss +=  losses[0]
            gen_loss +=  losses[1]
        
        disc_losses.append(disc_loss.numpy())
        gen_losses.append(gen_loss.numpy())
        generate_and_save_images(generator, epoch, seed)
        if epoch % gap_epoch == 0:
            save_results(epoch)
    
    save_results(trained_epoch + num_epoch)

    

# -----------------------------------------------------------------------------
train(dataset, num_epoch)

plt.figure()
plt.plot(disc_losses, label = 'Discriminator Loss')
plt.plot(gen_losses, label = 'Generator Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(save_dir + 'losses.png')
plt.close()

# -----------------------------------------------------------------------------
generator = load_model(save_dir + 'Generator_epoch_40')

for i in range(50):
    plt.subplot(5, 10, i + 1)
    random_noise = tf.random.normal([1, 100])
    generated_image = generator(random_noise, training = False)
    plt.imshow((generated_image[0] + 1) / 2)
    # plt.xticks([])
    # plt.yticks([])
    plt.axis('off')
plt.tight_layout()
plt.show()