# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 14:06:59 2023

@author: wcfda
"""
import os
os.path.insert('.../image generation', 0)
from DCGAN.utilize import collect_img_index, collect_img_data
from adjust_image import get_SSE, find_main_color, gather_color, unify_color

import keras
from keras import ops
from keras import layers
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

#------------------------------------------------------------------------------
#(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
WT_idx = 230
source = 'Ningxia'
feature = 'Nace_Vib_X'
img_dir = r'C:/Users/wcfda/Desktop/Study/Projects/ViT_WTPM/codes/'\
    f'image generation/img_data/{source}_{WT_idx}/{feature}/' + '{} state/{} images/'
idx_dir = r'C:/Users/wcfda/Desktop/Study/Projects/ViT_WTPM/codes/'\
    f'benchmarks/results/{source}_{WT_idx}/{feature}/'
save_dir = 'C:/Users/wcfda/Desktop/Study/Projects/ViT_WTPM/codes/'\
    f'fault classification/training results/{source}_{WT_idx}/{feature}/'

if not os.path.exists(idx_dir):
    os.makedirs(idx_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
test_idx_fp = save_dir + 'test_indices'
history_fp = save_dir + 'history'
model_fp = save_dir + 'last_model.weights.h5'
checkpoint_fp = save_dir + 'best_model.weights.h5'

#------------------------------------------------------------------------------
image_load_dir = img_dir.format('healthy', 'train')
real_healthy_imgs = collect_img_data(image_load_dir, p = 1, flag = 0)
H_sample_size = len(real_healthy_imgs)
image_load_dir = img_dir.format('faulty', 'train')
syn_faulty_imgs = collect_img_data(image_load_dir, p = 1, flag = 0)
F_sample_size = len(syn_faulty_imgs)

if os.path.exists(test_idx_fp):
    with open(test_idx_fp, 'rb') as fp:
        test_indices = pickle.load(fp)
else:
    test_percent = 0.33
    H_test_size = int(H_sample_size*test_percent)
    np.random.seed(0)
    test_indices = np.random.choice(H_sample_size, H_test_size, replace = False)
    with open(test_idx_fp, 'wb') as fp:
        pickle.dump(test_indices, fp)
        
    real_H_image_dir = img_dir.format('healthy', 'real')
    image_indices = collect_img_index(real_H_image_dir)
    H_image_test_indices = image_indices[test_indices]
    with open(idx_dir + 'H_image_test_indices', 'wb') as fp:
        pickle.dump(H_image_test_indices, fp)
    
    real_F_image_dir = img_dir.format('faulty', 'real')
    F_image_test_indices = collect_img_index(real_F_image_dir)
    with open(idx_dir + 'F_image_test_indices', 'wb') as fp:
        pickle.dump(F_image_test_indices, fp)
        
H_test_size = len(test_indices)
H_train_size = H_sample_size - H_test_size
n_rep = F_sample_size // H_train_size
train_size = n_rep*H_train_size
F_test_size = F_sample_size - train_size

images = np.delete(real_healthy_imgs, test_indices, axis = 0)
train_H_images = np.repeat(images, n_rep, axis = 0)
test_H_images = real_healthy_imgs[test_indices]
train_H_labels = np.zeros((train_size, 1))
test_H_labels = np.zeros((H_test_size, 1))

train_F_images = syn_faulty_imgs[:train_size]
test_F_images = syn_faulty_imgs[train_size:]
train_F_labels = np.ones((train_size, 1))
test_F_labels = np.ones((F_test_size, 1))

image_load_dir = img_dir.format('faulty', 'real')
real_faulty_imgs = collect_img_data(image_load_dir, p = 1, flag = 0)
real_F_images = unify_color(real_faulty_imgs)
real_F_labels = np.ones((len(real_F_images), 1))

x_train = np.concatenate([train_H_images, train_F_images])
y_train = np.concatenate([train_H_labels, train_F_labels])
x_train, y_train = shuffle(x_train, y_train, random_state = 0)

x_train = x_train.astype(np.float64)
test_H_images = test_H_images.astype(np.float64)
test_F_images = test_F_images.astype(np.float64)
real_F_images = real_F_images.astype(np.float64)

print(f'[training data] input shape: {x_train.shape}, output shape: {y_train.shape}')
print(f'[testing data (healthy)] input shape: {test_H_images.shape}, output shape: {test_H_labels.shape}')
print(f'[testing data (faulty-generated)] input shape: {test_F_images.shape}, output shape: {test_F_labels.shape}')
print(f'[testing data (faulty-real)] input shape: {real_F_images.shape}, output shape: {real_F_labels.shape}')

#------------------------------------------------------------------------------
num_classes = 2
input_shape = (128, 128, 1)
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 16
num_epochs = 20  # For real training, use num_epochs=100. 10 is a test value
image_size = 128  # We'll resize input images to this size
patch_size = 8  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [
    2048,
    1024,
]  # Size of the dense layers of the final classifier

#------------------------------------------------------------------------------
data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",
)
# Compute the mean and the variance of the training data for normalization.
data_augmentation.layers[0].adapt(x_train)

#------------------------------------------------------------------------------
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

#------------------------------------------------------------------------------
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        images = tf.cast(images, tf.float32)
        patches = ops.image.extract_patches(images, size = self.patch_size)
        patches = ops.reshape(
            patches, (batch_size,
                      num_patches_h * num_patches_w,
                      self.patch_size * self.patch_size * channels)
            )
        
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

#------------------------------------------------------------------------------
plt.figure(figsize=(4, 4))
image = x_train[np.random.choice(range(x_train.shape[0]))]
if image.shape[2] == 1: image = np.repeat(image, 3, axis = 2)
plt.imshow(image.astype("uint8"))
plt.axis("off")

resized_image = ops.image.resize(
    ops.convert_to_tensor([image]), size=(image_size, image_size)
)
patches = Patches(patch_size)(resized_image)
print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")

n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = ops.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(ops.convert_to_numpy(patch_img).astype("uint8"))
    plt.axis("off")

#------------------------------------------------------------------------------
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = ops.expand_dims(
            ops.arange(start=0, stop=self.num_patches, step=1), axis=0
        )
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config

#------------------------------------------------------------------------------
def create_vit_classifier():
    inputs = keras.Input(shape=input_shape)
    # Augment data.
    augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon = 1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads = num_heads, 
            key_dim = projection_dim, 
            dropout = 0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon = 1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units = transformer_units, dropout_rate = 0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon = 1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units = mlp_head_units, dropout_rate = 0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    
    return model

#------------------------------------------------------------------------------
def run_experiment(model):
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)
    if os.path.exists(model_fp): 
        model.load_weights(model_fp)
    
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate, 
        weight_decay=weight_decay
        )
    
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )
    
    monitor_metric = 'val_accuracy'
    if os.path.exists(history_fp): 
        with open(history_fp, 'rb') as fp:
            prev_history = pickle.load(fp)
            monitor_metric_thres = max(prev_history[monitor_metric])
    else:
        monitor_metric_thres = 0

    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath = checkpoint_fp,
        monitor = monitor_metric,
        save_best_only = True,
        save_weights_only = True,
        initial_value_threshold = monitor_metric_thres
    )

    history = model.fit(
        x = x_train,
        y = y_train,
        batch_size = batch_size,
        epochs = num_epochs,
        validation_split = 0.1,
        callbacks = [checkpoint_callback],
    )

    if os.path.exists(history_fp):
        with open(history_fp, 'rb') as fp:
            hist_dict = pickle.load(fp)
        for key in hist_dict.keys():
            hist_dict[key] += history.history[key]
    else:
        hist_dict = history.history

    model.save_weights(model_fp)
    with open(history_fp, 'wb') as fp:
        pickle.dump(hist_dict, fp)

    return hist_dict

def plot_history(hist_dict, item):
    plt.plot(hist_dict[item], label = item)
    plt.plot(hist_dict["val_" + item], label = "val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize = 14)
    plt.legend()
    plt.grid()
    plt.show()
#------------------------------------------------------------------------------
vit_classifier = create_vit_classifier()
hist_dict = run_experiment(vit_classifier)
plot_history(hist_dict, 'loss')
plot_history(hist_dict, 'accuracy')

#------------------------------------------------------------------------------
ViT_model = create_vit_classifier()
optimizer = keras.optimizers.AdamW(learning_rate = learning_rate, weight_decay = weight_decay)
ViT_model.compile(
    optimizer = optimizer,
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
             keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),],
    )
ViT_model.load_weights(model_fp)

_, accuracy, _ = ViT_model.evaluate(test_H_images, test_H_labels)
print(f'Test accuracy (healthy): {round(100*accuracy, 2)}%')
_, accuracy, _ = ViT_model.evaluate(test_F_images, test_F_labels)
print(f'Test accuracy (generated faulty): {round(100*accuracy, 2)}%')
_, accuracy, _ = ViT_model.evaluate(real_F_images, real_F_labels)
print(f'Test accuracy (real faulty): {round(100*accuracy, 2)}%')

with open(history_fp, 'rb') as fp:
    hist_dict = pickle.load(fp)
plot_history(hist_dict, 'loss')
plot_history(hist_dict, 'accuracy')