# -*- coding: utf-8 -*-
"""
                        ECE 6258
                Digital Image Processing
                    Final Project
        Stephanie Sagun, Ashwini Rajasekhar, Dominick Freeman

            File to run for data augmentation of the images.

                Original file is located at
    https://colab.research.google.com/drive/1r7pNzfJIQc8xqtc3e7wZlDFuwkdZOl0e
"""

import os

import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

"""
Using the Image Data Generator class to make a series of distortions on 
the original image to generate a new image.
"""
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Change directories here

# The folder that needs Data Augmentation - Choose from "original_data", "hpf_data", 
# binary_data", "edge_data", "blur_data" and "cropped_data"
folder = "original_data"

# Main Directory
mainDir = "/content/drive/My Drive/pokeData/Final Project Combined"

# Origin and Destination directories
dataDir = os.path.join(mainDir, folder, "training")
saveDir = os.path.join(mainDir, folder, "newTraining")

# Every class/pokemon to be altered
categories = ["bulbasaur", "pikachu", "squirtle", "meowth", "charmander"]

# Generating a new directory to save new images
if not os.path.isdir(saveDir):
    os.mkdir(saveDir)
    for poke in categories:
        os.mkdir(os.path.join(saveDir, poke))

# Going through each image, generating 10 images of unique distortions listed in the 
# ImageGenerator class and savin them in the new directory
for category in categories:
    for image in os.listdir(os.path.join(dataDir, category)):
        os.chdir(os.path.join(dataDir, category))
        img = load_img(image)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_format='jpeg'):
            i += 1
            newim = batch[0, :, :, ::-1] * 255
            os.chdir(os.path.join(saveDir, category))
            cv2.imwrite(str(image) + str(i) + ".png", newim)
            if i > 9:
                break
