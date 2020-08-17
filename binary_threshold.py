"""
                        ECE 6258
                Digital Image Processing
                    Final Project
        Stephanie Sagun, Ashwini Rajasekhar, Dominick Freeman

    This file includes functions to perform binary thresholding on each of
         the images and store them in the binary_data directory
"""
import os

import cv2
import dippykit as dp
import numpy as np
from matplotlib import pyplot as plt

pokemons = ['bulbasaur', 'squirtle', 'meowth', 'pikachu', 'charmander']
trainTest = ['training', 'testing']


def bin_thresh(im, thresh):
    """
    Binary Threshold
    This function takes an image and converts it to a
    binary image for further processing in
    the image processing pipeline.
    """

    new_im = np.zeros(im.shape)
    M = im.shape[0]
    N = im.shape[1]
    for i in range(M):  # rows
        for j in range(N):  # columns
            pixel = im[i, j]
            if pixel > thresh:
                new_im[i, j] = 1
            else:
                new_im[i, j] = 0

    return new_im


def otsu_thresh(im):
    """
    Binary Threshold with Otsu's Method
    This function takes an image and converts it to a binary
    image using Otsu's Method for further processing in the image
    processing pipeline.
    """

    otsu_im = np.zeros(im.shape)
    M = im.shape[0]
    N = im.shape[1]
    MN = M * N

    # Compute histogram
    u_im, counts = np.unique(im, return_counts=True)

    # Begin Otsu's Method
    iterations = len(counts)
    p1 = np.zeros(iterations)
    p2 = np.zeros(iterations)
    m1 = np.zeros(iterations)
    m2 = np.zeros(iterations)
    var_b = np.zeros(iterations)
    max_var_b = 0
    for k in range(len(counts)):
        p1[k] = sum(counts[0:k + 1]) / MN
        p2[k] = 1 - p1[k]

        for i in range(k + 1):  # compute m1[k]
            pi1 = counts[i] / MN
            m1[k] = m1[k] + (i * pi1)

        for i in range(k + 1, len(counts)):  # compute m2[k]
            pi2 = counts[i] / MN
            m2[k] = m2[k] + (i * pi2)

        if k < (iterations - 1):
            m1[k] = m1[k] / p1[k]
            m2[k] = m2[k] / p2[k]

        first = p1[k] * p2[k]
        second = (m1[k] - m2[k]) ** 2
        var_b[k] = first * second

        # maximize between-class variance
        if var_b[k] > max_var_b:
            max_var_b = var_b[k]
            thresh = k

    # Begin binary thresholding
    for i in range(M):
        for j in range(N):
            pixel = im[i, j]
            if pixel <= u_im[thresh]:
                otsu_im[i, j] = 0
            else:
                otsu_im[i, j] = 1

    return otsu_im


def binary_threshold(display):
    """
    Main function that is called by main.py that runs through all the images
    in the dataset and uses the list of high-pass filtered files in a given folder
    (defined by the pokemon set earlier) and runs the image through binary thresholding,
    using a set threshold intensity = 0.4 and also using Otsu's Method
    The resultant images are stored in the binary_data folder.
    :param display: bool value to display/not display outputs
    """
    # Set paths
    hpfData = 'hpf_data'

    for pokemon in pokemons:
        for trainOrTest in trainTest:
            pokemon_path = os.path.join(hpfData, trainOrTest, pokemon)
            resultPath = os.path.join('binary_data', trainOrTest, pokemon)
            files = os.listdir(pokemon_path)

            for picture in files:
                # Setting path
                path = os.path.join(pokemon_path, picture)
                # Reading in image
                Img = dp.im_to_float(cv2.imread(path, 0))

                # Regular binary thresholding
                # reg_output = bin_thresh(Img, thresh=0.4)
                # reg_output = dp.float_to_im(reg_output)
                # dp.im_write(reg_output, 'binary_data' + '/' + set_type + '/' + pokemon + '/' + picture)

                # Otsu's Method thresholding
                otsu_output = otsu_thresh(Img)
                otsu_output = dp.float_to_im(otsu_output)

                if display:
                    plt.subplot(1, 2, 1)
                    plt.title('Original HPF Image')
                    plt.imshow(Img)
                    plt.subplot(1, 2, 2)
                    plt.title('Thresholded Image')
                    plt.imshow(otsu_output)
                    plt.show()

                # Output directory
                resultName = os.path.join(resultPath, picture)
                dp.im_write(otsu_output, resultName)
