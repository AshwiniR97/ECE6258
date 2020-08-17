"""
                        ECE 6258
                Digital Image Processing
                    Final Project
        Stephanie Sagun, Ashwini Rajasekhar, Dominick Freeman

    This file includes functions to perform selective blurring based on the line art
        on each of the images and store them in the blur_data directory
"""
import os

import cv2
import dippykit as dp
import numpy as np
from matplotlib import pyplot as plt

pokemons = ['bulbasaur', 'charmander', 'pikachu', 'squirtle', 'meowth']  # Select the pokemon (folder name)
trainTest = ['training', 'testing']


def selective_blur(im, lineart):
    """
    This function blurs image content that is not identified to be line art.
    :param im: Original Image
    :param lineart: The Binary Line Art Image
    :return new_im: The selectively blurred image
    """
    new_im = np.zeros(im.shape)
    # begin processing
    for i in range(3, im.shape[0] - 3):
        for j in range(3, im.shape[1] - 3):
            for k in range(3):
                im_window = np.array([[im[i - 3, j - 3, k], im[i - 3, j - 2, k], im[i - 3, j - 1, k], im[i - 3, j, k],
                                       im[i - 3, j + 1, k], im[i - 3, j + 2, k], im[i - 3, j + 3, k]],
                                      [im[i - 2, j - 3, k], im[i - 2, j - 2, k], im[i - 2, j - 1, k], im[i - 2, j, k],
                                       im[i - 2, j + 1, k], im[i - 2, j + 2, k], im[i - 2, j + 3, k]],
                                      [im[i - 1, j - 3, k], im[i - 1, j - 2, k], im[i - 1, j - 1, k], im[i - 1, j, k],
                                       im[i - 1, j + 1, k], im[i - 1, j + 2, k], im[i - 1, j + 3, k]],
                                      [im[i, j - 3, k], im[i, j - 2, k], im[i, j - 1, k], im[i, j, k], im[i, j + 1, k],
                                       im[i, j + 2, k], im[i, j + 3, k]],
                                      [im[i + 1, j - 3, k], im[i + 1, j - 2, k], im[i + 1, j - 1, k], im[i + 1, j, k],
                                       im[i + 1, j + 1, k], im[i + 1, j + 2, k], im[i + 1, j + 3, k]],
                                      [im[i + 2, j - 3, k], im[i + 2, j - 2, k], im[i + 2, j - 1, k], im[i + 2, j, k],
                                       im[i + 2, j + 1, k], im[i + 2, j + 2, k], im[i + 2, j + 3, k]],
                                      [im[i + 3, j - 3, k], im[i + 3, j - 2, k], im[i + 3, j - 1, k], im[i + 3, j, k],
                                       im[i + 3, j + 1, k], im[i + 3, j + 2, k], im[i + 3, j + 3, k]]])
                lin_window = np.array([[lineart[i - 1, j - 1], lineart[i - 1, j], lineart[i - 1, j + 1]],
                                       [lineart[i, j - 1], lineart[i, j], lineart[i, j + 1]],
                                       [lineart[i + 1, j - 1], lineart[i + 1, j], lineart[i + 1, j + 1]]])
                if np.any(lin_window[0:, 0:, ] == 0.0):
                    new_im[i, j, k] = im[i, j, k]
                else:
                    new_im[i, j, k] = np.mean(im_window)
    return new_im


def selective_blur_main(display):
    """
    Main function that is called by main.py that runs through all the images
    in the dataset and selectively blurres the image using the line art stored
    in binary_data folder.
    The resultant images are stored in the blur_data folder.
    :param display: bool value to display/not display outputs
    """
    origData = 'original_data'
    edgeData = 'edge_data'
    resultData = 'blur_data'

    for pokemon in pokemons:
        for trainOrTest in trainTest:
            origDataPath = os.path.join(origData, trainOrTest, pokemon)
            edgeDataPath = os.path.join(edgeData, trainOrTest, pokemon)
            resultDataPath = os.path.join(resultData, trainOrTest, pokemon)

            files = os.listdir(origDataPath)
            for picture in files:

                # Setting path
                origPicture = os.path.join(origDataPath, picture)
                edgePicture = os.path.join(edgeDataPath, picture)

                # Reading in image
                Img = dp.im_to_float(cv2.imread(origPicture, 1))
                Lineart = dp.im_to_float(cv2.imread(edgePicture, 0))

                # Apply selective blur
                output = selective_blur(Img, Lineart)
                output = dp.float_to_im(output)
                output = output[:, :, ::-1]

                if display:
                    plt.subplot(1, 3, 1)
                    plt.title('Original Image')
                    plt.imshow(Img[:, :, ::-1])
                    plt.subplot(1, 3, 2)
                    plt.title('Edge Image')
                    plt.imshow(Lineart)
                    plt.subplot(1, 3, 3)
                    plt.title('Selectively Blurred Image')
                    plt.imshow(output)
                    plt.show()

                # Write to folder
                diffImage = output - Img*255
                diffImage = np.array(diffImage)
                plt.imshow(diffImage)
                plt.show()
                resultPic = os.path.join(resultDataPath, picture)
                dp.im_write(diffImage, resultPic)
