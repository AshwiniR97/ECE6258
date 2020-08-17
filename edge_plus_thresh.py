"""
                        ECE 6258
                Digital Image Processing
                    Final Project
        Stephanie Sagun, Ashwini Rajasekhar, Dominick Freeman

    This file includes functions to perform edge detection and thresholding
        on each of the images and store them in the edge_data directory
"""
import os

import cv2
import dippykit as dp
import numpy as np
from matplotlib import pyplot as plt

pokemons = ['squirtle', 'bulbasaur', 'meowth', 'pikachu', 'charmander']
trainTest = ['training', 'testing']


def edge_plus_thresh(im, bin_im):
    """
    Edge Detection + Binary Thresholding
    This function takes an image, applies Canny edge detection, and
    compares it a binary thresholded version of the image, in order to
    combine the two images and produce a binary image that extracts line
    art, in preparation for selective blurring in the image processing
    pipeline.
    :param im: Original Image
    :param bin_im: Binary line art image
    :return im_comb: The combined Image
    """

    # Image parameters
    m = im.shape[0]
    n = im.shape[1]

    # Apply Canny edge detection
    im_canny = dp.edge_detect(im, mode='canny')

    # Begin image combination
    im_comb = np.ones((m, n))
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            window = np.array([[im_canny[i - 1, j - 1], im_canny[i - 1, j], im_canny[i - 1, j + 1]],
                               [im_canny[i, j - 1], im_canny[i, j], im_canny[i, j + 1]],
                               [im_canny[i + 1, j - 1], im_canny[i + 1, j], im_canny[i + 1, j + 1]]])
            case1 = bin_im[i, j] == 1
            case2 = np.any(window[0:, 0:] == 1)
            if case1 or case2:
                im_comb[i, j] = 0
            else:
                im_comb[i, j] = 1
    return im_comb


def edge_plus_thres(display):
    """
    Main function that is called by main.py that runs through all the images
    in the dataset and calls subsequent edge detection and thresholding functions.
    The resultant images are stored in the edge_data folder.
    :param display: bool value to display/not display outputs
    """
    origData = 'original_data'
    binaryData = 'binary_data'
    resultData = 'edge_data'

    for pokemon in pokemons:
        for trainOrTest in trainTest:
            origDataPath = os.path.join(origData, trainOrTest, pokemon)
            binDataPath = os.path.join(binaryData, trainOrTest, pokemon)
            resultDataPath = os.path.join(resultData, trainOrTest, pokemon)

            files = os.listdir(origDataPath)
            for picture in files:

                origPicture = os.path.join(origDataPath, picture)
                binPicture = os.path.join(binDataPath, picture)

                # Read in images
                Img = dp.im_to_float(cv2.imread(origPicture, 0))
                Bin_Img = dp.im_to_float(cv2.imread(binPicture, 0))

                # Generate new line art image for the selective blurring procedure
                output = edge_plus_thresh(Img, Bin_Img)

                # Write output to folder
                output = dp.float_to_im(output)
                if display:
                    plt.subplot(1, 3, 1)
                    plt.title('Original Image')
                    plt.imshow(Img)
                    plt.subplot(1, 3, 2)
                    plt.title('Binary Image')
                    plt.imshow(Bin_Img)
                    plt.subplot(1, 3, 3)
                    plt.title('Edge Detected + Thresholded Image')
                    plt.imshow(output)
                    plt.show()

                resultPic = os.path.join(resultDataPath, picture)
                dp.im_write(output, resultPic)
