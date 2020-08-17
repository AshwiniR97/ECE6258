"""
                        ECE 6258
                Digital Image Processing
                    Final Project
        Stephanie Sagun, Ashwini Rajasekhar, Dominick Freeman

    This file includes functions to perform highpass filtering on each of
         the images and store them in the hpf_data directory
"""

import os

import cv2
import dippykit as dip
import numpy as np
from matplotlib import pyplot as plt

pokemons = ['bulbasaur', 'squirtle', 'meowth', 'pikachu', 'charmander']
trainTest = ['training', 'testing']


def combiner(b, g, r, threshold):
    sum = np.zeros(b.shape)
    count = -1
    for color in [b, g, r]:
        count = 1 + count
        i, j = np.nonzero(color > threshold[count])
        bool = np.zeros(color.shape)
        bool[i, j] = color[i, j]
        sum = sum + bool

    sum = sum ** 2
    x, y = np.nonzero(sum <= np.mean(sum) + 3 * np.var(sum))  # Tweak
    sum[x, y] = 0
    return sum


def hpf(im, goal, window, j=0):
    """   ~~~~~~~~~~  High Pass Filter   ~~~~~~~~~~~~~~~
    This iterates until it finds a decent contrast, based on 'goal' mean value and 'window' accuracy.
    The function steps are:
    1.  Takes the FFT of each color section of the image
    2.  Removes (makes zero) pixels in each corner of the FFT image (as it is not centered at zero)
    3.  Does an iFFT to reproduce the image based on the modified frequency """

    # Fourier Transform
    F_im = dip.fft2(im)
    h, w = im.shape
    preset = False

    # Was there scope provided
    if j != 0:
        scope = np.array([j])
        preset = True
    else:
        scope = range(0, h)

    # Searching for the appropriate cutoff frequency
    for i in scope:
        freq_square = i

        # Error Check
        q = int(freq_square / 2)
        if q > w:  # Error code
            print("Error! The filter width is larger than the transform!")

        # Take a 1/4 square from each quadrant
        F_im[0:q, 0:q] = 0  # top left
        F_im[0:q, w - q:w] = 0  # top right
        F_im[h - q:h, 0:q] = 0  # bottom left
        F_im[h - q:h, w - q:w] = 0  # bottom right

        # Take real part only
        im_new = np.abs(dip.ifft2(F_im))

        # Loop if target frequency isn't provided
        if preset == False:
            if (np.mean(im_new) - goal) < window:
                return im_new, i
        else:
            return im_new, i


def highpass_filter(display):
    """
    Function to be called from the mail code, that reads images from
    desired directories, performs high pass filtering on each of them
    and saves the resultant images in new directories.
    :param display: bool value to display/not display outputs
    :return:
    """
    for trainOrTest in trainTest:
        resultPath = os.path.join('hpf_data', trainOrTest)
        originalPath = 'original_data'
        for pokemon in pokemons:
            pokeData = os.path.join(originalPath, trainOrTest, pokemon)
            files = os.listdir(pokeData)
            for picture in files:
                # Setting path
                path = os.path.join(pokeData, picture)

                # Reading image
                Img = dip.im_to_float(cv2.imread(path, 1))

                # Splitting the image into blue, green, red portions
                b, g, r = cv2.split(Img)

                # Splitting image, taking mean
                avg = np.mean([np.mean(b.flatten()), np.mean(g.flatten()), np.mean(r.flatten())])

                # Finding acceptable frequency
                precision = 0.002
                target = avg / 12
                _, j = hpf(b, target, precision)

                # Running hpf
                b_out, _ = hpf(b, target, precision, j)
                g_out, _ = hpf(g, target, precision, j)
                r_out, _ = hpf(r, target, precision, j)

                # Normalizing mean to 1
                b_out = b_out * (1 / np.max(b_out))
                g_out = g_out * (1 / np.max(g_out))
                r_out = r_out * (1 / np.max(r_out))

                # Combiner (Logic)
                std = 100  # how many standard deviations above mean for rgb parts
                sigmas = [np.var(b_out) ** 0.5, np.var(g_out) ** 0.5, np.var(r_out) ** 0.5]
                means = [np.mean(b_out), np.mean(g_out), np.mean(r_out)]
                output = combiner(b_out, g_out, r_out, means + sigmas * std)

                output = dip.float_to_im(output)

                if display:
                    plt.subplot(1, 2, 1)
                    plt.title('Original Image')
                    plt.imshow(Img)
                    plt.subplot(1, 2, 2)
                    plt.title("High pass filter result")
                    plt.imshow(output)

                resultPic = os.path.join(resultPath, pokemon, picture)
                # Saving resultant image
                dip.im_write(output, resultPic)
