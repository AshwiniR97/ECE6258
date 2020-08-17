"""
                        ECE 6258
                Digital Image Processing
                    Final Project
        Stephanie Sagun, Ashwini Rajasekhar, Dominick Freeman

    This file includes functions to perform contour detection and corner
    identification (Shi-Tomasi Algorithm) on each of the images and store
                    them in the edge_data directory
"""
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

pokemons = ['pikachu', 'squirtle', 'bulbasaur', 'meowth', 'charmander']  # Select the pokemon (folder name)
trainTest = ['training', 'testing']


def minArea(im):
    """
    Function to calculate area of an image
    :param im: Image of which area is to be found
    :return: Area of the Image in sq. pixels
    """
    w = im.shape[0]
    h = im.shape[1]
    return (w * h) / 25


def contourPlot(im, contourIm):
    """
    Function to find various contours on the image, pick
    the largest 10 out of them all and find the largest outline
    that encases the Pokemon.
    :param im: Image to find prominent contours on
    :return: Dimensions of the largest contour on the image
    """
    cont, hier = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contours = sorted(cont, key=cv2.contourArea)[-10:]

    x_fin = []
    y_fin = []
    x_fin_end = []
    y_fin_end = []

    for i in range(len(largest_contours)):
        x, y, w, h = cv2.boundingRect(largest_contours[i])
        # draw the contours in blue
        cv2.rectangle(contourIm, (x, y), (x + w, y + h), (0, 0, 255), 5)

        x_fin.append(x)
        y_fin.append(y)
        x_fin_end.append(x + w)
        y_fin_end.append(y + h)

    xMin = np.min(x_fin)
    yMin = np.min(y_fin)
    xMax = np.max(x_fin_end)
    yMax = np.max(y_fin_end)

    return xMin, yMin, xMax, yMax


def cornerPlot(im, cornerIm):
    """
    Function to find various corners and features on the image, pick
    the largest 10 out of them all and find the largest outline
    that encases the Pokemon.
    :param im:
    :return:
    """
    w, h = im.shape
    points = round(max(w, h) / 8 + 10)
    corners = cv2.goodFeaturesToTrack(im, points, 0.01, 10)
    corners = np.int0(corners)

    X = []
    Y = []

    for i in corners:
        x, y = i.ravel()
        X.append(x)
        Y.append(y)
        cv2.circle(cornerIm, (x, y), 5, 255, -1)

    xMax = np.max(X)
    xMin = np.min(X)
    yMax = np.max(Y)
    yMin = np.min(Y)

    return xMin, yMin, xMax, yMax


def crop_image(display):
    """
    Main function that is called by main.py that runs through all the images
    in the dataset and uses the list of line art in the binary_data folder to
    find appropriate bounding boxes and crop the image using the best bounding
    box.
    Resultant cropped images are stored in cropped_data folder.
    :param display: bool value to display/not display outputs
    """
    origData = 'original_data'
    binData = 'binary_data'
    resultData = 'cropped_data'

    for pokemon in pokemons:
        for trainOrTest in trainTest:
            origDataPath = os.path.join(origData, trainOrTest, pokemon)
            binDataPath = os.path.join(binData, trainOrTest, pokemon)
            resultDataPath = os.path.join(resultData, trainOrTest, pokemon)

            origImages = os.listdir(origDataPath)
            binImages = os.listdir(binDataPath)

            totFiles = len(origImages)
            for imageNumber in range(totFiles):
                origImageName = os.path.join(origDataPath, origImages[imageNumber])
                binName = os.path.join(binDataPath, binImages[imageNumber])

                image = cv2.imread(origImageName)
                cornerIm = cv2.imread(origImageName)
                contourIm = cv2.imread(origImageName)
                binImage = cv2.imread(binName, 0)

                x1, y1, x2, y2 = contourPlot(binImage, contourIm)

                x3, y3, x4, y4 = cornerPlot(binImage, cornerIm)

                if display:
                    plt.subplot(1, 3, 1)
                    plt.title('Corner Method')
                    cv2.rectangle(cornerIm, (x3, y3), (x4, y4), (0, 0, 255), 2)
                    plt.imshow(cornerIm[:, :, ::-1])

                    plt.subplot(1, 3, 2)
                    plt.title('Contour Method')
                    cv2.rectangle(contourIm, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    plt.imshow(contourIm[:, :, ::-1])

                areaContour = (x2 - x1) * (y2 - y1)
                areaCorner = (x4 - x3) * (y4 - y3)

                if areaCorner <= areaContour:
                    image = image[y3:y4, x3:x4]

                elif areaCorner <= minArea(image):
                    pass
                else:
                    image = image[y1:y2, x1:x2]

                if display:
                    plt.subplot(1, 3, 3)
                    plt.title('Cropped Image')
                    plt.imshow(image[:, :, ::-1])
                    plt.show()

                resultPic = os.path.join(resultDataPath, origImages[imageNumber])
                cornerPic = os.path.join(resultDataPath, "isthisoutput.png")
                contourPic = os.path.join(resultDataPath, "isthisoutput2.png")
                cv2.imwrite(resultPic, image)
                cv2.imwrite(cornerPic, cornerIm)
                cv2.imwrite(contourPic, contourIm)
