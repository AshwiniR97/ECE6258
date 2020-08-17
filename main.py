"""
                        ECE 6258
                Digital Image Processing
                    Final Project
        Stephanie Sagun, Ashwini Rajasekhar, Dominick Freeman

    This is the main file that is to be run to call the subsequent Image
                Processing functions
"""
import warnings

from binary_threshold import binary_threshold
from crop_image import crop_image
from edge_plus_thresh import edge_plus_thres
from highpass_filter import highpass_filter
from makeDir import makeDir
from selective_blur import selective_blur_main

warnings.filterwarnings("ignore")

# Set display = True if outputs are to be seen.
# Note that this displays every single output from each folder
# Only set display = True when one of the following functions is being run and outputs are desired
display = True

makeDir()
print("Directories Made")

# highpass_filter(display)
print("High pass filter completed")

# binary_threshold(display)
print("Binary Thresholding Complete")

# edge_plus_thres(display)
print("Edge Detection + Thresholding Complete")

selective_blur_main(display)
print("Selective Blurring Complete")

crop_image(display)
print("Image cropping complete")
