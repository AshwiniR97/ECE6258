# ECE6258
ECE6258 At Georgia Tech - Final Project

Required Libraries:
Tensorflow
Numpy
OpenCV (cv2)
Dippykit
random
matplotlib
os

Folder structure required for execution:
├───Final Project Combined
│   ├───original_data
│       ├───testing
│       │   ├───bulbasaur
│       │   ├───charmander
│       │   ├───meowth
│       │   ├───pikachu
│       │   └───squirtle
│       └───training
│           ├───bulbasaur
│           ├───charmander
│           ├───meowth
│           ├───pikachu
│           └───squirtle
├───cropped_model
│   ├───model.json
│   ├───weights.h5
│   ├───weights.best.h5
├───blur_model
│   ├───model.json
│   ├───weights.h5
│   ├───weights.best.h5
├───binary_threshold.py
├───crop_image.py
├───edge_plus_thresh.py
├───highpass_filter.py
├───imageGenerator.py
├───main.py
├───makeDir.py
├───selective_blur.py
├───test.py
├───training.py
├───README.txt


Image directories of original data and data generated by the program and that can be used to run the classifier on different processed images:

1. original_data
2. hpf_data
3. binary_data
4. edge_data
5. blur_data
6. cropped_data


The entire training dataset and testing images can be found here.

Execution process:
1. Set variable display in main.py (Line 24) as True or False depending on whether output display for each image is required or not.
2. Run main.py. This will complete all the image processing techniques. Resultant images can be found in the newly generated directories as mentioned in the code.
3. Change variable mainDir in imageGenerator.py (Line 40) to point to the Final Project Combined folder on the user’s system.
4. Change variable folder in imageGenerator.py (Line 37) to point to the folder chosen for image augmentation.
5. Run imageGenerator.py. This should generate a new directory newTraining under every image directory and contain the augmented data for each Pokemon.
6. Change variable mainDir in training.py (Line 25) to point to the Final Project Combined folder on the user’s system.
7. Change variable folder in training.py (Line 26) to choose between all the image directories mentioned above. 
8. Change variable typeO in training.py (Line 29) to choose between augmented data and original data. Enter training for original data and newTraining for augmented data.
9. Run training.py. This should train the CNN and save the weight files in their respective directories.
10. Change variable mainDir in test.py (Line 23) to point to the Final Project Combined folder on the user’s system.
11. Change variable folder in test.py (Line 26) to choose between all the image directories mentioned above. 
12. Change variable typeO in test.py (Line 29) to choose between augmented data and original data. Enter training for original data and newTraining for augmented data.
13. Adjust values of num_rows and num_cols in test.py(Line 128-129) to output the 
14. Run test.py. This should load the weights of the CNN file and test the images in the folder test and print a testing accuracy.

Note: 
Variables folder and typeO must be the same in both training.py and test.py to ensure CNN is training and testing similarly processed data.

To run the test files directly:
1. Go to folder cropped_model or blur_model to use the weight files and model file directly. 
2. Change variable filepath in training.py (Line 32) to point to weights.best.h5 file on user system.
3. Change variable modelpath in training.py (Line 33) to point to model.json file on user system.
4. Change variable weightFile in training.py (Line 34) to point to weights.h5 file on user system.
5. Change variable filepath in test.py (Line 32) to point to weights.best.h5 file on user system.
6. Change variable modelpath in test.py (Line 33) to point to model.json file on user system.
7. Change variable weightFile in test.py (Line 34) to point to weights.h5 file on user system.

Note: The weight files and model files are values that were trained on augmented data only, that produce the best results.

