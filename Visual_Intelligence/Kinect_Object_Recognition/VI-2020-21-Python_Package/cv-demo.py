#!/usr/bin/env python

from argparse import ArgumentParser
from FreenectPlaybackWrapper.PlaybackWrapper import FreenectPlaybackWrapper
import numpy as np

# Used for labelling the objects
import datetime as dt

# Will be used for keeping directories tidy
import os
import shutil

import cv2

# For Validating
from sklearn.ensemble import RandomForestClassifier
import h5py
import buildClassifier as bC


# If this is True, we build the training set
TRAINING = False

# If this is True, we want to build and apply the classifier
VALIDATING = True
img_size = bC.imageSize

def main():
    parser = ArgumentParser(description="OpenCV Demo for Kinect Coursework")
    parser.add_argument("videofolder", help="Folder containing Kinect video. Folder must contain INDEX.txt.",
                        default="ExampleVideo", nargs="?")
    parser.add_argument("--no-realtime", action="store_true", default=False)

    args = parser.parse_args()

    # Read the labels into a list
    set1_labels_file = open("Set1Labels.txt")
    set1_labels = set1_labels_file.readlines()
    # Remove the newline characters for each element
    set1_labels = [x.replace('\n','') for x in set1_labels]

    if TRAINING:
        label_count = 0
        frame_count = 0
        image_count = 0
        frame_threshold = 145

        # Remove old Training sample runs, and make new dirs
        for x in set1_labels:
            filename = "./TrainingSets/" + x
            shutil.rmtree(filename + "/depth")
            shutil.rmtree(filename + "/rgb")
            os.makedirs(filename + "/depth")
            os.makedirs(filename + "/rgb")

    if VALIDATING:
        set1_labels.sort()
        # Load in the classifier data
        h5f_data = h5py.File('output/data.h5', 'r')
        h5f_label = h5py.File('output/labels.h5', 'r')

        global_features_string = h5f_data['dataset_1']
        global_labels_string = h5f_label['dataset_1']

        # Get the global features / labels as arrays
        global_features = np.array(global_features_string)
        global_labels = np.array(global_labels_string)

        # Build the Random Forest Classifier
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(global_features,global_labels)

    # Depth images and colour images are offset by an amount
    # Found through trial and error
    x_offset = -45
    y_offset = 30

    # Used to make the threshold better
    morphology_kernel = np.ones((5,5),np.uint8)

    # Thresholds for the size a contour should be
    # This removes noise and stops detecting the whole screen as a contour
    upperThresh = 65000
    lowerThresh = 5000

    for status, rgb, depth in FreenectPlaybackWrapper(args.videofolder, not args.no_realtime):

        # If we have an updated Depth image, then display
        if status.updated_depth:
            # Thresholder
            # This puts any grey values below 75 to 0 (which tends to be the outline of the object)
            _, depth = cv2.threshold(depth,75,255,cv2.THRESH_TOZERO)
            # Any Non-black grey values turn to black, any black grey values turn to white.
            _, depth = cv2.threshold(depth,1,255,cv2.THRESH_BINARY)

            # Equivalent of Dilating the black bit :-)
            depth = cv2.erode(depth,morphology_kernel,iterations = 3)
                
            if TRAINING:
                # This is to make sure all the objects are correctly identified in the training set
                # Trial and error.
                if label_count == 4:
                    frame_threshold = 100
                if label_count == 5:
                    frame_threshold = 75
                elif label_count == 6:
                    frame_threshold = 160
                elif label_count == 12:
                    frame_threshold = 70

            contours, hierarchy = cv2.findContours(depth, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            reduced_contours = [ x for x in contours if cv2.contourArea(x) > lowerThresh and cv2.contourArea(x) < upperThresh]

            if TRAINING:
                # If a certain number of frames have passed without any detected areas, assume the next object has appeared.
                if len(reduced_contours) == 0:
                    frame_count += 1
                    if frame_count > frame_threshold:
                        frame_count = 0
                        label_count +=1

            if len(reduced_contours) != 0:
                contour = reduced_contours[0]
                # Find bounding rectangles
                x,y,w,h = cv2.boundingRect(contour)

                if TRAINING:
                    image_count += 1
                    filename = "./TrainingSets/" + set1_labels[label_count]
                    label_timestamp = dt.datetime.now()
                    # Get the cropped depth image and write it to disk
                    crop_img_depth = depth[y:y+h, x:x+w].copy()
                    cv2.imwrite(filename + "/depth/" + "depth_" + str(image_count) + ".jpg",crop_img_depth)

                if VALIDATING:
                    crop_img_depth = cv2.resize(depth[y:y+h, x:x+w].copy(),img_size)

                x = x + x_offset
                y = y + y_offset

                if TRAINING:
                    # Get cropped RGB image and write it to disk
                    crop_img_rgb = rgb[y:y+h, x:x+w].copy()
                    cv2.imwrite(filename + "/rgb/" + "rgb_" + str(image_count) + ".jpg",crop_img_rgb)
                    # Write what it is above the rectangle
                    cv2.putText(rgb,set1_labels[label_count], (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0),thickness=2)

                if VALIDATING:
                    # Produce the RGB image in a context we need it
                    crop_img_rgb = cv2.resize(rgb[y:y+h, x:x+w].copy(),img_size)
                    
                    # Find the features of all the images
                    F_Hu = bC.Hu_Moments(crop_img_depth.copy())
                    F_ColorHist = bC.RGB_Histogram(crop_img_rgb.copy())
                    F_PCA = bC.PCA(crop_img_rgb.copy(), bC.PCA_COMPONENT_NUMBER)
                    F_Zernike = bC.Zernike_Moments(crop_img_depth.copy())

                    # Put all features together
                    global_feature = np.hstack([F_Hu,F_ColorHist, F_PCA, F_Zernike])
                    # Run it through the forest classifier
                    prediction = clf.predict(global_feature.reshape(1,-1))[0]
                    # Label the object with which object its predicted as
                    cv2.putText(rgb,set1_labels[prediction], (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0),thickness=2)

                # Draw the rectangle
                cv2.rectangle(rgb,(x,y),(x+w,y+h),(0,255,0),2)
                

            cv2.imshow("Thresholded Depth", depth)


        # RGB needs to be shown after depth image has been processed.
        if status.updated_rgb:
            cv2.imshow("RGB", rgb)

        # Check for Keyboard input.
        key = cv2.waitKey(10)

        # Break out of the program if ESC is pressed (OpenCV KeyCode 27 is ESC key)
        if key == 27:
            break

    return 0

if __name__ == "__main__":
    exit(main())
