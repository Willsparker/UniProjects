import cv2
import numpy as np
import os
import h5py

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA as skPCA
import mahotas #(For Zernike)

# Big Big Help:
# Random Forest Classifier: https://github.com/87surendra/Random-Forest-Image-Classification-using-Python/blob/master/Random-Forest-Image-Classification-using-Python.ipynb

imageSize = (250,250)
PCA_COMPONENT_NUMBER = 1
ZERNIKE_RADIUS = 30

def Hu_Moments(image):
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def Zernike_Moments(image):
    return mahotas.features.zernike_moments(image,ZERNIKE_RADIUS)

def RGB_Histogram(image):
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Calculate Histogram, Normalise it and put it in the right format e.g. (x,) 
    hist  = cv2.calcHist([image],[0,1,2],None,[8,8,8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist,hist)
    return hist.flatten()

def PCA(image, number_of_components=1):
    image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    if number_of_components == 1:
    # Returns the first principle component of the depth image
    # Retrieved from : https://stackoverflow.com/a/43446362
        data = image - np.mean(image, axis=0)
        scatter_matrix = np.dot(data, data.T)
        eig_val, eig_vec = np.linalg.eig(scatter_matrix)
        new_reduced_data = np.real(np.sqrt(eig_val[0]) * eig_vec.T[0].reshape(-1,1))
        return np.squeeze(new_reduced_data)
    else:
        pca = skPCA(n_components=number_of_components)
        pca.component = True
        pca_img = pca.fit_transform(image)
    # Puts the PCA image into a 1D list, as required.
    return np.squeeze(pca_img.reshape(-1,1))


def main():
    trainingPath = "./TrainingSet/TrainingSets/"
    # Read in the training set
    set1_labels_file = open("Set1Labels.txt")
    TrainingLabels = set1_labels_file.readlines()
    # Remove the newline characters for each element
    TrainingLabels = [x.replace('\n','') for x in TrainingLabels]
    GlobalFeatures = []
    GlobalLabels = []
    TrainingLabels.sort()
    # For each label (i.e. Android, Baby, etc)
    for TrainLabel in TrainingLabels:
        filepath = TrainLabel + "/depth/"
        dir = os.path.join(trainingPath,filepath)
        # For all the files
        for file in os.listdir(dir):
            # Depth and RGB images have the same name (<number>.jpg), but in their own depth or rgb directory
            depth_image = cv2.resize(cv2.imread(trainingPath + TrainLabel + "/depth/" + file, cv2.IMREAD_GRAYSCALE), imageSize)
            rgb_image = cv2.resize(cv2.imread(trainingPath + TrainLabel + "/rgb/" + file, cv2.IMREAD_ANYCOLOR), imageSize)

            # Find the Hu / Zernike Moments for the depth images
            if depth_image is not None:
                F_Hu = Hu_Moments(depth_image.copy())
                F_Zernike = Zernike_Moments(depth_image.copy())
            
            # Find the Color Histogram & reduce RGB image to 1st principle component for
            if rgb_image is not None:
                F_ColorHist = RGB_Histogram(rgb_image.copy())
                F_PCA = PCA(rgb_image.copy(),PCA_COMPONENT_NUMBER)

            #print(F_Hu.shape)
            #print(F_ColorHist.shape)
            #print(F_Zernike.shape)
            #print(F_PCA.shape)

            # Put into array, add that array to the list of global features
            global_feature = np.hstack([F_Hu,F_ColorHist,F_PCA,F_Zernike])
            GlobalFeatures.append(global_feature)
            # Track which label this set of features belongs to
            GlobalLabels.append(TrainLabel)
    
    GlobalFeatures = np.array(GlobalFeatures)
    print("GlobalFeatures Size:", GlobalFeatures.shape )
    
    # Encode the target labels
    targetNames = np.unique(GlobalLabels)
    le = LabelEncoder()
    target = le.fit_transform(GlobalLabels)

    # normalize the feature vector in the range (0-1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaled_features = scaler.fit_transform(GlobalFeatures)

    print("Target labels: {}".format(target)) # This should output a [0 , 0 , 0 ... 13 , 13 , 13]
    print("Target labels shape: {}".format(target.shape))

    # save the feature vector using HDF5
    h5f_data = h5py.File('output/data.h5', 'w')
    h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))
    # Save the labels vector too
    h5f_label = h5py.File('output/labels.h5', 'w')
    h5f_label.create_dataset('dataset_1', data=np.array(target))

    h5f_data.close()
    h5f_label.close()

    print("Dataset for Random Forest produced")

if __name__ == "__main__":
    main()
    exit()