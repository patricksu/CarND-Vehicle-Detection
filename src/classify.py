import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from readImageDataset import read_img_path
from sklearn.cross_validation import train_test_split
from get_hog import get_hog_features
from extract_features import (bin_spatial, color_hist, extract_features)




def main():
    (cars, notcars) = read_img_path()
    # Reduce the sample size because HOG features are slow to compute
    sample_size = 5000 #min(len(cars), len(notcars))
    cars = cars[0:sample_size]
    notcars = notcars[0:sample_size]

    has_bin_features = False
    has_hist_features = False
    has_hog_features = True

    ### TODO: Tweak these parameters and see how the results change.
    # Bin feature param
    spatial_size = (32, 32)
    # Hist feature param
    hist_bins=32
    hist_range=(0, 256)
    # Hog feature param
    colorspace = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"

    t=time.time()

    # Extract the features (three kinds)
    car_features = extract_features(cars, has_bin_features, has_hist_features, has_hog_features, cspace=colorspace, spatial_size=spatial_size,
                            hist_bins=hist_bins, hist_range=hist_range, orient=orient, 
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                            hog_channel=hog_channel)
    notcar_features = extract_features(notcars, has_bin_features, has_hist_features, has_hog_features, cspace=colorspace, spatial_size=spatial_size,
                            hist_bins=hist_bins, hist_range=hist_range, orient=orient, 
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                            hog_channel=hog_channel)

    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to extract HOG features...')
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)
    if has_bin_features:
        print('Using bin features, spatial_size is: {}'.format(spatial_size))
    if has_hist_features:
        print('Using hist features, hist_bins is: {}, hist_range is: {}'.format(hist_bins, hist_range))
    if has_hog_features:
        print('Using hog features, orient is: {}, pix_per_cell is: {}, cell_per_block is: {}'.format(orient, pix_per_cell, cell_per_block))
    print('The total feature vector length is: {}'.format(len(X_train[0])))
    

    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = len(X_test)
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')


if __name__ == '__main__':
    main()