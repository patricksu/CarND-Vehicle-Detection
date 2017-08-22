import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from skimage.feature import hog
from readImageDataset import read_img_path
from sklearn.cross_validation import train_test_split
from get_hog import get_hog_features
from extract_features import (bin_spatial, color_hist, extract_features)
import pickle as pkl



def main():
    (cars, notcars) = read_img_path()
    # Reduce the sample size because HOG features are slow to compute
    # cars = cars[0:1000]
    # notcars = notcars[0:1000]

    has_bin_features = True
    has_hist_features = True
    has_hog_features = True

    ### TODO: Tweak these parameters and see how the results change.
    # Bin feature param
    spatial_size = (32, 32)
    # Hist feature param
    hist_bins=32
    hist_range=(0, 256)
    # Hog feature param
    colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
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

    X_reducer = PCA(n_components=200)
    X_reducer.fit(scaled_X)

    reduced_X = X_reducer.transform(scaled_X)

    # plt.plot(np.array(range(200)), np.cumsum(X_pca.explained_variance_ratio_))
    # plt.show()
    print('Total explained variance ratio is: {}'.format(np.sum(X_reducer.explained_variance_ratio_)))


    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


    # Split up data into randomized training and test sets
    rand_state = np.random.randint(1, 1500)
    X_train, X_test, y_train, y_test = train_test_split(
        reduced_X, y, test_size=0.2, random_state=rand_state)
    if has_bin_features:
        print('Using bin features, spatial_size is: {}'.format(spatial_size))
    if has_hist_features:
        print('Using hist features, hist_bins is: {}, hist_range is: {}'.format(hist_bins, hist_range))
    if has_hog_features:
        print('Using hog features, orient is: {}, pix_per_cell is: {}, cell_per_block is: {}'.format(orient, pix_per_cell, cell_per_block))
    print('The total feature vector length is: {}'.format(len(X_train[0])))
    

    use_CV = False
    if use_CV:
        # Use a linear SVC 
        svc = LinearSVC()
        # Check the training time for the SVC
        t=time.time()
        parameters = {'C':[0.01, 0.1, 0.5, 1, 5, 10]}
        clf = GridSearchCV(svc, parameters)
        clf.fit(X_train, y_train)
        t2 = time.time()

        print('Best parameters are:{}'.format(clf.best_params_))

        print(round(t2-t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t=time.time()
        n_predict = len(X_test)
        print('My SVC predicts: ', clf.predict(X_test[0:n_predict]))
        print('For these',n_predict, 'labels: ', y_test[0:n_predict])
        t2 = time.time()
        print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

        plt.plot(parameters['C'], clf.cv_results_['mean_test_score'])
        plt.show()
    else:
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

        dist_pickle = {}
        dist_pickle['has_bin_features'] = has_bin_features
        dist_pickle['has_hist_features'] = has_hist_features
        dist_pickle['has_hog_features'] = has_hog_features
        dist_pickle['colorspace'] = colorspace
        dist_pickle['spatial_size'] = spatial_size
        dist_pickle["hist_bins"] = hist_bins
        dist_pickle['hist_range'] = hist_range
        dist_pickle['orient'] = orient
        dist_pickle['pix_per_cell'] = pix_per_cell
        dist_pickle['cell_per_block'] = cell_per_block
        dist_pickle['hog_channel'] = hog_channel
        dist_pickle['svc'] = svc
        dist_pickle['scaler'] = X_scaler
        dist_pickle['reducer'] = X_reducer

        with open('../classifier/svc_reduced_bin_hist_hog.pkl','wb') as file:
            pkl.dump(dist_pickle, file)

if __name__ == '__main__':
    main()