import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import pickle
from sklearn.cross_validation import train_test_split
from get_hog import get_hog_features
from extract_features import (bin_spatial, color_hist, convert_color, extract_single_feature, extract_features)
from scipy.ndimage.measurements import label

# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    img_size = [img.shape[1], img.shape[0]]
    if x_start_stop[0] == None or x_start_stop[1] == None:
        x_start_stop = [0, img_size[0]]
    if y_start_stop[0] == None or y_start_stop[1] == None:
        y_start_stop = [0, img_size[1]]

    # Compute the span of the region to be searched  
    regionX = x_start_stop[1] - x_start_stop[0]
    regionY = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    numPixX = np.int(xy_window[0]*(1-xy_overlap[0]))
    numPixY = np.int(xy_window[1]*(1-xy_overlap[1]))
    
    bufferX = np.int(xy_window[0]*xy_overlap[0])
    bufferY = np.int(xy_window[1]*xy_overlap[1])
    
    # Compute the number of windows in x/y
    numWinX = np.int((regionX - bufferX)/numPixX)
    numWinY = np.int((regionY - bufferY)/numPixY)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    #     Note: you could vectorize this step, but in practice
    #     you'll be considering windows one by one with your
    #     classifier, so looping makes sense
        # Calculate each window position
        # Append window position to list
    # Return the list of windows
    for j in range(numWinY):
        for i in range(numWinX):
            window_list.append(((x_start_stop[0]+i*numPixX, y_start_stop[0]+j*numPixY),(x_start_stop[0]+i*numPixX+xy_window[0], y_start_stop[0]+j*numPixY+xy_window[1])))
    return window_list

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, svc, scaler, reducer, has_bin_features, has_hist_features, has_hog_features, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256), orient=9, pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = extract_single_feature(test_img, has_bin_features, has_hist_features, has_hog_features, cspace=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, hist_range=hist_range, 
                        orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
        #5) Scale extracted features to be fed to classifier
        # print(np.array(features).reshape(1, -1).shape)
        # print(scaler.mean_.shape)

        test_features = scaler.transform(np.array(features).reshape(1, -1))
        test_features = reducer.transform(test_features)
        #6) Predict using your classifier
        prediction = svc.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_boxes(img, xstart, xstop, ystart, ystop, scale, svc, X_scaler, X_reducer, has_bin_features, has_hist_features, 
            has_hog_features, cells_per_step = 2, color_space='RGB', spatial_size=(32, 32), hist_bins=32, 
            hist_range=(0, 256), orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0):

    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,xstart:xstop]
    ctrans_tosearch = convert_color(img_tosearch, color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
    hogs = []
    if has_hog_features == True:
        for i in range(3):
            hogs.append(get_hog_features(ctrans_tosearch[:,:,i], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False))

    # Define blocks and steps as above
    nxblocks = (ctrans_tosearch.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ctrans_tosearch.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    bboxes = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
            # Extract HOG for this patch
            bin_features = []
            hist_features = []
            hog_features = []

            if has_bin_features == True or has_hist_features == True:

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))                
                if has_bin_features == True:
                    bin_features = bin_spatial(subimg, size = spatial_size)
                    # print('bin features:{}'.format(bin_features.shape))
                if has_hist_features == True:
                    hist_features = color_hist(subimg, nbins=hist_bins, bins_range=hist_range)
                    # print('hist features:{}'.format(hist_features.shape))

            if has_hog_features == True:
                # Call get_hog_features() with vis=False, feature_vec=True
                if hog_channel == 'ALL':
                    for i in range(3):
                        hog_features.extend(hogs[i][ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel())      
                else:
                    hog_features = hogs[hog_channel][ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                # print('hog features:{}'.format(hog_features.shape))
            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((bin_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_features = X_reducer.transform(test_features)
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                bboxes.append(((xbox_left+xstart, ytop_draw+ystart),(xbox_left+xstart+win_draw,ytop_draw+win_draw+ystart)))
                # cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                
    return bboxes


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels, color = (255,0,0), box_threshold = 50):
    imcpy = np.copy(img)
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        minx = np.min(nonzerox)
        miny = np.min(nonzeroy)
        maxx = np.max(nonzerox)
        maxy = np.max(nonzeroy)
        if maxx - minx > box_threshold and maxy - miny > box_threshold:

            bbox = ((minx, miny), (maxx, maxy))
            # Draw the box on the image
            cv2.rectangle(imcpy, bbox[0], bbox[1], color, 6)
    # Return the image
    return imcpy

def main():
    dist_pickle = pickle.load(open("../classifier/svc_reduced_bin_hist_hog.pkl", "rb" ))
    has_bin_features = dist_pickle['has_bin_features']
    has_hist_features = dist_pickle['has_hist_features']
    has_hog_features = dist_pickle['has_hog_features']
    colorspace = dist_pickle['colorspace']
    spatial_size = dist_pickle['spatial_size']
    hist_bins = dist_pickle["hist_bins"]
    hist_range = dist_pickle['hist_range']
    orient = dist_pickle['orient']
    pix_per_cell = dist_pickle['pix_per_cell']
    cell_per_block = dist_pickle['cell_per_block']
    hog_channel = dist_pickle['hog_channel']
    svc = dist_pickle['svc']
    X_scaler = dist_pickle['scaler']
    X_reducer = dist_pickle['reducer']
    

    # image = mpimg.imread('../../Vehicle-Detection-Dataset/orig_frames/frame0000.jpg')[:,:,0:3]
    image = mpimg.imread('../test_images/test5.jpg')


    draw_image = np.copy(image)
    draw_image_heat = np.copy(image)
    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    
    image = image.astype(np.float32)/255


    # # below is sliding windows. Slow
    # xy_windows = [(64, 64), (96, 96), (116, 116)] #[(64, 64),(128, 128),(140, 140)]
    # threshold = 2
    # xy_overlap = (0.75, 0.75)
    # y_start_stop = [400, 650] # Min and max in y to search in slide_window()
    # x_start_stop = [550, 1280]
    # hot_windows_list = []
    # t = time.time()
    # for xy_window in xy_windows:
    #     windows = slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop, 
    #                         xy_window=xy_window, xy_overlap=xy_overlap)

    #     hot_windows = search_windows(image, windows, svc, X_scaler, X_reducer, has_bin_features, has_hist_features, has_hog_features, 
    #                     color_space=colorspace, spatial_size=spatial_size, hist_bins=hist_bins, hist_range=hist_range, 
    #                     orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
    #     hot_windows_list.extend(hot_windows)
    # draw_image = draw_boxes(draw_image, hot_windows_list, color=(0, 0, 255), thick=6)                    
    # t2 = time.time()
    # print('{} seconds to find the boxes'.format(round(t2-t, 2)))

    # heat = np.zeros_like(image[:,:,0]).astype(np.float)
    # # Add heat to each box in box list
    # heat = add_heat(heat,hot_windows_list)        
    # # Apply threshold to help remove false positives
    # heat = apply_threshold(heat,threshold)
    # # Visualize the heatmap when displaying    
    # heatmap = np.clip(heat, 0, 255)
    # # Find final boxes from heatmap using label function
    # labels = label(heatmap)
    # draw_image_heat = draw_labeled_bboxes(draw_image_heat, labels, color=(255,0,0))




    # Below are the tunable parameters
    ystart = 400
    ystop = 650
    xstart = 620
    xstop = 1280
    scales = [1, 1.5, 1.8]
    cells_per_step = 2
    threshold = 2

    boxes_list = []

    t = time.time()
    for scale in scales:
        boxes = find_boxes(image, xstart, xstop, ystart, ystop, scale, svc, X_scaler, X_reducer, has_bin_features, has_hist_features, has_hog_features, 
            cells_per_step, color_space=colorspace, spatial_size=spatial_size, hist_bins=hist_bins, hist_range=hist_range, 
                orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
        boxes_list.extend(boxes)
    draw_image = draw_boxes(draw_image, boxes_list, color=(0, 0, 255), thick=6)
    t2 = time.time()
    print('{} seconds to find the boxes'.format(round(t2-t, 2)))

    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat,boxes_list)        
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,threshold)
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_image_heat = draw_labeled_bboxes(draw_image_heat, labels, color=(255,0,0))

    fig = plt.figure()
    plt.subplot(221)
    plt.imshow(image)
    plt.title('Car Positions')
    plt.subplot(222)
    plt.imshow(draw_image)
    plt.title('Boxes')
    plt.subplot(223)
    plt.imshow(heatmap, cmap='hot')
    plt.title('Heat Map')
    plt.subplot(224)
    plt.imshow(draw_image_heat)
    plt.title('Box draw back')
    fig.tight_layout()
    plt.savefig('../output_images/bounding_box_pipeline_eg5.png')
    # plt.show()

if __name__ == '__main__':
    main()



