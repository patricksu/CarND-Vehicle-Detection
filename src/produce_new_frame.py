import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from extract_features import (bin_spatial, color_hist, convert_color, extract_single_feature, extract_features)
from scipy.ndimage.measurements import label
from box_search import *


def pipeline(image, xstart, xstop, ystart, ystop, box_threshold, scales, svc, X_scaler, X_reducer, has_bin_features, has_hist_features, has_hog_features, 
            cells_per_step, threshold, color_space, spatial_size, hist_bins, hist_range, 
                orient, pix_per_cell, cell_per_block, hog_channel):
    draw_image = np.copy(image)
    draw_image_heat = np.copy(image)
    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    image = image.astype(np.float32)/255
    # Below are the tunable parameters
    boxes_list = []
    for scale in scales:
        boxes = find_boxes(image, xstart, xstop, ystart, ystop, scale, svc, X_scaler, X_reducer, has_bin_features, has_hist_features, has_hog_features, 
            cells_per_step, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins, hist_range=hist_range, 
                orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
        boxes_list.extend(boxes)
    draw_image = draw_boxes(draw_image, boxes_list, color=(0, 0, 255), thick=6)

    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat,boxes_list)        
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,threshold)
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_image_heat = draw_labeled_bboxes(draw_image_heat, labels, (255,0,0), box_threshold)
    return draw_image_heat



def generate_orig_images():
    vidcap = cv2.VideoCapture('../project_video.mp4')
    vidcap.set(cv2.CAP_PROP_POS_MSEC, 38000)
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
        success,image = vidcap.read()
        if count % 24 == 0:            
            image = image[:,:,::-1]
            print('Read a new frame: ', success)
            mpimg.imsave("../../Vehicle-Detection-Dataset/orig_frames/frame" + "0" * (4-len(str(count))) + "{:d}.jpg".format(count), image)     # save frame as JPEG file
        count += 1


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
       
    ystart = 400
    ystop = 650
    xstart = 620
    xstop = 1280
    scales = [1, 1.5]
    cells_per_step = 2
    threshold = 2
    box_threshold = 60

    # INPUTVIDEO = '../test_video.mp4'  
    INPUTVIDEO =  '../project_video.mp4'
    # OUTPUTPATH = "../../Vehicle-Detection-Dataset/test_video_output_frames/"
    OUTPUTPATH = "../../Vehicle-Detection-Dataset/project_video_output_frames/"

    vidcap = cv2.VideoCapture(INPUTVIDEO)
    # vidcap.set(cv2.CAP_PROP_POS_MSEC, 6000)
    count = 0
    success = True
    while success:
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        if count % 1 == 0:
            image = image[:,:,::-1]
            new_img = pipeline(image, xstart, xstop, ystart, ystop, box_threshold, scales, svc, X_scaler, X_reducer, has_bin_features, has_hist_features, has_hog_features, 
                cells_per_step, threshold, color_space=colorspace, spatial_size=spatial_size, hist_bins=hist_bins, hist_range=hist_range, 
                orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
            mpimg.imsave(OUTPUTPATH + "frame" + "0" * (4-len(str(count))) + "{:d}.jpg".format(count), new_img)     # save frame as JPEG file
            
        count += 1



if __name__ == '__main__':
    main()
