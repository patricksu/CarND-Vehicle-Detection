## Project Writeup

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/example_car_notcar.png
[image2]: ./output_images/YCrCb_Hog.png
[image3]: ./output_images/bounding_box_pipeline_eg1.png
[image4]: ./output_images/bounding_box_pipeline_eg2.png
[image5]: ./output_images/bounding_box_pipeline_eg3.png
[image6]: ./output_images/bounding_box_pipeline_eg4.png
[image7]: ./output_images/bounding_box_pipeline_eg5.png
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the files "readImageDataset.py", "get_hog.py".  

I started by reading in all the `vehicle` and `non-vehicle` images, in lines 10 to 18 in the file "readImageDataset.py".  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like. The Bin, histogram, and hog feature extraction code is in lines 75 to 92 in the file "extract_features.py"

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. I found that the set of "orient=8, pixels_per_cell=(8,8), cells_per_block=(2,2)" is pretty good. Then I tried different color spaces using a linear SVM classifier. The YCrCb is the best colorspace.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using HOG features, in line 91 to 105 of file "classify.py". I used all the images in the GTI and KITTI datasets. The test accuracy with different color spaces are shown below. 
RGB:    0.9625
HLS:    0.9895
HSV:    0.9945
LUV:    0.9915
YUV:    0.9930
YCrCb:  0.9955

Then I tested different combinations of the features as shown below. It seems like HOG is the best, and adding Bin and/or Histogram features are not adding much benefits to HOG. 
Bin features:             0.9205
Histogram features:       0.488
Hog+Bin features:         0.997
Hog+Histogram features:   0.994
Hog+Bin+Histogram:        0.994

I also used GridSearchCV to find the best parameters for the linear SVC classifier. It seems like the test score is not sensitive at all to the C value. So I simply use C value 1.

The final features include Bin, histogram, and hog. I reduced the dimension from over 8000 to 200 using PCA without jeopardizing the model performance. 

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Lines from 128 to 169 in the file "box_search.py" implement the sliding window search. For each scale, it generates the hog features for the whole image (only once), and searches all the windows for vehicles. It is based on subsampling approach from the Udacity class.
Using more scales and/or smaller scales and/or bigger overlap windows will generate more bounding boxes, but at the same time more false positives. Thus the heatmap threshold value needs to be higher to eliminate the false positives. On the other hand, using less scales and/or smaller overlap windows will produce less bounding boxes, less false-positives, but doing this might miss some vehicles. Thus picking the best parameters set is a trade-off between precision, recall, and speed. I used images with different vehicle sizes and different lighting conditions as test cases, and decided to use the following parameters: scales = [1, 1.5, 1.8], cells_per_step = 2, heatmap threshold = 1. Having scales smaller than 1 generates a lot of false-positives, thus the minimum scale I used is 1. Having scale 1.8 helps detecting close-by vehicles that are bigger. Using three different scales generate enough bounding boxes that cover the whole vehicle. The cells_per_step is chosen as 2, since having 1 will generate a lot of overlapping boxes, and makes it hard to eliminate false-positive boxes, while using 3 could not generate enough boxes, and miss vehicles. 

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The figures below are showing how the pipeline works. It takes a raw image, use a classifier to search by sliding windows and give the vehicle boxes, then it draws a heatmap, and outputs the final bounding box. 

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. The filters I use for the false-positives include restricting search to the right-down corner of the frames (vehicles appear in this area only). The heatmap threshold is critical for filtering out the false-positives. 

The bound boxes, heatmap, and resulting bounding boxes are shown in the figures above. 

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

A major problem I faced is that the classifier is not quite effective with incomplete vehicles, i.e. when half of the vehicles are in the image. This is because the training dataset has only images of full vehicles, thus it does not generalize to incomplete vehicles. Doing data augmentation (cutting the raw traing images to half or so) will help the classifier. 

