import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from skimage.feature import hog
from readImageDataset import read_img_path


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        # Use skimage.hog() to get both features and a visualization
        features, hog_image = hog(img, orientations=orient, 
        pixels_per_cell=(pix_per_cell, pix_per_cell), 
        cells_per_block=(cell_per_block, cell_per_block), visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:
        # Use skimage.hog() to get features only
        features = hog(img, orientations=orient, 
        pixels_per_cell=(pix_per_cell, pix_per_cell), 
        cells_per_block=(cell_per_block, cell_per_block), visualise=vis, feature_vector=feature_vec)
        return features

def main():
    (cars, notcars) = read_img_path()
    # Generate a random index to look at a car image
    carind = 13 #np.random.randint(0, len(notcars))
    notcarind = 13

    # COLOR_RGB2HSV
    # COLOR_RGB2LUV
    # COLOR_RGB2HLS
    # COLOR_RGB2YUV
    # COLOR_RGB2YCrCb

    # Read in the image
    carimage = mpimg.imread(cars[carind])
    carconvert = cv2.cvtColor(carimage, cv2.COLOR_RGB2YCrCb) #np.copy(carimage) #
    notcarimage = mpimg.imread(notcars[notcarind])
    notcarconvert = cv2.cvtColor(notcarimage, cv2.COLOR_RGB2YCrCb) #np.copy(notcarimage) #
    # Define HOG parameters
    orient = 9
    pix_per_cell = 12
    cell_per_block = 2
    # Call our function with vis=True to see an image output



    # Plot the examples
    fig, axes = plt.subplots(3, 4, figsize=(10,8))
    # fig.set_size_inches(28,4)
    for i in range(3):
        cargray = carconvert[:,:,i]
        carfeatures, carhog_image = get_hog_features(cargray, orient, 
                            pix_per_cell, cell_per_block, 
                            vis=True, feature_vec=False)
        axes[i, 0].imshow(cargray, cmap='gray')
        axes[i, 0].set_title('Car Channel {}'.format(i), fontsize = 12)
        axes[i, 1].imshow(carhog_image, cmap='gray')
        axes[i, 1].set_title('Car HOG Channel {}'.format(i), fontsize = 12)

        notcargray = notcarconvert[:,:,i]
        notcarfeatures, notcarhog_image = get_hog_features(notcargray, orient, 
                            pix_per_cell, cell_per_block, 
                            vis=True, feature_vec=False)
        axes[i, 2].imshow(notcargray, cmap='gray')
        axes[i, 2].set_title('Not Car Channel {}'.format(i), fontsize = 12)
        axes[i, 3].imshow(notcarhog_image, cmap='gray')
        axes[i, 3].set_title('Not Car HOG Channel {}'.format(i), fontsize = 12)
    plt.subplots_adjust(left=0.05, right =0.95, top =0.95, bottom =0.05)
    plt.savefig('../output_images/YCrCb_Hog.png')

if __name__ == '__main__':
    main()

