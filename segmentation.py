import cv2
import skimage.filters
import numpy as np
from skimage import color

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def segmentImage(image):

    # maybe need some preprocessing to rotate the image to be oriented horizontally... 
    image=image[2:image.shape[0] - 2, 2:image.shape[1] - 2]
    #grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale = color.rgb2gray(image)




    (thresh, binarized) = cv2.threshold(grayscale, 0.5, 0.9, cv2.THRESH_BINARY)
    
    #new_path = "binarized.png"
    #cv2.imwrite(new_path,image)

    blurred = skimage.filters.gaussian(binarized)
    
    frame_normed = 255 * (blurred - blurred.min()) / (blurred.max() - blurred.min())
    frame_normed = np.array(frame_normed, np.int)

    cv2.imwrite("savedImage.png", frame_normed)
    # find histograms of horizontal lines, and if above certain threshold, determine upper/lower bounds of characters
    vertical_threshold = 0.5
    row_histograms = np.average(blurred,axis=1)
    top_border, bottom_border = np.nonzero(row_histograms > vertical_threshold)[0][[0,-1]]
    top_border -= 2
    if (top_border < 0):
        top_border = 0
    bottom_border += 2
    # find histograms of vertical lines, if above threshold you find regions separating individual characters. 
    
    horizontal_threshold = 0.45
    column_histograms = np.average(blurred, axis=0)
    
    vertical_boundaries = np.nonzero(column_histograms > horizontal_threshold)[0]
    # separate regions into individual sections and return. 
    subImages = []
    for i in range(len(vertical_boundaries) - 1):
        if (vertical_boundaries[i+1] - vertical_boundaries[i]) <= 5:
            continue
        print(vertical_boundaries[i], vertical_boundaries[i+1], top_border, bottom_border)
        subImage = blurred[:, vertical_boundaries[i]: vertical_boundaries[i+1]]
        subImages.append(subImage)


    #cur_img = subImages[3]
    #frame_normed = 255 * (cur_img - cur_img.min()) / (cur_img.max() - cur_img.min())
    #frame_normed = np.array(frame_normed, np.int)

    #cv2.imwrite("savedImage.png", frame_normed)

    return subImages