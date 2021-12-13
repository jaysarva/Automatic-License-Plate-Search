import numpy as np
from skimage.io import imread, imsave
import glob
import os
import cv2

path = "preprocessed_data"
isExist = os.path.exists(path)
if not isExist:
    os.makedirs(path)


img_dir = "raw_data/dataset1"
data_dir = os.path.join(img_dir, '*.png')
files = glob.glob(data_dir)
images = []
kernel_size = 3
ddepth = cv2.CV_16S
i = 0
for f in files:
    # Import data
    img = cv2.imread(f, cv2.IMREAD_COLOR)
    # Blur image
    # img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Sobel Edge Detection
    grad_x = cv2.Sobel(img, ddepth=ddepth, dx=1, dy=0, ksize=kernel_size)
    grad_y = cv2.Sobel(img, ddepth=ddepth, dx=0, dy=1, ksize=kernel_size)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    # Export image
    i += 1
    file_name = "preprocessed_data/img" + str(i) + ".png"
    imsave(file_name, grad)
    images.append(grad)
images = np.array(images)


