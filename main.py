import numpy as np
#from load_data import preprocessing
#from model import create_model
#from ocr import train, predict
from segmentation import segmentImage
#from predict_box import predict_image
#import tensorflow as tf 
#import glob
import cv2
import imageio
from matplotlib.image import imread
from skimage.transform import resize


bounded_image_path = 'cropped_licenses_v5/cropped_license2.png'
bounded_plate = imread(bounded_image_path)

segmented_plates = segmentImage(bounded_plate)
print(len(segmented_plates))
nums = []
for plate_nums in segmented_plates:
    plate_nums = np.expand_dims(plate_nums, axis=0)
    all_nums = predict(resize(plate_nums, (1,28,28), anti_aliasing=True))
    nums.append(all_nums)
