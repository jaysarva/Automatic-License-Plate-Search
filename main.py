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

#file = open("boundingbox.csv")
#rows = np.loadtxt(file, delimiter=",")
# use saved weights for ocr model
#license_plate_detect = tf.keras.models.load_model("my_model.h5")

#images = glob.glob("preprocessed_data/resized_images/*.png")[:10]

#license_plates_bounding_points = [predict_image(path, license_plate_detect) for path in images]

bounded_image_path = 'cropped_licenses_v5/cropped_license2.png'
bounded_plate = imread(bounded_image_path)
#print(bounded_plate)

segmented_plates = segmentImage(bounded_plate)
print(len(segmented_plates))
#imageio.imwrite('segmentedImage.jpeg', segmented_plates[0])
for plate_nums in segmented_plates:
    plate_nums = np.expand_dims(plate_nums, axis=0)
    print(plate_nums.shape)
    all_nums = predict(np.array(plate_nums))
print(all_nums)
#for index, row in enumerate(license_plates_bounding_points):
#    image = cv2.imread(license_plates_bounding_points[0])
#    segmented_plates = map(lambda subIm: segmentImage(image[subIm[1]:subIm[3],subIm[0]:subIm[2]), license_plates_bounding_points)

#    for plate_nums in segmented_plates:
#        all_nums = predict(np.array(plate_nums))

