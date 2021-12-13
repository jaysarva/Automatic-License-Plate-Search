import numpy as np
from load_data import preprocessing
from model import create_model
from ocr import train, predict
from segmentation import segmentImage
from predict_box import predict_image
import tensorflow as tf 
import glob
import cv2

#file = open("boundingbox.csv")
#rows = np.loadtxt(file, delimiter=",")
# use saved weights for ocr model
#license_plate_detect = tf.keras.models.load_model("my_model.h5")

#images = glob.glob("preprocessed_data/resized_images/*.png")[:10]

#license_plates_bounding_points = [predict_image(path, license_plate_detect) for path in images]

bounded_image_path = ???
bounded_plate = imread(bounded_image_path)
segmented_plates = segmentImage(bounded_plate)
for plate_nums in segmented_plates:
    all_nums = predict(np.array(plate_nums))
print(all_nums)
#for index, row in enumerate(license_plates_bounding_points):
#    image = cv2.imread(license_plates_bounding_points[0])
#    segmented_plates = map(lambda subIm: segmentImage(image[subIm[1]:subIm[3],subIm[0]:subIm[2]), license_plates_bounding_points)

#    for plate_nums in segmented_plates:
#        all_nums = predict(np.array(plate_nums))

