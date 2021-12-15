import numpy as np
import argparse
from preprocess_v2 import preprocessing
from model import create_model
from predict_box import calculate_rough_accuracy
from ocr import train, predict
from segmentation import segmentImage
from matplotlib.image import imread
from skimage.transform import resize

parser = argparse.ArgumentParser()
parser.add_argument('--t', '--train-model', action="store_true")
args = parser.parse_args()

preprocessing(training_size=0.9)

if args.t:
    file = open("train_data/boundingbox.csv")
    rows = np.loadtxt(file, delimiter=",")
    create_model(rows)

calculate_rough_accuracy()

if args.t:
    train()

bounded_image_path = 'cropped_licenses_v5/cropped_license2.png'
bounded_plate = imread(bounded_image_path)

segmented_plates = segmentImage(bounded_plate)
print(len(segmented_plates))
#print(segmented_plates[0])
#cur_img = segmented_plates[0]
#frame_normed = 255 * (cur_img - cur_img.min()) / (cur_img.max() - cur_img.min())
#frame_normed = np.array(frame_normed, np.int)

#cv2.imwrite("savedImage.png", frame_normed)
nums = []
for plate_nums in segmented_plates:
    plate_nums = np.expand_dims(plate_nums, axis=0)
    # print(plate_nums.shape)
    all_nums = predict(resize(plate_nums, (1,28,28), anti_aliasing=True))
    # print(all_nums)
    nums.append(all_nums)
print(nums)
#for index, row in enumerate(license_plates_bounding_points):
#    image = cv2.imread(license_plates_bounding_points[0])
#    segmented_plates = map(lambda subIm: segmentImage(image[subIm[1]:subIm[3],subIm[0]:subIm[2]), license_plates_bounding_points)

#    for plate_nums in segmented_plates:
#        all_nums = predict(np.array(plate_nums))

