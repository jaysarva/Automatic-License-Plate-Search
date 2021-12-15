import numpy as np
#from load_data import preprocessing
#from model import create_model
#from ocr import train, predict
import argparse
from preprocess_v2 import preprocessing
from model import create_model
from predict_box import calculate_rough_accuracy
from ocr import train, predict
from segmentation import segmentImage
from matplotlib.image import imread
from skimage.transform import resize

parser = argparse.ArgumentParser()
parser.add_argument('--t1', '--train-VGG-model', action="store_true")
parser.add_argument('--t2', '--train-OCR-model', action="store_true")
args = parser.parse_args()

preprocessing(training_size=0.9)

if args.t1:
    file = open("train_data/boundingbox.csv")
    rows = np.loadtxt(file, delimiter=",")
    create_model(rows)

calculate_rough_accuracy()

if args.t2:
    train()

bounded_image_path = 'cropped_licenses_v5/cropped_license23.png'
bounded_plate = imread(bounded_image_path)

segmented_plates = segmentImage(bounded_plate)
print(len(segmented_plates))
nums = []
for plate_nums in segmented_plates:
    plate_nums = np.expand_dims(plate_nums, axis=0)
    all_nums = predict(resize(plate_nums, (1,28,28), anti_aliasing=True))
    nums.append(all_nums)
