from tensorflow.keras.models import load_model
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import os
import shutil
from PIL import Image
import cv2

path, dirs, files = next(os.walk("test_data/data"))
n = len(files)
my_model = load_model("my_model")

def predict_image(image_path, model):
    image = imread(image_path)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    
    preds = model.predict(image)[0]
    (startX, startY, endX, endY) = preds
    # (startY, startX, endY, endX) = preds
    
    if startX > endX:
        startX2 = startX
        startX = endX
        endX = startX2
    if startY > endY:
        startY2 = startY
        startY = endY
        endY = startY2
        
    w = 224
    h = 224
    
    startX = int(startX * w)
    startY = int(startY * h)
    endX = int(endX * w)
    endY = int(endY * h)
    
    if startX == endX :
        startX = startX - 1
        endX = endX + 1
    if startY == endY:
        startY = startY - 1
        endY = endY + 1
    
    return [image_path, startX, startY, endX, endY]

def create_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
   
def calculate_rough_accuracy():
    file = open("test_data/boundingbox.csv")
    rows = np.loadtxt(file, delimiter=",")
    #print(rows)
    total = 0
    for i in range(n):
        image_path = "test_data/data/Cars" + str(i) + ".png"
        [image_path, startX, startY, endX, endY] = predict_image(image_path,my_model)
        row = rows[i]
        
        w = row[3] - row[1]
        h = row[4] - row[2]
        
        if (abs(startX - row[1]) <= 0.3*w) and (abs(endX - row[3]) <= 0.3*w) and (abs(startY - row[2]) <= 0.3*h) and (abs(endY - row[4]) <= 0.3*h):
            total += 1
        print(i)
        image = imread(image_path)
        # cropped = image[startY:endY,startX:endX,:]
        cropped = image[startX:endX,startY:endY,:]
        cv2.rectangle(image, (startY, startX), (endY, endX),(255, 255, 0), 2)
        
        im = Image.fromarray(image)
        create_directory("licenses")
        im.save("licenses/license"+str(i)+".png")

        cropped_im = Image.fromarray(cropped)
        create_directory("cropped_licenses")
        cropped_im.save("cropped_licenses/cropped_license"+str(i)+".png")
    
    print("Accuracy = " + str(total / n))
                 
calculate_rough_accuracy()
