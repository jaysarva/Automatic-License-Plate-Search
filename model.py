from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from skimage.io import imread

from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
import numpy as np
import io
import csv


def create_model(rows):
    img_size = 224
    data = np.empty((0,img_size,img_size,3))
    targets = np.empty((0,4))
    filenames = np.array([])

    for row in rows:
        
        
        (filename, startX, startY, endX, endY) = row

        image = imread("preprocessed_data/resized_images/Cars" + str(int(filename)) + ".png")
        
        h = img_size
        w = img_size
        #print(image)
        startX = float(startX) / w
        startY = float(startY) / h
        endX = float(endX) / w
        endY = float(endY) / h
    
        data = np.append(data, np.array([image]), axis=0)
        targets = np.append(targets, np.array([[startX, startY, endX, endY]]), axis=0)
        #targets.append(np.array(startX, startY, endX, endY))
        #filenames= np.append(filenames,filename)
        print(filename)
    
    print("DONE 1")

    data = data / 255.0
    
    print(np.shape(data))
    split = train_test_split(data, targets, test_size=0.15, random_state=42)
    
    print("DONE 2")

    (trainImages, testImages) = split[:2]
    (trainTargets, testTargets) = split[2:4]
#    (trainFilenames, testFilenames) = split[4:]

    vgg = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
# freeze all VGG layers so they will *not* be updated during the
# training process
    vgg.trainable = False
# flatten the max-pooling output of VGG
    flatten = vgg.output
    flatten = Flatten()(flatten)
# construct a fully-connected layer header to output the predicted
# bounding box coordinates
    bboxHead = Dense(128, activation="relu")(flatten)
    bboxHead = Dense(64, activation="relu")(bboxHead)
    bboxHead = Dense(32, activation="relu")(bboxHead)
    bboxHead = Dense(4, activation="sigmoid")(bboxHead)
# construct the model we will fine-tune for bounding box regression
    model = Model(inputs=vgg.input, outputs=bboxHead)


    opt = Adam(lr= 1e-4)
    model.compile(loss="mse", optimizer=opt)
    print(model.summary())
# train the network for bounding box regression
    print("[INFO] training bounding box regressor...")
    H = model.fit(trainImages, trainTargets, validation_data=(testImages, testTargets), batch_size= 32, epochs= 30, verbose=1)
    
    model.save("my_model", save_format="h5")
    print("Finish Saving the model")




file = open("boundingbox.csv")
rows = np.loadtxt(file, delimiter=",")

create_model(rows)
#print(rows)

 

