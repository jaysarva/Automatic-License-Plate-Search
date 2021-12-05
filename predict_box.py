from tensorflow.keras.models import load_model
import numpy as np
from skimage.io import imread

from PIL import Image

my_model = load_model("my_model")

def predict_image(image_path, model):
    image = imread(image_path)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    
    preds = model.predict(image)[0]
    (startX, startY, endX, endY) = preds
    
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
    return [image_path, startX, startY, endX, endY]


   
def calculate_rough_accuracy():
    file = open("boundingbox.csv")
    rows = np.loadtxt(file, delimiter=",")
    #print(rows)
    total = 0
    for i in range(400):
        image_path = "preprocessed_data/dataset1/resized_images/Cars" + str(i) + ".png"
        [a,b,c,d,e] = predict_image(image_path,my_model)
        row = rows[i]
        
        w = row[3] - row[1]
        h = row[4] - row[2]
        
        if (abs(b - row[1]) <= 0.2*w) and (abs(d - row[3]) <= 0.2*w) and (abs(c - row[2]) <= 0.2*h) and (abs(e - row[4]) <= 0.2*h):
            total += 1
        print(i)
    print("Accuracy = " + str(total / 400))
              

   
   
for i in range(433):
    path = "preprocessed_data/dataset1/resized_images/Cars" + str(i) + ".png"
    [x1,x2,x3,x4,x5] = predict_image(path, my_model)
    image = imread(path)
    new_image = image[x2:x4, x3:x5, :]
    im = Image.fromarray(new_image)
    im.save("licenses/license"+str(i)+".png")

    