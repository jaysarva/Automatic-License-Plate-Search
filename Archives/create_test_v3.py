import cv2

import numpy as np

#236, 160 take 100
for i in range(100, 150):
    path = "preprocessed_data/resized_images/Cars" + str(i) + ".png"
    image = cv2.imread(path)
    new_path = "test_data_v4/data/Cars" + str(i - 100) + ".png"
    cv2.imwrite(new_path, image)

file2 = open("boundingbox.csv")
rows2 = np.loadtxt(file2, delimiter=",")

for i in range(100, 150):
    rows2[i][0] = int(rows2[i][0]) - 100

np.savetxt("test_data_v4/boundingbox.csv", rows2[100:150], delimiter=',', fmt='%d')