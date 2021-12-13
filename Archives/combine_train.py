import cv2

import numpy as np

#236, 160 take 100
for i in range(100):
    path = "preprocessed_data/resized_images/Cars" + str(i) + ".png"
    image = cv2.imread(path)
    new_path = "train_data_v3/data/Cars" + str(i + 236) + ".png"
    cv2.imwrite(new_path, image)

file1 = open("test_data_v2/boundingbox.csv") #size = 235
file2 = open("boundingbox.csv")
rows1 = np.loadtxt(file1, delimiter=",")
rows2 = np.loadtxt(file2, delimiter=",")

for i in range(100):
    rows2[i][0] = int(rows2[i][0]) + 236
# rows_out = np.concatenate(rows1,rows2[:101,:], axis = 0)
rows_out = np.vstack((rows1, rows2[:100]))

np.savetxt("train_data_v3/boundingbox_net.csv", rows_out, delimiter=',', fmt='%d')