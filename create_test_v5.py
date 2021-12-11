import numpy as np
import cv2

l = [0,5,8,9,13,14,15,21,22,23,24,28,29,31,32,35,38,40,41,42,44,45,49]
index = 0
for elm in l:
    path = "test_data_v4/data/Cars" + str(elm) + ".png"
    image = cv2.imread(path)
    new_path = "test_data_v5/data/Cars" + str(index) + ".png"
    cv2.imwrite(new_path,image)
    index += 1

file = open("test_data_v4/boundingbox.csv")
rows = np.loadtxt(file, delimiter=",")

rows_out = np.zeros((73,5))
index = 0
for i in l:
    rows_out[index] = rows[i]
    rows_out[index][0] = index
    index += 1 

file2 = open("train_data_v3/boundingbox_net.csv")
rows2 = np.loadtxt(file2, delimiter=",")

index = 23
for i in range(50,100):
    path = "train_data_v3/data/Cars" + str(i) + ".png"
    image = cv2.imread(path)
    new_path = "test_data_v5/data/Cars" + str(index) + ".png"
    cv2.imwrite(new_path,image)

    rows_out[index] = rows2[i]
    rows_out[index][0] = index
    index += 1


np.savetxt("test_data_v5/boundingbox.csv", rows_out, delimiter=',', fmt='%d')





