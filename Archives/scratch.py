import csv
import numpy as np
data = np.array([[1,1,1]])

a = np.array([[1,2,3]])
b = np.array([[4,5,6]])
data = np.append(data,a,0)
data = np.append(data,b,0)
#data = np.concatenate((data,b),axis=1)
print(data)

# file = open("boundingbox.csv")
# rows = np.loadtxt(file, delimiter=",")

# print(rows)
