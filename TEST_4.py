import struct
import numpy as np
import pandas as pd
from array import array
import matplotlib.pyplot as plt
from pandas import ExcelFile
import openpyxl
import numpy.linalg
import math

# if chage values in a matrix without for loop
T_6class = np.full((5, 5), -1)
v = np.ndarray([0,1,2,3,4])
print("T_6class:", T_6class)
#T_6class[:, v] = 1
print("T_6class:", T_6class)

# np.where
a_org = np.arange(9).reshape((3, 3))
a_new = np.where(a_org < 4, -1, a_org)
print("a_new:", a_new)

# get index np.where
print(np.where(a_org < 4))
print(np.column_stack(np.where(a_org < 4)))


b = np.array([(1.5, 2, 3), (4, 5, 6)], dtype=float)
b=b[[1, 0, 1, 0]][:, [0, 1, 2, 0]]
print("b:", b)

# get Max value index
v = numpy.array([1, 2, 3, 4])
v_index = np.argmax(v)
print("v_index:", v_index)

# get Max value index in multi arrays
v2 = numpy.array([[1, 2, 3], [5, 6, 8], [7, 10, 9]])
print("np.argmax(v2):", np.argmax(v2))
print("np.argmax(v2, axis=1):", np.argmax(v2, axis=1))

print("v2 sum:", v2[:, 0].sum())

ppv = np.empty((len(v2), 1))

for i in range(len(v2[0, :])):
    ppv[i] =  v2[i, i]/v2[:, i].sum()
    print("v2 {}:{}".format(i, ppv[i]))

print("np.argmax(ppv):", np.argmax(ppv))
print("np.argmin(ppv):", np.argmin(ppv))

print("np.round(3.14):", np.round(3.1353, decimals=2))

# check array
array_b = np.array([[1,2,4],[2,2,3],[4,3,3],[3,3,1]])
print('array_b[0]->{}'.format(array_b[3]))

# get pseudoinverse
mat_org = np.array([[1,0,1],[0,1,0]])
mat_pinv = np.linalg.pinv(mat_org)
print('mat_pinv->{}'.format(mat_pinv))

mat_org = np.array([[1,1],[1,-1]])
mat_inv = np.linalg.inv(mat_org)
print('mat_inv->{}'.format(mat_inv))

print('exp(2)->{}'.format(np.exp(2)))
