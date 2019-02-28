############################################################
#
# Assingment 4
#
#
############################################################
import struct
import numpy as np
from array import array
import matplotlib.pyplot as plt
from pandas import ExcelFile
import openpyxl
import numpy.linalg
import math
import pandas as pd

X = np.array(([1, 0, 1], [0, 1, 0]))
w = np.array(([0, 1]))
b = np.array(([-1, 2, 0]))

X_pinv = np.linalg.pinv(X)

print(X_pinv)


bp = np.matmul(w, X)
w1 = np.matmul(b, X_pinv)
bp2 = np.matmul(w1, X)
print(b)
print(bp)
print(w1)
print(bp2)

