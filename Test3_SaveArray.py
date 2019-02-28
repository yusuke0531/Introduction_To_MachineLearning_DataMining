

import struct
import numpy as np
from array import array
import matplotlib.pyplot as plt
from pandas import ExcelFile
import openpyxl
import numpy.linalg
import math

def load_mnist(dataset="training", selectedDigits=range(10),
               path=r'C:\Users\uskya\OneDrive\document_usk\UCSC\02_SecondQuater\Introduction_To_MachineLearning_DataMining\Assignment\3_Assignment'):
    # Check training/testing specification. Must be "training" (default) or "testing"
    if dataset == "training":
        fname_digits = path + '\\' + 'train-images.idx3-ubyte'
        fname_labels = path + '\\' + 'train-labels.idx1-ubyte'
    elif dataset == "testing":
        fname_digits = path + '\\' + 't10k-images.idx3-ubyte'
        fname_labels = path + '\\' + 't10k-labels.idx1-ubyte'
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Import digits data, Open read binary mode
    digitsFileObject = open(fname_digits, 'rb')

    # FileObject.read(16) read 16 bytes as binary data
    # bytes object (in binary mode). size is an optional numeric argument
    # ">IIII" means > Big Endian, I is unsigned int (4bytes)
    #
    magic_nr, size, rows, cols = struct.unpack(">IIII", digitsFileObject.read(16))

    # read remaining all data. array.array("B",data) "B" means (unsigned char in C) int in Python 1 byte
    digitsData = array("B", digitsFileObject.read())
    digitsFileObject.close()
    print("digitsData SIZE_>{}".format(len(digitsData)))
    print("magic_nr, size, rows, cols->{},{},{},{}".format(magic_nr, size, rows, cols))

    # Import label data
    labelsFileObject = open(fname_labels, 'rb')
    magic_nr, size = struct.unpack(">II", labelsFileObject.read(8))

    labelsData = array("B", labelsFileObject.read())
    labelsFileObject.close()

    # Find indices of selected digits
    indices = [k for k in range(size) if labelsData[k] in selectedDigits]
    N = len(indices)

    # Create empty arrays for X and T
    X = np.zeros((N, rows * cols), dtype=np.uint8)
    T = np.zeros((N, 1), dtype=np.uint8)


    # Fill X from digitsdata
    # Fill T from labelsdata
    for i in range(N):
        # get digitsdata as array from data array
        X[i] = digitsData[indices[i] * rows * cols:(indices[i] + 1) * rows * cols]

        # get label data
        T[i] = labelsData[indices[i]]

    return X, T


######################################################
#
# Main
#
######################################################
path=r'C:\Users\uskya\OneDrive\document_usk\UCSC\02_SecondQuater\Introduction_To_MachineLearning_DataMining\Assignment\3_Assignment'
dataset_name = "training"
negative = 5
positive = 6

X_data, T_data = load_mnist(dataset=dataset_name, selectedDigits=[negative, positive])

np.save(path+"\\"+dataset_name+"_image", X_data)
np.save(path+"\\"+dataset_name+"_label", T_data)

X_data_test, T_data_test = load_mnist(dataset="testing", selectedDigits=[negative, positive])

np.save(path+"\\"+"testing"+"_image", X_data_test)
np.save(path+"\\"+"testing"+"_label", T_data_test)