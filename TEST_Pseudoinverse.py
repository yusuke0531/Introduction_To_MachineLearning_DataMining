import numpy as np

X = np.array([[1,2],[2,3],[5,6]])
print("Pseudoinverse X->{}".format(np.linalg.pinv(X)))

# get Psuedoinverse
X_1= np.matmul(X.T, X)
X_1_inv = np.linalg.inv(X_1)
X_1_pinv = np.matmul(X_1_inv, X.T)
print("X_1_inv ->{}".format(X_1_pinv))


