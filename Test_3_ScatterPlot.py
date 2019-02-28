import numpy as np
import matplotlib.pyplot as plt

######################################################
#
# Scatter Plot
#
# UCSC/02_SecondQuater/Introduction_To_MachineLearning_DataMining/Misc/ScatterPlot.html
#
######################################################
np.random.seed(2)
# rand(2,2) returns 2*2 matrix with random number
randommatrix = np.random.rand(2, 2)

# np.identity(2) returns the 2*2 identity matrix
# 1 0
# 0 1
cn = 2 * np.dot(randommatrix, randommatrix.T) + np.identity(2)
mun = [-2, -1]

np.random.seed(3)
randommatrix = np.random.rand(2, 2)
cp = 2 * np.dot(randommatrix, randommatrix.T) + np.identity(2)
mup = [0, 0]

Nn = 2500
Np = 2500

labeln = 1
labelp = 2

# multivariate_normal() returns Multivariate Normal Distribution
Pn = np.random.multivariate_normal(mun, cn, Nn)
# np.full() returns a matrix filled in by number.
# np.full(shape, number, dtype=np.int). shape (Nn,) represents vector[Nn].
Tn = np.full((Nn,), labeln, dtype=np.int)

Pp = np.random.multivariate_normal(mup, cp, Np)
Tp = np.full((Np,), labelp, dtype=np.int)

P = np.concatenate((Pn, Pp))
T = np.concatenate((Tn, Tp))

# For best effect, points should not be drawn in sequence but in random order
np.random.seed(0)

# permutation() Randomly permute a sequence. permute is near change meaning
randomorder = np.random.permutation(np.arange(len(T)))
print("randomorder->{}".format(randomorder))
# randomorder = np.arange(len(T))

# Set colors
cols = np.zeros((len(T), 4))  # Initialize matrix to hold colors

# if T is labels
cols[T == labeln] = [1, 0, 0, 0.25]  # Negative points are red (with opacity 0.25)
cols[T == labelp] = [0, 1, 0, 0.25]  # Positive points are green (with opacity 0.25)

# Draw scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, facecolor='black')

# scatter plot . scatter(x,y, s=the marker size, facecolors=color matrix, marker=symbol's shape."o" means circle )
ax.scatter(P[randomorder, 1], P[randomorder, 0], s=5, linewidths=0, facecolors=cols[randomorder, :], marker="o");
ax.set_aspect('equal')

plt.gca().invert_yaxis()
plt.show()
