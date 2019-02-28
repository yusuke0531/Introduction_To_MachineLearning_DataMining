# TEST FOR Assignment 3
import numpy as np
import math

arrayT = np.array([[1,2,3],[4,5,6],[7,8,9]])
print("arrayT->{}".format(arrayT[:,2]))

array_after = arrayT.T
print("array_after->{}".format(array_after))


# mean test row column
X = np.array([[72.,101.,94.],[50.,96.,70.],[14.,79.,10.],[8.,70.,1.]], np.float64);
print(X)
u=np.mean(X,axis=0);print(u)   # column mean
u2=np.mean(X,axis=1);print(u2)  # row mean

# np.dot() np.matmul()

print('log2->{}'.format(math.log2(60000)))

# get the first index of the value
test_array = np.array([5,6,5,6,6,6,6])
print('(np.where(test_array==6)->{}'.format(np.where(test_array==6)))
print('(np.where(test_array==6)->{}'.format(np.array(np.where(test_array==6)).item(0)))
# to 1 line for above 2 lines
test_index = np.array(np.where(np.array([5,6,5,6,6,6,6])==6)).item(0)
print('test_index->{}'.format(test_index))


a=[[0,1],[1,2]]
b=[[10,11],[11,12]]
np.dot(a,b)
print(a)

array_a = np.array([1,2,3,4,5])

print('array_a->{}'.format(array_a[-1]))
print('array_a->{}'.format(array_a[-3:-1]))
print('array_a->{}'.format(array_a[-3:]))
print('array_a->{}'.format(array_a[:3]))
print('array_a->{}'.format(array_a[1:3]))
print('array_a->{}'.format(array_a[1]))

array_b = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20]])

print('array_b->{}'.format(array_b))
print('array_b[]->{}'.format(array_b[0,1]))
print('array_b[]->{}'.format(array_b[[0,1],:]))


