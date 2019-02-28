import math
import numpy as np

print(math.pow(2, 3))
print(2**3)

print(np.zeros((3, 2), dtype='int32'))

print(np.round(3.7))

print(2 > 3)

# exponent
print('np.exp(2)->{}'.format(np.exp(2)))
print('np.power(2,2)->{}'.format(np.power(2, 2)))

mat_a = [[1,3,3],[2,5,3],[2,5,2]]
print(np.linalg.det(mat_a))

# covariance
a = np.array([[10, 5, 2, 4, 9, 3, 2.5], [10, 2, 8, 3, 7, 4, 1]])
# get covariance . if bias=False is to get Unbiased Covariance. Unbiased Covariance means divided by len(array) -1
# bias option's default is False.
cov_a=np.cov(a, bias=True)
print('np.cov(a)->{}'.format(cov_a))

# Transport Matrix
b = np.array([[1,2,3]])
print('b ->{}'.format(b ))
print('b.T ->{}'.format(b.T ))

# calculate matrix
mat_c = np.array([1,2])
mat_d = np.array([1,2]).T
print('mat_c*mat_d->{}'.format(mat_c*mat_d))
print('np.matmul.(mat_c,mat_d)->{}'.format(np.matmul(mat_c,mat_d)))

target_array = np.array([[60, 20]])
print('target_array->{}'.format(target_array))
diff_array = np.array([[target_array[0, 0] - 10, target_array[0, 1] - 5]])
print('diff_array->{}'.format(diff_array))
print('diff_array.T->{}'.format(diff_array.T))

mat_e = np.array([[0,1],[1,2]])

for item in mat_e:
    print('item->{}'.format(item))

print('E+3->{}'.format('E'+str(3)))

######## calculate constant
constant = 15 - 5 + 1

########################
print('log2->{}'.format(math.log2(167)))

# create empty array
empty_array = np.empty((3,2))
print('np.empty((3,2))->{}'.format(empty_array))
empty_array[0] = [3,2]
print('empty_array->{}'.format(empty_array))

## create array
height_array = []
gender_array = []

height_array.append(1)
height_array.append(2)
height_array.append(3)
gender_array.append('Female')
gender_array.append('Male')
gender_array.append('Female')

stack_array = np.vstack([height_array, gender_array])

print('stack_array->{}; type(stack_array)->{}'.format(stack_array,type(stack_array)))




