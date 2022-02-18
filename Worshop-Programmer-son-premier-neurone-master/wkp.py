import numpy as np
from util import *
from utils import *
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

my_matrix = np.array([1, 2, 3])
create_first_matrix(my_matrix)
###
b = my_new_array()
print(np.shape(b))

a1 = np.array([[np.random.rand(), np.random.rand()],
                [np.random.rand(), np.random.rand()]])
a2 = np.array([[np.random.rand(), np.random.rand(), np.random.rand(), np.random.rand()],
                [np.random.rand(), np.random.rand(), np.random.rand(), np.random.rand()],
                [np.random.rand(), np.random.rand(), np.random.rand(), np.random.rand()],
                [np.random.rand(), np.random.rand(), np.random.rand(), np.random.rand()]])
a3 = np.array([[np.random.rand(), np.random.rand()],
                [np.random.rand(), np.random.rand()],
                [np.random.rand(), np.random.rand()],
                [np.random.rand(), np.random.rand()],
                [np.random.rand(), np.random.rand()]])
check_random_matrix(a1, a2, a3)

a4 = np.dot(a3, a1)
# print(np.shape(a4))
check_mul(a4)
# print(a)
###

X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
y = y.reshape((y.shape[0], 1))

print('dimensions de X:', X.shape)
print('dimensions de y:', y.shape)

plt.scatter(X[:,0], X[:, 1], c=y, cmap='summer')
plt.show()

W, b = initialisation(12)
print(np.shape(W))
print(np.shape(b))

print(model(X, W, b))