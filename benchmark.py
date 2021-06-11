from typing import no_type_check


import numpy as np 
from scipy.linalg import blas
import time

a = np.ones(shape=(250, 250)).astype(np.float32)
b = np.transpose(a).astype(np.float32)
print(type(a[0][0]))

print('started')
s = time.time()
c = np.matmul(b,a)
print(time.time()-s)



s = time.time()
c = np.dot(b,a)
print(time.time()-s)

s = time.time()
c = b@a
print(time.time()-s)

s = time.time()
c = blas.sgemm(2,b,a)
print(time.time()-s)

