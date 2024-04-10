"""
Exercises in Pytorch basics
"""

import torch
import numpy as np

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Create a anumpy array called "arr" that contains 6 random integers
# between 0(inclusive) and 5 (exclusive)
arr = np.random.randint(low=0, high=5, size=6)
print(arr)

# Create a tensor "x" from the array above
x = torch.from_numpy(arr)
print(x)

# Change the dtype of x from int32 to int64
print(x.dtype) # torch.int32
x = x.to(torch.int64) # WRONG// THE RIGHT --> x = x.type(torch.int64)
print(x.dtype) # torch.int64

# Reshape x into a 3 by 2 tensor
x = x.reshape(3, 2)
x = x.view(3, 2)
print(x)
# Return the right-hand column of tensor x
print(x[:, 1])

# Without changing x, return a tensor of square values of x
print(x)
print(torch.square(x))
# or
print(x * x)

# Create a tensor "y" with the same number of elements as x, that can 
# be matrix-multiplied with x
y = torch.tensor([[2,2,1], [4,1,0]])
print(y)
# or
y = torch.randint(0,5,(2,3))
print(y)

# Find the matrix product of x and y
torch.mm(x, y)
# or
x.mm(y)