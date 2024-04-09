""" 
How to use pytorch as a tensor array library?
What is a tensor?
 """

# scalar
# vector = 1D-TENSOR
# matrix = 2D-TENSOR
# tensor = 3D -TENSOR

"""
Tensor basics -- part 1
Concepts: 1. Create array
          2. Convert array to tensor (torch.from_numpy)
          3. Make a 2d array (np.arange && reshape)
          4. SOS: How to make a tensor untached by a numpy array?
"""

import torch
import numpy as np

torch.__version__

## what is a tensor
# A tensor is a multidimensional matrix containing elements of a single data type

arr = np.array([1,2,3,4,5])
type(arr)

# Convert numpy array to a tensor 1st option
x = torch.from_numpy(arr)

# Convert numpy array to a tensor 2nd option
x = torch.as_tensor(arr)

# make a 2d numpy array
arr2d = np.arange(0.0, 12.0) #Return evenly spaced values within a given interval
print(arr2d)
arr2d = arr2d.reshape(4, 3)
arr2d

#
x2 = torch.from_numpy(arr2d)
x2 ## in that point check output (float64)

# Take the first element from array and assign different variable
arr[0] = 99
arr
x

##### SOS COMMENT #####
# How to make a np.array object not to be connected to the tensor object?
# E.G
my_arr = np.array([1,2,3,4,5,6,7,8,9,10])
my_tensor = torch.from_numpy(my_arr)
my_tensor

my_other_tensor = torch.tensor(my_arr)
my_other_tensor

# Now change first element in array
my_arr[0] = 99
my_arr
my_tensor
# My_tensor is connected with array by the function .from_numpy(my_arr)
# But my_other_tensor is not connected because we used .tensor(my_arr)
my_other_tensor
# Conclusion: Use a .tensor if you want to make a tensor object untacked
##### END OF SOS COMMENT #####

"""
Tensor basics -- part 2
# How to create tensors from scratch using pytorch?
"""

new_arr = np.array([1,2,3]) 
# What are the differences between torch.tensor and torch.Tensor?
# The difference is in the dtype
new_arr.dtype # dtype('int32')
torch.tensor(new_arr) # dtype=torch.int32
o = torch.Tensor(new_arr) 
o.dtype # dtype=torch.float32

# Allocate a block memory(an unititialized data)
torch.empty(2,2) 

# Tensor with actual zeros
torch.zeros(4,3) # as float
torch.zeros(4,3, dtype=torch.int32) # as int

# Tensor with ones
torch.ones(4,3)

# Arrange values
torch.arange(0,18)
# Reshape to a tensor
torch.arange(0,18).reshape(6,3)
# Return back linear space points
torch.linspace(0, 10, 6).reshape(2, 3)

# How change the precision of data points(it depends of how precise you want to be)
""""
Important notes
int32: This data type represents 32-bit signed integers.
It can store integer values ranging from -2,147,483,648 to 2,147,483,647.

int64: This data type represents 64-bit signed integers. 
It can store integer values ranging from -9,223,372,036,854,775,808 
to 9,223,372,036,854,775,807.
""" 
# Create a variable
my_tensor = torch.tensor([1,2,3]) #dtype int64
my_tensor.dtype
n_tensor = my_tensor.type(torch.int32)
n_tensor.dtype

# Create tensor with random numbers from a uniform distribution
# Uniform distribution means that all number between 0 and 1
# have the same likelyhood to be picked
torch.rand(4, 3)

# Create tensor with random numbers from a standard normal distribution
# standard normal distribution means that we have a mean of 0
# and a stand deviation to 1
torch.randn(4, 3)

# Cre
torch.randint(low=0, high=10, size=(5,5))