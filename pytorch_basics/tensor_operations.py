"""
Tensor operations -- Part 1
We're going to be covering things like indexing and slicing,
reshaping, tensors of tensor views,
as well as tensor arithmetic and basic operations
Concepts:   1. Indexing and slicing columns and elements
            2. .view() and .reshape() tensors
            3. SOS INFER the 1st or 2nd dimension of the tensors
            4. Arithmetic operations
            5. Underscore operator ("_") in arithmetics
"""
import torch
import numpy as np

# How indexing works in tensors???

# Create a tensor of values 0 to 5
x = torch.arange(6).reshape(3,2)
x
# Grab a single value from a tensor
print(x[1,1])

# Grab a column from a tensor
# Here we indexing for column 1
x[:, 1]

# Grab a column in vertical output
# Here we slicing for column 1
x[:, 1:]

# How to just view tensors .view()
# .view() provides a way of displaying tensors but doesn't save it in the variable

x = torch.arange(10)
x
x.view(2,5) # view tensor in a 2 by 5 tensor

# .view() reflects the most current data
# EXAMPLES --> 
x = torch.arange(10)
z = x.view(2, 5)
x[2] = 9999 # change the 3rd value from the tensor
print(x) # tensor x changed in elements
print(z) # But also tensor z changed in elements

# INFER what the 2nd dimension should be.
# in case you have large tensors and you 
# can't calculate the second dimension of the tensor 
# you simply do x.view(2, -1) meaning that
# I want a tensor that the first dimension is 5
# and figure the second dimension
x = torch.arange(20)
x.shape
x.view(5, -1)
# or
x.view(-1, 5) # INFER the first dimension

# Reshape tensors
x.reshape(2,5)

##### Basic arithmetic with tensors #####

# Add tensors 
a = torch.tensor([1.,2.,3.])
b = torch.tensor([4.,5.,6.])
a+b # output --> tensor([5., 7., 9.])
# OR
torch.add(a, b)

# Multiplication
a.mul(b) # output --> tensor([ 4., 10., 18.])

# How to re-assign variables in arithmetics (with the underscore operator)
a.mul_(b) # Use underscore operator
a # output --> tensor([ 4., 10., 18.]) now "a" is a variable that has
  # the multiplied elements from tensors "a" and "b"


"""
Tensor operations -- Part 2
We're gonna be covering things like dot products,
matrix multiplication, and some more advanced operations
Concepts:   1. 2-D true multiplucation
            2. numel() and len() in tensor
            3. numel() and len() in (torhc.rand)random number tensor
"""

# Matrix multiplucation (true dimensional multiplucation)
a = torch.tensor([[1,2,3],[4,5,6]]) # 2-D 
b = torch.tensor([[6,7],[8,9],[10,11]]) # 2-D

# Matrix multiplucation requires the columns of "a" tensor
# to match the number of rows in "b" tensor 
# Check shapes before multiplying
a.shape #torch.Size([2, 3])
b.shape #torch.Size([3, 2])

# torch.mm supports 2-D tensors as inputs
torch.mm(a,b) #--> mm = multiplucation

# Find the number of elements in a tensor
x = torch.tensor([1.,2.,3.,4.])
print(x)
x.numel() # number of elements
#or
len(x) # number of elements in a tensor

# If we use torch.rand() then the len() is not going to give you
# the correct len. In that case you use only the .numel()
y = torch.rand(2,3)
print(y)
len(y) # len() gives you the first dimension of the tensor



""""
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