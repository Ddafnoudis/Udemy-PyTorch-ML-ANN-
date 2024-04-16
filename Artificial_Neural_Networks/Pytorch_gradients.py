"""
PyTorch Gradients
"""
import torch

# Set up computational tracking on the tensor
x = torch.tensor(2.0, requires_grad=True)
# This is the function
y = 2*x**4+x**3+3*x**2+5*x+1 
print(f'The output of x is: {x}') # The output of x is: 2.0
print(f"The ouput of y is: {y}") # The ouput of y is: 63.0

# check the type of y
print(type(y)) # <class 'torch.Tensor'>

# .backward() performs backpropagation
y.backward()

# Display the resulting gradient off of x
print(f"The x gradient is: {x.grad}") # --> results tensor(93.)
# 93. represents the slope of the polynomial 
# at that point (2, 63) where X is equal to two x = tensor(2.0)
# The gradient of a function y with respect to a vector x
# represents the vector of partial derivatives of y
# with respect to each variable in x, providing information about
# the rate of change of y in different directions.

### Backpropagation on multiple steps ###
x = torch.tensor([[1.,2.,3.], [3.,2.,1.]], requires_grad=True)
print(x)
# Create the first layer
y = 3*x+2
print(y)

# Create second layer
z = 2*y**2
print(z)

# Set the output to be the matrix mean
matrix_mean = z.mean()
print(matrix_mean)

# Perform backpropagation to find the gradient of x
# with respect to the output layer
matrix_mean.backward()
print(x.grad)
