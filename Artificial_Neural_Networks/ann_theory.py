"""
Summary
Concepts:       1) Perceptron model
                2) Neural Networks, 
                2) Activation Functions, 
                3) Cost functions, 
                4) Feed Forward Networks, 
                5) Backpropagation
"""

"""
Perceptron Model (single Neural Network)
"""
# Dendrites(input)--> Nucleous() --> Axon(output)
# To let the model (perceptron) able to learn we need
# to assign weights in the inputs--> x1w1 + x2w2
# But if x is equal to zero then weights want change anything since
# x1 * w1 will give zero
# Therefore we add bias --> x1 * w1 + b && x2 * w2 + b

### SOS POINT HERE ###
# The effect on the output "y" depends on the 
# overcome of the x1 * w1 against the bias

"""
Multi-Layer perceptron Model (Neural Networks)
"""

# Question:  When NNs become "deep neural networks"?
# When there are 2 or more hidden layers!

"""
Activation Functions for a single output
"""

# Binary step
# In a binary classification we have an output of 
# 0 or 1. Therefore any output that is between 
# 0 and 1 will take a solid 1 but any values that is 
# < 0 will take a solid value of zero in wx+b.
# Therefore we dont have a dynamic function.

# Sigmoid function --> Gives 0 and 1 but in a dynamic way

# Hyperbolic Tangent: tanhz(z) --> similar to the sigmoid function but!!!
# Gives outputs between -1 and 1 instead of 0 and 1

# ReLU (Rectified Linear Unit) --> gives the actual value if value > 0
# max(0,z)
# you can describe it as max zero, z,
# which essentially states that if the output
# of the value is less than zero, we treat it as zero.
# Otherwise, if it's greater than zero, we go ahead
# and output that actual z value

"""
Multi-Class Activation Functions
"""

# Types of multi-class situations
# 1) Non-Exclusive Classes --> a data point can have multiple classes/categories
#   assigned to it (e.g photos with different labels)
# E.G.
# Data point 1 = A,B
# Data point 2 = A,
# Data point 3 = C,B 
# 2) Mutually Exclusive Classes --> Only one class per data point 
#                            (e.g photos categorized based on grayscaled(white, black))

# How to choose correct Activation functions based on types of multi-class
#????

"""
Cost Functions or loss functions or error funtions
Gradient Descent
"""
###### Questions ######
# 1) After the network creates its prediction, how do we evalueate it?
# 2) After evaluation how can we update the network's weights and biases? -->
# --> go to backpropagation

# Quadratic Cost function
# calculate the difference between the real valuεs 
# against our predicted values.
# Cost function C(W, B, Sr, Er) where:1) W --> weights
#                                     2) B --> bias 
#                                     3) Sr --> input of a single training sample
#                                     4) Er --> desired output of that training sample

# Question
# How do we calculate cost funtion?
# what number of weight minimize the cost function?

# Answer:
# First we know that the C(w) will be n-dimensional
# We can use gradient descent for that problem
# so..to find the w-minimum we need to take a slope and then 
# figure the learning rate of the model. The learning 
# rate describes the speed of finding the weights values
# OR
# We can use adaptive gradient descent --> larger steps and then
# smaller as we realize the slope gets closer to zero
# Tool: Adam --> optimize the way of finding the minimum weight


##### NOTE ##### 
# Gradient notation --> when dealing with N-dimensional vectors(tensors)
# ΔC(w1,w2,...wn)

# For binary classification --> cross entropy --> propability distribution for 
# all the classes.


"""
Backpropagation
"""

# How it works?
