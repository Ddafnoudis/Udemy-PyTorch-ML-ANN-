"""
Machine Learning
"""

# machine learning is a method of data analysis
# that automates analytical model building,
# and the keyword here is automates.
# Using algorithms that iteratively learn from the data,
# machine learning allows computers to find hidden insights
# without being explicitly programmed where to look.

# So in classical programming,
# you would tell a computer what to do
# and what to look for in your data.
# With machine learning,
# you're just following a general set of rules,
# and then the program or algorithm itself
# will be able to find
# where these hidden insights are within your data.

"""
Supervised ML
"""
# Machine learning Process

# 1) Data Acquisition
# 2) Data cleaning
# 3) Split Test data and Training data
# 4) Fit a model to a training data
# 5) How the data performed --> test data
# 6) If model is not performing well examine it in training data

###### SOS question #######
# Is it actually fair to use
# the accuracy you get off that test data
# as your model's final performance metric
# since technically, after all,
# you were given the chance to update the model parameters
# again and again after evaluating your results on that test set?

# Answer --> Split data into training/validation/test sets
# where:    1) Training set --> trains the model
#           2) Validation set --> Tests the model performance and 
#                                  keeps room for improvements in the model
#           3) Test set --> 1 test for the real world performance of the model


"""
Overfitting and Underfitting
"""

# Overfitting --> Low error in training set / High error in test set
# fit too much to the noise of the data



"""
Evaluating Performance Classification
"""

# Accuracy, Precision, Recall, F1-score, 

##### Accuracy ######
# Number of predictions / total Number of predictions --> well balanced target classes

##### Recall ######
# Number of True positives / Number of True positives + Number of false negatives
# Recall = TP / TP+FN
# Find all relevant instances in dataset

##### Precision ######
# Number of True positives / Number of True positives + Number of false positives
# Precision = TP / TP+FP
# The proportion of data points our model says was relevant actually were relevant

##### F1-score ######
# Takes the harmonic mean of precision and recall
# F1-score = 2 * (precision * recall / precision + recall)
# This is because it punishes extreme values and 
# gives more weight to lower values, 
# making it sensitive to imbalanced classes.

"""
Evaluating Performance Regression
"""

# Metrics for continuous values

# Mean Absolute Error (MAE)
# Absolute value means how far a number is from zero
# https://www.mathsisfun.com/numbers/absolute-value.html
# MAE wont punish outliers

# Mean Squared Error (MSE)

# RMSE

"""
Unsupervised ML
"""

# Tasks in Unsupervised ML --> Clustering, Anomaly Detection,  Dimensionality Reduction

# Clustering --> Based on similarity
# Anomaly Detection --> Detect outliers (dissimilar points)
# Dimensionality Reduction --> Reducing number of features


