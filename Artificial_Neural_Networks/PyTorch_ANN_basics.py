"""
Basic Pytorch with ANN
"""
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F # import activation and loss functions instantly without hand-written code
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# Load the iris dataset
iris = load_iris()

# Create a DataFrame from the iris dataset
iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)
# Add target values to the DataFrame
iris_data['target'] = iris.target

class Model(nn.Module): # call the class Model, and inherit from Module
    # Decide how many layers 
    # Input layer (4 features)--> h1 Neuron --> h2 Neuron--> output (3 classes)
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        """
        By calling super().__init__(), 
        you're essentially inheriting and extending the functionalities provided 
        by nn.Module in your custom neural network model, allowing you to add your own layers, parameters, 
        and methods on top of the base functionality provided by PyTorch.
        """
    # How many layers?
        super().__init__() # Instantiate the module (nn.Module). In essence,
        # it is a python constructor that calls the parent constructor(nn.Module)
        # Create the layers that are connected
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
        
    def forward(self, x):
        """
        Set the propagation method that propagates forward
        Activation functions are applied to the output of each layer
        """
        # Define the activation functions that will be used
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x

torch.manual_seed(32)
model = Model()

# Visualization
fig,axes = plt.subplots(nrows=1, ncols=4, figsize=(10,8))
fig.tight_layout()

plots = [(0,1), (2,3), (0,2), (1,3)]
colors = ['blue','red', 'green']
labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

for i, ax in enumerate(axes.flat):
    for j in range(3):
        x = iris_data.columns[plots[i][0]] # will give the first element from each tuple in the "plots" variable
        y = iris_data.columns[plots[i][1]] # will give the second element from each tuple in the "plots" variable
        # Filters the target column --> if j=1 selecte that target
        ax.scatter(iris_data[iris_data["target"]==j][x], iris_data[iris_data["target"]==j][y], color=colors[j], label=labels[j])
        ax.set(xlabel=x, ylabel=y)

fig.legend(labels=labels, loc=3, bbox_to_anchor=(1.0, 0.85))
plt.show()

# Define Target and features
X = iris_data.drop(columns=['target'])
y = iris_data['target']
# Define as numpy
X_array = X.values # returns a numpy array
y_array = y.values # returns a numpy array

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, test_size=0.2, random_state=33)

# Convert to tensors
X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
# Because we will use the cross entropy later we dont need to 
# convert y_train and y_test to float points
y_train = torch.from_numpy(y_train).long() 
y_test = torch.from_numpy(y_test).long()


# Criterian of our model
# We measure how far are you predicting from the actual values
criterion = nn.CrossEntropyLoss()
# print(model.parameters())
# print(model.parameters) --> layers of the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# lr = learning rate --> if the error or loss is not really going down
# after epochs we can decrease the lr.

# How many epochs do we want to train?
# An epoch is one run through training data
epochs = 100
losses = []

for i in range(epochs):
    # Forward through my network and get a prediction
    y_pred = model(X_train)
    # Calculate the loss
    loss = criterion(y_pred, y_train)
    # append to list losses
    losses.append(loss)

    if i%10 == 0:
        print(f'epoch: {i} loss: {loss.item():10.6f}')
        # The output will be:
        # epoch: 0 loss:   1.150745
        # epoch: 10 loss:   0.937145
        # epoch: 20 loss:   0.779624

    # Perform Backpropagation Process
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

##### Error ######
# if we do:
# plt.plot(range(epochs), losses)
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.show()
# Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.

# To fix the issue
losses = torch.stack(losses).detach().numpy()
plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()


# Validation of the model
with torch.no_grad(): # no gradient tracking # We dont care about backpropagation anymore
    # so we remove it and this makes the model faster

    y_val_pred = model.forward(X_test)
    val_loss = criterion(y_val_pred, y_test)
    print(f'Validation Loss: {val_loss.item():10.6f}') # Validation Loss:   0.058169

# How to see how many flowers we predicted correctly?
correct = 0
"""
This code block is used to iterate through the test data, make predictions using the model, and print the predicted value and the true value for each test sample.
The `with torch.no_grad():` block is used to disable gradient tracking, which is not needed for inference. This can make the model faster.
The `for i, data in enumerate(X_test):` loop iterates through the test data, 
and for each sample, it calls the `model.forward(data)` method to get the predicted value, and then prints the predicted value and the true value (`y_test[i]`) for that sample.
"""
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)
        print(f"{i+1}.) {str(y_val)}, {y_test[i]}")
        # The output will be
        # 1.) tensor([-2.1235,  4.8067, -0.8803]), 1
        # 2.) tensor([-1.7920,  5.3100, -1.5693]), 1
        # 3.) tensor([  6.3723,   0.8741, -10.0971]), 0

        if y_val.argmax() == y_test[i]:
            correct += 1
print(f" We got {correct} correct.")

# How to see the predicted class vs the true class
with torch.no_grad():
    print("Predicted vs True values")
    for i, data in enumerate(X_test):
        y_val = model.forward(data)
        print(f"{i+1}.) {str(y_val.argmax().item())}, {y_test[i]}")
        # the output will be
        # 1.) 1, 1
        # 2.) 1, 1
        # 3.) 0, 0
        # 4.) 1, 1

#### How to save the model? #####
# save the model as a dictionary in the current working folder and give the name.pt 
#torch.save(model.state_dict(),'my_iris_model.pth')
# state_dict saves only the parameters (weights and bias)

# If you want to save all the model you say 
#torch.save(model,'my_iris_model.pth')
# If we want to create a new model or update this one with new parameters
new_model = Model()
new_model.load_state_dict(torch.load('my_iris_model.pth'))
print(new_model.eval())

##### How to see the model in a new unseen data? ####
# We will create a new flower
mystery_iris = torch.tensor([5.6, 3.2, 4.5, 1.5])

# Visualization
fig,axes = plt.subplots(nrows=1, ncols=4, figsize=(10,8))
fig.tight_layout()

# Plot the iris data with the mystery_iris
plots = [(0,1), (2,3), (0,2), (1,3)]
colors = ['blue','red', 'green']
labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica', 'mystery iris']
for i, ax in enumerate(axes.flat):
    for j in range(3):
        x = iris_data.columns[plots[i][0]]
        y = iris_data.columns[plots[i][1]]
        ax.scatter(iris_data[iris_data["target"]==j][x], iris_data[iris_data["target"]==j][y], color=colors[j], label=labels[j])
        ax.set(xlabel=x, ylabel=y)
    # Add the mystery iris
    ax.scatter(mystery_iris[plots[i][0]], mystery_iris[plots[i][1]], color='yellow', label=labels[3])

fig.legend(labels=labels, loc=3, bbox_to_anchor=(1.0, 0.85))
plt.show()

## How to see the model in a new unseen data?
with torch.no_grad():
    print(new_model(mystery_iris))
    print(new_model(mystery_iris).argmax())
