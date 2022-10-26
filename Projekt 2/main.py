#%%
import pandas as pd
import numpy as np
import matplotlib as plt
from scipy.io import loadmat
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error


# Load Penguin data
data = pd.read_csv (r'/Users/frederikravnborg/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU-Frederik’s MacBook Pro/ML/Project 1/penguins.csv')
# data = pd.read_csv (r'C:/Users/Lachl/OneDrive/Documents/Intro ML/Projekt-2-Intro-ML/Projekt 2/penguins_size.csv')

# Seperate data
cullen = data["culmen_length_mm"]
culdep = data["culmen_depth_mm"]
flilen = data["flipper_length_mm"]
bodmas = data["body_mass_g"]

# Standardize
cullens = (cullen-np.mean(cullen))/np.std(cullen)
culdeps = (culdep-np.mean(culdep))/np.std(culdep)
flilens = (flilen-np.mean(flilen))/np.std(flilen)
bodmass = (bodmas-np.mean(bodmas))/np.std(bodmas)

# Create X and y, X needs to be reshaped to be used later
X = np.array([(flilens, culdeps, cullens)]).reshape(-1,3)
y = np.array(bodmass)

n = len(y)

#%%
# Define test proportion, and then create k-folds of training and test data
k = 10
N_trials = 10
test_proportion = 1/k

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_proportion, shuffle=True)
ntest = len(y_test)
# Find baseline (mean of the y training set)
baseline = np.mean(y_train)

# Compute Mean Squared Error
baseerror =1/ntest * np.sum(y_test-baseline)**2
print("Baseline mean squared error: ", baseerror)

#%%
# Define the LinearRegression model
LRmodel = LinearRegression()

# Define ANN model

#%%
# Fit the model to the training data
cv_acc = cross_val_score(LRmodel, X, y, cv=k, scoring="neg_mean_squared_error")

#%%
LRmodel.fit(X_train, y_train)

# Make predictions based on the data in the test set
predictions = LRmodel.predict(X_test)

#%%
# Find overall error by finding mean squared error between the predictions and the true labels
print("Linear Regression mean squared error: ", mean_squared_error(y_test, predictions))