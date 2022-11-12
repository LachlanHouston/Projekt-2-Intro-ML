# Regression, part a
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Read the data set as a pandas dataframe
filename = "C:/Users/ssjsi/Documents/GitHub/Projekt-2-Intro-ML/Projekt 2/penguins_size.csv"
data = pd.read_csv(filename)

# 1: Feature transformations and preparing the dataset and target variable

# Extract attribute names
islandNames = np.array(["Torgersen" , "Biscoe" , "Dream"])
speciesNames = np.array(["Adelie" , "Chinstrap" , "Gentoo"])
sexNames = np.array(["Female", "Male"])

# Extract target attribute
bodmas = data["body_mass_g"].to_numpy()

# Put categorical data on right side of data matrix X
data = data.drop(["body_mass_g"],axis = 1)
attributeNames = np.asarray(data.columns[range(0,6)])
data = data.to_numpy()
data[:,[0,4]] = data[:, [4,0]]
data[:,[1,3]] = data[:, [3,1]]
attributeNames[[0,4]] = attributeNames[[4,0]]
attributeNames[[1,3]] = attributeNames[[3,1]]
X = data

# Create one-out-of-K encodings from categorical attributes
sex = (np.array(X[:, -1], dtype=int).T)-1
K = sex.max()+1
sex_encoding = np.zeros((sex.size, K))
sex_encoding[np.arange(sex.size), sex] = 1

species = (np.array(X[:, -2], dtype=int).T)-1
K = species.max()+1
species_encoding = np.zeros((species.size, K))
species_encoding[np.arange(species.size), species] = 1

island = (np.array(X[:, -3], dtype=int).T)-1
K = island.max()+1
island_encoding = np.zeros((island.size, K))
island_encoding[np.arange(island.size), island] = 1

# Create y with the target attribute
bodmas = (bodmas-np.mean(bodmas))/np.std(bodmas)
y = np.array(bodmas)

N = len(y)

# Standardize
X2 = X - np.ones((N, 1))*X.mean(0)
X = X2*(1/np.std(X2,0))

# Add one-out-of-K encodings to data matrix X
X = np.concatenate( (X[:, :-3], island_encoding), axis=1)
attributeNames = np.concatenate((attributeNames[range(0,3)], islandNames), axis = 0)

X = np.concatenate( (X, species_encoding), axis=1)
attributeNames = np.concatenate((attributeNames, speciesNames), axis = 0)

X = np.concatenate( (X, sex_encoding), axis=1)
attributeNames = np.concatenate((attributeNames, sexNames), axis = 0)

# Create separate X data matrices with and without one-of-K encodings
Xk = X
X = X[:,range(0,3)]
N,M = X.shape
Nk, Mk = Xk.shape
attributeNamesK = attributeNames
attributeNames = attributeNames[range(0,3)]

## 2: Introducing a regularization parameter and estimating generalization error

# Fitting a linear regression model for predicting y
import sklearn.linear_model as lm
model = lm.LinearRegression()

modelK = model.fit(Xk,y)
model = model.fit(X,y)

# Predict body mass
y_est = model.predict(X)
residual = y_est-y

# Display scatter plot of predictions vs true values, and a histogram
# of residual error
plt.figure(figsize=[6.4,6.6])
plt.subplot(2,1,1)
plt.plot(y, y_est, '.')
plt.xlabel('Bodymass (true)')
plt.ylabel('Body mass (estimated)')
plt.subplot(2,1,2)
plt.hist(residual,40)

# Introduce lambda and estimate generalization error
from sklearn import model_selection
from toolbox_02450 import rlr_validate
from matplotlib.pyplot import plot, figure, show, xlabel, ylabel, legend, subplot, semilogx, grid, loglog, title, clim

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = np.concatenate(([u'Offset'], attributeNames),axis=0)
M = M+1

# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)

# Values of lambda
lambdas = np.power(10.,range(-5,9))

# Initialize variables
#T = len(lambdas)
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))

k=0
for train_index, test_index in CV.split(X,y):
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10    
    
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]

    # Estimate weights for unregularized linear regression, on entire training set
    w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
    # Compute mean squared error without regularization
    Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]
    # OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
    #m = lm.LinearRegression().fit(X_train, y_train)
    #Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    #Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    # Display the results for the last cross-validation fold
    if k == K-1:
        figure(k, figsize=(13,8))
        subplot(1,2,1)
        semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
        xlabel('Regularization factor')
        ylabel('Mean Coefficient Values')
        grid()
        # You can choose to display the legend, but it's omitted for a cleaner 
        # plot, since there are many attributes
        legend(attributeNames[1:], loc='best')
        
        subplot(1,2,2)
        title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        legend(['Train error','Validation error'])
        grid()
    
    # To inspect the used indices, use these print statements
    #print('Cross validation fold {0}/{1}:'.format(k+1,K))
    #print('Train indices: {0}'.format(train_index))
    #print('Test indices: {0}\n'.format(test_index))

    k+=1

show()
# Display results
print('Linear regression without feature selection:')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))

print('Weights in last fold:')
for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m,-1],2)))

## 3: How is a new data observation predicted according to the model?
# Find the weights for opt_lambda and write out the prediction model

y_model_weights = np.array([w_rlr[1,-1],w_rlr[2,-1],w_rlr[3,-1]])

