#%%
import pandas as pd
import numpy as np
import matplotlib as plt
from scipy.io import loadmat
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
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
cullen = (cullen-np.mean(cullen))/np.std(cullen)
culdep = (culdep-np.mean(culdep))/np.std(culdep)
flilen = (flilen-np.mean(flilen))/np.std(flilen)
bodmas = (bodmas-np.mean(bodmas))/np.std(bodmas)

# Create X and y, X needs to be reshaped to be used later
X = np.array([(flilen, culdep, cullen)]).reshape(-1,3)
y = np.array(bodmas)

n = len(y)

#%%
# Define test proportion, and then create k-folds of training and test data
K = 10
N_trials = 10
test_proportion = 1/K

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_proportion, shuffle=True)
ntest = len(y_test)
#%%
# Find baseline (mean of the y training set)
baseline = np.mean(y_train)

# Compute Mean Squared Error
baseerror =1/ntest * np.sum(y_test-baseline)**2
print("baseline mean squared error: ", baseerror)
#%%

# Classification (a.2)
###############################################################################

def rlr_validate(X,y,lambdas,cvf=10):
    ''' Validate regularized linear regression model using 'cvf'-fold cross validation.
        Find the optimal lambda (minimizing validation error) from 'lambdas' list.
        The loss function computed as mean squared error on validation set (MSE).
        Function returns: MSE averaged over 'cvf' folds, optimal value of lambda,
        average weight values for all lambdas, MSE train&validation errors for all lambdas.
        The cross validation splits are standardized based on the mean and standard
        deviation of the training set when estimating the regularization strength.
        
        Parameters:
        X       training data set
        y       vector of values
        lambdas vector of lambda values to be validated
        cvf     number of crossvalidation folds     
        
        Returns:
        opt_val_err         validation error for optimum lambda
        opt_lambda          value of optimal lambda
        mean_w_vs_lambda    weights as function of lambda (matrix)
        train_err_vs_lambda train error as function of lambda (vector)
        test_err_vs_lambda  test error as function of lambda (vector)
    '''
    
    
    CV = model_selection.KFold(cvf, shuffle=True)
    M = X.shape[1]
    w = np.empty((M,cvf,len(lambdas)))
    train_error = np.empty((cvf,len(lambdas)))
    test_error = np.empty((cvf,len(lambdas)))
    f = 0
    y = y.squeeze()
    baseerror = []
    for train_index, test_index in CV.split(X,y):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        ntest = len(y_test)
        
        baseline = np.mean(y_train)

        # Compute Mean Squared Error
        baseerror.append( 1/ntest * np.sum(y_test-baseline)**2 )
        
        # Standardize the training and set set based on training set moments
        mu = np.mean(X_train[:, 1:], 0)
        sigma = np.std(X_train[:, 1:], 0)
        
        X_train[:, 1:] = (X_train[:, 1:] - mu) / sigma
        X_test[:, 1:] = (X_test[:, 1:] - mu) / sigma
        
        # precompute terms
        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train
        for l in range(0,len(lambdas)):
            # Compute parameters for current value of lambda and current CV fold
            # note: "linalg.lstsq(a,b)" is substitue for Matlab's left division operator "\"
            lambdaI = lambdas[l] * np.eye(M)
            lambdaI[0,0] = 0 # remove bias regularization
            w[:,f,l] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
            # Evaluate training and test performance
            train_error[f,l] = np.power(y_train-X_train @ w[:,f,l].T,2).mean(axis=0)
            test_error[f,l] = np.power(y_test-X_test @ w[:,f,l].T,2).mean(axis=0)
    
        f=f+1

    opt_val_err = np.min(np.mean(test_error,axis=0))
    opt_lambda = lambdas[np.argmin(np.mean(test_error,axis=0))]
    train_err_vs_lambda = np.mean(train_error,axis=0)
    test_err_vs_lambda = np.mean(test_error,axis=0)
    mean_w_vs_lambda = np.squeeze(np.mean(w,axis=1))
    
    return opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda, baseerror
#%%
N, M = X.shape

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
M = M+1

## Crossvalidation
# Create crossvalidation partition for evaluation
CV = model_selection.KFold(K, shuffle=True)
#CV = model_selection.KFold(K, shuffle=False)

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

k = 0
for train_index, test_index in CV.split(X,y):
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10    
    
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda, baseerror = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

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
    Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/len(y_train)
    Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/len(y_test)

    # Estimate weights for unregularized linear regression, on entire training set
    w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
    # Compute mean squared error without regularization
    Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]
    # OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
    #m = lm.LinearRegression().fit(X_train, y_train)
    #Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    #Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]
    
    k += 1

print(lambdaI)
print(baseerror)




#%% Classification (Virker)


from matplotlib.pyplot import (figure, plot, title, xlabel, ylabel, 
                               colorbar, imshow, xticks, yticks, show)
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.metrics import confusion_matrix
from numpy import cov
from sklearn import metrics
from sklearn.model_selection import KFold
from scipy import stats
from sklearn.linear_model import LogisticRegression
from math import floor, log10
from statsmodels.stats.contingency_tables import mcnemar
import matplotlib as plt
from scipy.stats import beta


# function to round a number to specified number of significant figures 
def sig_figs(x: float, precision: int):
    x = float(x)
    precision = int(precision)

    return round(x, -int(floor(log10(abs(x)))) + (precision - 1))

# Seperate data
cullen = data["culmen_length_mm"]
culdep = data["culmen_depth_mm"]
flilen = data["flipper_length_mm"]
bodmas = data["body_mass_g"]

spec   = data["species"]

# Standardize
cullen = (cullen-np.mean(cullen))/np.std(cullen)
culdep = (culdep-np.mean(culdep))/np.std(culdep)
flilen = (flilen-np.mean(flilen))/np.std(flilen)
bodmas = (bodmas-np.mean(bodmas))/np.std(bodmas)

# Create X and y, X needs to be reshaped to be used later
X = np.array([(flilen, culdep, cullen,bodmas)]).reshape(-1,4)
y = np.array(spec)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4, test_size=0.1)


cv_outer = KFold(n_splits=10, shuffle=True, random_state=42)
# enumerate splits
KNN_errors = []
KNNoptK = []
baseline_errors = []
logistic_errors = []
logistic_optLambda = []

base_preds = []
KNN_preds = []
log_preds = []
true_labels = []
log_weights = []

for train_ix, test_ix in cv_outer.split(X):
	# split data
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]
    error_rate = []
    true_labels += list(y_test)
    
    "KNN model"
    for i in range(1,40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train,y_train)
        pred_i = knn.predict(X_test)
        error_rate.append(np.mean(pred_i != y_test))
    
    # plt.figure(figsize=(10,6))
    # plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', 
    #           marker='o',markerfacecolor='red', markersize=10)
    # plt.title('Error Rate vs. K Value')
    # plt.xlabel('K')
    # plt.ylabel('Error Rate')
    
    KNNoptK.append(error_rate.index(min(error_rate)))
    KNN_errors.append(round(np.mean(error_rate),2))
    
    # Get predictions for optimal K
    knn = KNeighborsClassifier(n_neighbors=  error_rate.index(min(error_rate)))
    knn.fit(X_train,y_train)
    KNN_preds += list(knn.predict(X_test))
    
    "Baseline model"
    baseline_pred = stats.mode(y_train)
    baseline_error = np.mean(baseline_pred != y_test)
    baseline_errors.append(round(baseline_error,2))
    base_preds += [baseline_pred[0][0]]*len(y_test)
    
    "Logistic regression model"
    # Fit regularized logistic regression model to training data to predict 
    # the type of wine
    lambda_interval = np.logspace(-8, 2, 50)
    train_error_rate = np.zeros(len(lambda_interval))
    test_error_rate = np.zeros(len(lambda_interval))
    coefficient_norm = np.zeros(len(lambda_interval))
    for k in range(0, len(lambda_interval)):
        mdl = LogisticRegression(penalty='l2', C=1/lambda_interval[k] )
        
        mdl.fit(X_train, y_train)

        y_train_est = mdl.predict(X_train).T
        y_test_est = mdl.predict(X_test).T
        
        train_error_rate[k] = np.sum(y_train_est != y_train) / len(y_train)
        test_error_rate[k] = np.sum(y_test_est != y_test) / len(y_test)

        w_est = mdl.coef_[0] 
        coefficient_norm[k] = np.sqrt(np.sum(w_est**2))

    min_error = np.min(test_error_rate)
    opt_lambda_idx = np.argmin(test_error_rate)
    opt_lambda = lambda_interval[opt_lambda_idx]  
    
    logistic_optLambda.append(sig_figs(opt_lambda,2))
    logistic_errors.append(round(min_error,2))
    
    
    # Get predictions for optimal lambda
    mdl = LogisticRegression(penalty='l2', C=1/opt_lambda )
    mdl.fit(X_train, y_train)
    y_train_est = mdl.predict(X_train).T
    y_test_est = mdl.predict(X_test).T  
    log_preds += list(y_test_est)
    w_est = mdl.coef_[0]
    log_weights.append(w_est)

print("KNN errors: ", KNN_errors)
print("Optimal K: ", KNNoptK)
print("\nBaseline errors: ", baseline_errors)
print("\nLogistic errors: ", logistic_errors)
print("Logistic lambda: ", logistic_optLambda)
#%%

w1 = np.zeros(10)
w2 = np.zeros(10)
w3 = np.zeros(10)
w4 = np.zeros(10)
for i in range(10):
    w1[i] = log_weights[i][0]
    w2[i] = log_weights[i][1]
    w3[i] = log_weights[i][2]
    w4[i] = log_weights[i][3]

w1 = np.mean(w1)
w2 = np.mean(w2)
w3 = np.mean(w3)
w4 = np.mean(w4)
print("w1: ", w1, "\nw2: ", w2, "\nw3: ", w3, "\nw4: ", w4)
#%%
### McNemars Test ###

KNN_correct = log_correct = base_correct = 0

# KNN vs Baseline
n = 333
n00 = n01 = n10 = n11 = 0
for i in range(len(true_labels)):
    if KNN_preds[i] == true_labels[i]:
        KNN_correct += 1
    if base_preds[i] == true_labels[i]:
        base_correct += 1
        
    if (KNN_preds[i] != true_labels[i]) and (base_preds[i] != true_labels[i]):
        n00 += 1
    elif (KNN_preds[i] != true_labels[i]) and (base_preds[i] == true_labels[i]):
        n01 += 1
    elif (KNN_preds[i] == true_labels[i]) and (base_preds[i] != true_labels[i]):
        n10 += 1
    elif (KNN_preds[i] == true_labels[i]) and (base_preds[i] == true_labels[i]):
        n11 += 1    

Contingency_KNN_base = [[n00, n01], [n10, n11]]
print(Contingency_KNN_base)
test = mcnemar(Contingency_KNN_base, correction=True)
print("pvalue (KNN vs Baseline)", test.pvalue)

alpha = 0.05
E0 = (n10-n01)/n
theta = E0
Q = ( n**2*(n+1)*(E0+1)*(1-E0) ) / ( n*(n10+n01)-(n10-n01)^2 )
f = ((E0+1)/2) * (Q-1)
g = ((1-E0)/2) * (Q-1)
lower = 2 * beta.ppf(alpha/2, f, g) - 1
upper = 2 * beta.ppf(1-alpha/2, f, g) - 1
print("CI KNN base: ", lower, upper)

# Logistic vs Baseline
n00 = n01 = n10 = n11 = 0
for i in range(len(true_labels)):
    if log_preds[i] == true_labels[i]:
        log_correct += 1
    
    if (log_preds[i] != true_labels[i]) and (base_preds[i] != true_labels[i]):
        n00 += 1
    elif (log_preds[i] != true_labels[i]) and (base_preds[i] == true_labels[i]):
        n01 += 1
    elif (log_preds[i] == true_labels[i]) and (base_preds[i] != true_labels[i]):
        n10 += 1
    elif (log_preds[i] == true_labels[i]) and (base_preds[i] == true_labels[i]):
        n11 += 1    

Contingency_log_base = [[n00, n01], [n10, n11]]
print(Contingency_log_base)
test = mcnemar(Contingency_log_base, correction=True)
print("pvalue (Logistic vs Baseline)", test.pvalue)

alpha = 0.05
E0 = (n10-n01)/n
theta = E0
Q = ( n**2*(n+1)*(E0+1)*(1-E0) ) / ( n*(n10+n01)-(n10-n01)^2 )
f = ((E0+1)/2) * (Q-1)
g = ((1-E0)/2) * (Q-1)
lower = 2 * beta.ppf(alpha/2, f, g) - 1
upper = 2 * beta.ppf(1-alpha/2, f, g) - 1
print("CI log base: ", lower, upper)

# Logistic vs KNN
n00 = n01 = n10 = n11 = 0
for i in range(len(true_labels)):
    if (log_preds[i] != true_labels[i]) and (KNN_preds[i] != true_labels[i]):
        n00 += 1
    elif (log_preds[i] != true_labels[i]) and (KNN_preds[i] == true_labels[i]):
        n01 += 1
    elif (log_preds[i] == true_labels[i]) and (KNN_preds[i] != true_labels[i]):
        n10 += 1
    elif (log_preds[i] == true_labels[i]) and (KNN_preds[i] == true_labels[i]):
        n11 += 1    

Contingency_log_KNN = [[n00, n01], [n10, n11]]
print(Contingency_log_KNN)
test = mcnemar(Contingency_log_KNN, correction=True)
print("pvalue (Logistic vs KNN)", test.pvalue)

alpha = 0.05
E0 = (n10-n01)/n
theta = E0
Q = ( n**2*(n+1)*(E0+1)*(1-E0) ) / ( n*(n10+n01)-(n10-n01)^2 )
f = ((E0+1)/2) * (Q-1)
g = ((1-E0)/2) * (Q-1)
lower = 2 * beta.ppf(alpha/2, f, g) - 1
upper = 2 * beta.ppf(1-alpha/2, f, g) - 1
print("CI log KNN: ", lower, upper)

print("\nMean KNN acc: ", KNN_correct/n)
print("\nMean Logistic acc: ", log_correct/n)
print("\nMean Baseline acc: ", base_correct/n)


#%%


n = len(y)

K=10
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)
CV = model_selection.KFold(n_splits=K,shuffle=True)

for train_index, test_index in CV.split(X):
    print('Computing CV fold: {0}/{1}..'.format(k+1,K))

    # extract training and test set for current CV fold
    X_train, y_train = X[train_index,:], y[train_index]
    X_test, y_test = X[test_index,:], y[test_index]

npdata = data.to_numpy()

attributeNames = np.asarray(data.columns[range(2,6)])
classNames = np.unique(npdata[:,0])
N, M = X.shape
C = len(classNames)


# Plot the training data points (color-coded) and test data points.
figure(1)
styles = ['.b', '.r', '.g']
for c in range(C):
    class_mask = (y_train==c)
    plot(X_train[class_mask,0], X_train[class_mask,1], styles[c])


# K-nearest neighbors
K=5

# Distance metric (corresponds to 2nd norm, euclidean distance).
# You can set dist=1 to obtain manhattan distance (cityblock distance).
dist=2
metric = 'minkowski'
metric_params = {} # no parameters needed for minkowski

# You can set the metric argument to 'cosine' to determine the cosine distance
#metric = 'cosine' 
#metric_params = {} # no parameters needed for cosine

# To use a mahalonobis distance, we need to input the covariance matrix, too:
#metric='mahalanobis'
#metric_params={'V': cov(X_train, rowvar=False)}

# Fit classifier and classify the test points
knclassifier = KNeighborsClassifier(n_neighbors=K, p=dist, 
                                    metric=metric,
                                    metric_params=metric_params)
knclassifier.fit(X_train, y_train)
y_est = knclassifier.predict(X_test)


# Plot the classfication results
styles = ['ob', 'or', 'og', 'oy']
for c in range(C):
    class_mask = (y_est==c)
    plot(X_test[class_mask,0], X_test[class_mask,1], styles[c], markersize=10)
    plot(X_test[class_mask,0], X_test[class_mask,1], 'kx', markersize=8)
title('Synthetic data classification - KNN');

# Compute and plot confusion matrix
cm = confusion_matrix(y_test, y_est);
accuracy = 100*cm.diagonal().sum()/cm.sum()
error_rate = 100-accuracy;
print("Accuracy", accuracy)
figure(2);
imshow(cm, cmap='binary', interpolation='None');
colorbar()
xticks(range(C)); yticks(range(C));
xlabel('Predicted class'); ylabel('Actual class');
title('Confusion matrix (Accuracy: {0}%, Error Rate: {1}%)'.format(accuracy, error_rate));

show()






