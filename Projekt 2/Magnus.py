#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, figure, show, xlabel, ylabel, legend, subplot, semilogx, grid, loglog, title, clim
from sklearn import model_selection, tree

from toolbox_02450 import rlr_validate, feature_selector_lr, bmplot
import random

#%%
#Dataload
data = pd.read_csv (r'C:/Users/Lachl/OneDrive/Documents/Intro ML/Projekt-2-Intro-ML/Projekt 2/penguins_size.csv')

#Uncomment if validation set is wanted
# random.seed(1234)
# validation_idxs = random.sample(range(0,213),11)
# validation_dataset = glass_dataset.iloc[validation_idxs]
# glass_dataset = glass_dataset.drop(validation_idxs)

labels = list(data.columns)
X = np.array(data[['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm']])
y = np.array(data['body_mass_g'])
attributeNames = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm']
N, M = X.shape
print(data.head())
# validation_dataset

#%%
standardized_X = (X-np.mean(X))/np.std(X)
standardized_y = (y-np.mean(y))/np.std(y)
# Kinda normal fordelt /TODO
# plt.hist(X[:,3],bins = 25)
# np.mean(X,0)
# np.std(X,0)
X = standardized_X
y = standardized_y
N, M = X.shape
# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = [u'Offset']+attributeNames
M = M+1

#%%
K = 10
CV = model_selection.KFold(K, shuffle=True)
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
lambdas = np.power(10.,range(-5,9))

#%%
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
        figure(k, figsize=(12,8))
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
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m,-1],4)))

#%%
# ANN setup
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from toolbox_02450 import train_neural_net

loss_fn = torch.nn.MSELoss() 

#%%
from tqdm import tqdm


outer_cross_validation  = 2
internal_cross_validation = 2
outer_Kfold = model_selection.KFold(outer_cross_validation, shuffle=True)
inner_Kfold = model_selection.KFold(internal_cross_validation, shuffle=True)


lambdas = np.power(10.,range(-5,9))

error_train_baseline = np.empty((outer_cross_validation,1))
error_test_baseline = np.empty((outer_cross_validation,1))

opt_lambda_array = np.empty((outer_cross_validation,1))
error_test_lin_mod = np.empty((outer_cross_validation,1))

ANN_outer_MSE_list = []
ANN_outer_hiddenH_list = []

k = 0
outer_fold_counter = 0
for train_partition, test_partition in outer_Kfold.split(X,y):
    outer_fold_counter +=1
    # print(train_partition, test_partition)
    X_partition_training = X[train_partition]
    y_partition_training = y[train_partition]
    X_partition_test = X[test_partition]
    y_partition_test = y[test_partition]

    #Baseline
    error_train_baseline[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    error_test_baseline[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]
    
    f = 0
    ANN_inner_fold_error = []
    ANN_inner_fold_nets = []
    ANN_inner_fold_n_list = []
    
    inner_fold_counter = 0
    for train_index, test_index in outer_Kfold.split(X_partition_training, y_partition_training):
        inner_fold_counter += 1
        # print(train_index, test_index)
        X_train = X_partition_training[train_index]
        y_train = y_partition_training[train_index]
        X_test = X_partition_training[test_index]
        y_test = y_partition_training[test_index]
        
        
        #Linear regression model
        X_for_lin_mod = np.concatenate((np.ones((X_train.shape[0],1)),X_train),1)
        X_for_lin_mod_test = np.concatenate((np.ones((X_test.shape[0],1)),X_test),1)
        # y_for_lin_mod = np.concatenate((np.ones((X_train.shape[0],1)),y_train),1)

        test_error = np.empty((internal_cross_validation,len(lambdas)))
        w = np.empty((M+1,internal_cross_validation,len(lambdas)))
        
        Xty = X_for_lin_mod.T @ y_train
        XtX = X_for_lin_mod.T @ X_for_lin_mod
        # print(Xty.shape)
        # print(XtX.shape)
        for l in range(0,len(lambdas)):
            # Compute parameters for current value of lambda and current CV fold
            # note: "linalg.lstsq(a,b)" is substitue for Matlab's left division operator "\"
            lambdaI = lambdas[l] * np.eye(M+1)
            lambdaI[0,0] = 0 # remove bias regularization
            w[:,f,l] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
            # Evaluate training and test performance
            test_error[f,l] = np.power(y_test-X_for_lin_mod_test @ w[:,f,l].T,2).mean(axis=0)
        f += 1   
        
        # ANN
        X_train_tensor = torch.Tensor(X[train_index,:])
        y_train_tensor = torch.Tensor(y[train_index]).unsqueeze(1)
        X_test_tensor = torch.Tensor(X[test_index,:])
        y_test_tensor = torch.Tensor(y[test_index]).unsqueeze(1)
        hidden_units = [5,10,20,50]
        for n in hidden_units:
            print(f'Currently on outerfold: {outer_fold_counter}, innerfold: {inner_fold_counter}, n = {n}')
            ANN_inner_fold_n_list.append(n)
            n_hidden_units = n
            model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(4, n_hidden_units), # Input equal to amount of attributes in dataset
                        torch.nn.ReLU(),   # 1st transfer function,
                        torch.nn.Linear(n_hidden_units,n_hidden_units ), 
                        # no final tranfer function, i.e. "linear output"
                        torch.nn.ReLU(),
                        torch.nn.Linear(n_hidden_units, 1 ) #Output is 1 continous variable equal to RI
                        )    
            net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train_tensor,
                                                       y=y_train_tensor,
                                                       n_replicates=1,)
                # Determine errors and errors
            # Determine estimated class labels for test set
            ANN_inner_fold_nets.append(net)
            y_test_est_inner = net(X_test_tensor)
            se = (y_test_est_inner.float()-y_test_tensor.float())**2 # squared error
            mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
            ANN_inner_fold_error.append(mse) # store error rate for current CV fold 
        
    
    
    # ANN Outer
    idx_for_lowest_error = np.argmin(ANN_inner_fold_error)
    best_hidden_units_for_inner_fold = ANN_inner_fold_n_list[idx_for_lowest_error]
    outer_fold_net = ANN_inner_fold_nets[idx_for_lowest_error]
        
    y_test_est_outer = outer_fold_net(X_test_tensor)
    se = (y_test_est_outer.float()-y_test_tensor.float())**2 # squared error
    mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
    
    ANN_outer_MSE_list.append(mse)
    ANN_outer_hiddenH_list.append(best_hidden_units_for_inner_fold)
    
    #RLR Outer
    opt_lambda_array[k] = lambdas[np.argmin(np.mean(test_error,axis=0))]
    
    
    X_partiton_for_lin_mod_test = np.concatenate((np.ones((X_partition_test.shape[0],1)),X_partition_test),1)
    
    lambdaI = opt_lambda_array[k] * np.eye(M+1)
    lambdaI[0,0] = 0
    Xty = X_for_lin_mod.T @ y_train
    XtX = X_for_lin_mod.T @ X_for_lin_mod
    w_rlr = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    error_test_lin_mod[k] = np.square(y_partition_test-X_partiton_for_lin_mod_test @ w_rlr).sum(axis=0)/y_partition_test.shape[0]





    
    #Enumerator for array indexies
    k += 1
    
error_for_pd = {'error_baseline':list(error_test_baseline),'error_lin_mod': list(error_test_lin_mod), 'error_ANN': list(ANN_outer_MSE_list)}
# print(error_test_baseline)
# print(error_test_lin_mod)
error_for_pd
# pd.DataFrame.from_dict(error_for_pd)

#%%
plot(error_test_baseline)
plot(error_test_lin_mod)
plot(ANN_outer_MSE_list)
legend(['baseline', 'lin mod','ANN'])