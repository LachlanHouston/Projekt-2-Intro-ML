#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, figure, show, xlabel, ylabel, legend, subplot, semilogx, grid, loglog, title, clim
from sklearn import model_selection
import torch

#%%
# From toolbox_02450
def rlr_validate(X,y,lambdas,cvf=10):
    CV = model_selection.KFold(cvf, shuffle=True)
    M = X.shape[1]
    w = np.empty((M,cvf,len(lambdas)))
    train_error = np.empty((cvf,len(lambdas)))
    test_error = np.empty((cvf,len(lambdas)))
    f = 0
    y = y.squeeze()
    for train_index, test_index in CV.split(X,y):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        
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
    
    return opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda

#%%
# From toolbox_02450
def train_neural_net(model, loss_fn, X, y,
                     n_replicates=3, max_iter = 10000, tolerance=1e-6):
    import torch
    # Specify maximum number of iterations for training
    logging_frequency = 1000 # display the loss every 1000th iteration
    best_final_loss = 1e100
    for r in range(n_replicates):
        print('\n\tReplicate: {}/{}'.format(r+1, n_replicates))
        # Make a new net (calling model() makes a new initialization of weights) 
        net = model()
        
        # initialize weights based on limits that scale with number of in- and
        # outputs to the layer, increasing the chance that we converge to 
        # a good solution
        torch.nn.init.xavier_uniform_(net[0].weight)
        torch.nn.init.xavier_uniform_(net[2].weight)
                     
        # We can optimize the weights by means of stochastic gradient descent
        # The learning rate, lr, can be adjusted if training doesn't perform as
        # intended try reducing the lr. If the learning curve hasn't converged
        # (i.e. "flattend out"), you can try try increasing the maximum number of
        # iterations, but also potentially increasing the learning rate:
        #optimizer = torch.optim.SGD(net.parameters(), lr = 5e-3)
        
        # A more complicated optimizer is the Adam-algortihm, which is an extension
        # of SGD to adaptively change the learing rate, which is widely used:
        optimizer = torch.optim.Adam(net.parameters())
        
        # Train the network while displaying and storing the loss
        print('\t\t{}\t{}\t\t\t{}'.format('Iter', 'Loss','Rel. loss'))
        learning_curve = [] # setup storage for loss at each step
        old_loss = 1e6
        for i in range(max_iter):
            y_est = net(X) # forward pass, predict labels on training set
            loss = loss_fn(y_est, y) # determine loss
            loss_value = loss.data.numpy() #get numpy array instead of tensor
            learning_curve.append(loss_value) # record loss for later display
            
            # Convergence check, see if the percentual loss decrease is within
            # tolerance:
            p_delta_loss = np.abs(loss_value-old_loss)/old_loss
            if p_delta_loss < tolerance: break
            old_loss = loss_value
            
            # display loss with some frequency:
            if (i != 0) & ((i+1) % logging_frequency == 0):
                print_str = '\t\t' + str(i+1) + '\t' + str(loss_value) + '\t' + str(p_delta_loss)
                print(print_str)
            # do backpropagation of loss and optimize weights 
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            
            
        # display final loss
        print('\t\tFinal loss:')
        print_str = '\t\t' + str(i+1) + '\t' + str(loss_value) + '\t' + str(p_delta_loss)
        print(print_str)
        
        if loss_value < best_final_loss: 
            best_net = net
            best_final_loss = loss_value
            best_learning_curve = learning_curve
        
    # Return the best curve along with its final loss and learing curve
    return best_net, best_final_loss, best_learning_curve
#%%
#Dataload
# data = pd.read_csv (r'C:/Users/Lachl/OneDrive/Documents/Intro ML/Projekt-2-Intro-ML/Projekt 2/penguins_size.csv')
data = pd.read_csv (r'/Users/frederikravnborg/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU-Frederik’s MacBook Pro/ML/Project 1/penguins.csv')


labels = list(data.columns)
X = np.array(data[['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm']])
y = np.array(data['body_mass_g'])
attributeNames = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm']

N, M = X.shape

#%%
# Standardize data
standardized_X = (X-np.mean(X))/np.std(X)
standardized_y = (y-np.mean(y))/np.std(y)

X = standardized_X
y = standardized_y
N, M = X.shape

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = [u'Offset'] + attributeNames
M = M+1

#%%
# Set k-fold strategy
K = 10
CV = model_selection.KFold(K, shuffle=True)

# Dataframes for linear regression
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
# ANN setup
loss_fn = torch.nn.MSELoss()

#%%
# Data frames for the different tests
outer_cross_validation  = 10
internal_cross_validation = 10
outer_Kfold = model_selection.KFold(outer_cross_validation, shuffle=True)
inner_Kfold = model_selection.KFold(internal_cross_validation, shuffle=True)


lambdas = np.power(10.,range(-5,9))

error_train_baseline = np.empty((outer_cross_validation,1))
error_test_baseline = np.empty((outer_cross_validation,1))

opt_lambda = np.empty((outer_cross_validation,1))
error_test_linreg = np.empty((outer_cross_validation,1))

hidden_units = [x for x in range(1,17,2)]
ANN_n_inner_error = np.empty((internal_cross_validation,len(hidden_units)))
ANN_outer_MSE = []
ANN_outer_hiddenH = []

outer_fold_counter = 0



#%%

# Dataframes for mcnemars test
baseline_outer_err = []
linreg_outer_err = []
ANN_outer_err = []

# Primary loop
k = 0
for train_partition, test_partition in outer_Kfold.split(X,y):
    outer_fold_counter +=1
    X_partition_training = X[train_partition]
    y_partition_training = y[train_partition]
    X_partition_test = X[test_partition]
    y_partition_test = y[test_partition]

    
    
    f = 0
    ANN_inner_fold_error = []
    ANN_inner_fold_nets = []
    ANN_inner_fold_n = []
    ANN_n_inner_error = np.empty((internal_cross_validation,len(hidden_units)))
    
    for inner_fold_counter,(train_index, test_index) in enumerate(outer_Kfold.split(X_partition_training, y_partition_training)):
        # print(train_index, test_index)
        X_train = X_partition_training[train_index]
        y_train = y_partition_training[train_index]
        X_test = X_partition_training[test_index]
        y_test = y_partition_training[test_index]
        
        
        #Linear regression model
        X_linreg = np.concatenate((np.ones((X_train.shape[0],1)),X_train),1)
        X_linreg_test = np.concatenate((np.ones((X_test.shape[0],1)),X_test),1)

        test_error = np.empty((internal_cross_validation,len(lambdas)))
        w = np.empty((M+1,internal_cross_validation,len(lambdas)))
        
        Xty = X_linreg.T @ y_train
        XtX = X_linreg.T @ X_linreg

        for l in range(0,len(lambdas)):
            # Compute parameters for current value of lambda and current CV fold
            # note: "linalg.lstsq(a,b)" is substitue for Matlab's left division operator "\"
            lambdaI = lambdas[l] * np.eye(M+1)
            lambdaI[0,0] = 0 # remove bias regularization
            w[:,f,l] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
            # Evaluate training and test performance
            test_error[f,l] = np.power(y_test-X_linreg_test @ w[:,f,l].T,2).mean(axis=0)
        f += 1   
        
        
        
    #Baseline Outer
    error_train_baseline[k] = np.square(y_partition_training-y_partition_training.mean()).sum(axis=0)/y_partition_training.shape[0]
    error_test_baseline[k] = np.square(y_partition_test-y_partition_test.mean()).sum(axis=0)/y_partition_test.shape[0]

    btemp_err = np.square(y_partition_test-y_partition_test.mean())
    baseline_outer_err.extend(btemp_err)
    
    #RLR Outer
    opt_lambda[k] = lambdas[np.argmin(np.mean(test_error,axis=0))]
    
    
    X_partiton_for_linreg_test = np.concatenate((np.ones((X_partition_test.shape[0],1)),X_partition_test),1)
    
    lambdaI = opt_lambda[k] * np.eye(M+1)
    lambdaI[0,0] = 0
    Xty = X_linreg.T @ y_train
    XtX = X_linreg.T @ X_linreg
    w_rlr = np.linalg.solve(XtX+lambdaI,Xty).squeeze()

    cur_err = np.square(y_partition_test-X_partiton_for_linreg_test @ w_rlr)
    linreg_outer_err.extend(cur_err)
    # Compute mean squared error with regularization with optimal lambda
    error_test_linreg[k] = np.square(y_partition_test-X_partiton_for_linreg_test @ w_rlr).sum(axis=0)/y_partition_test.shape[0]

    # ANN
    X_train_tensor = torch.Tensor(X[train_index,:])
    y_train_tensor = torch.Tensor(y[train_index]).unsqueeze(1)
    X_test_tensor = torch.Tensor(X[test_index,:])
    y_test_tensor = torch.Tensor(y[test_index]).unsqueeze(1)
    
    for n_idx, n in enumerate(hidden_units):
        n_hidden_units = n
        inner_model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units), # Input equal to amount of attributes in dataset
                    torch.nn.ReLU(),   # 1st transfer function,
                    torch.nn.Linear(n_hidden_units,n_hidden_units ), 
                    # no final tranfer function, i.e. "linear output"
                    torch.nn.ReLU(),
                    torch.nn.Linear(n_hidden_units, 1 ) #Output is 1 continous variable equal to RI
                    )    
        net, final_loss, learning_curve = train_neural_net(inner_model,
                                                    loss_fn,
                                                    X=X_train_tensor,
                                                    y=y_train_tensor,
                                                    n_replicates=1,)
            # Determine errors and errors
        # Determine estimated class labels for test set
        y_test_est_inner = net(X_test_tensor)
        se = (y_test_est_inner.float()-y_test_tensor.float())**2 # squared error
        mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
        ANN_n_inner_error[inner_fold_counter,n_idx] = mse

    # ANN Outer
    mean_error_of_inner_h = np.mean(ANN_n_inner_error,axis = 0)
    best_h = hidden_units[np.argmin(mean_error_of_inner_h)]
    ANN_outer_hiddenH.append(best_h)
    
    X_train_tensor = torch.Tensor(X_partition_training)
    y_train_tensor = torch.Tensor(y_partition_training).unsqueeze(1)
    X_test_tensor = torch.Tensor(X_partition_test)
    y_test_tensor = torch.Tensor(y_partition_test).unsqueeze(1)

    
        
    n_hidden_units = best_h
    outer_model = lambda: torch.nn.Sequential(
                torch.nn.Linear(4, n_hidden_units), # Input equal to amount of attributes in dataset
                torch.nn.ReLU(),   # 1st transfer function,
                torch.nn.Linear(n_hidden_units,n_hidden_units ), 
                # no final tranfer function, i.e. "linear output"
                torch.nn.ReLU(),
                torch.nn.Linear(n_hidden_units, 1 ) #Output is 1 continous variable equal to RI
                )    
    
    net, final_loss, learning_curve = train_neural_net(outer_model,
                                                loss_fn,
                                                X=X_train_tensor,
                                                y=y_train_tensor,
                                                n_replicates=1,)
        
    y_test_est_outer = net(X_test_tensor)
    se = ((y_test_est_outer.float()-y_test_tensor.float())**2) # squared error
    mse = (sum(se).type(torch.float)/len(y_test_est_outer)).data.numpy() #mean
    
    ANN_outer_err.extend(se)
    ANN_outer_MSE.append(mse)
    print(len(se))

    # Update counter
    k += 1


# Convert tensor dataframe to array
ANN_err = ANN_outer_err
for i in range(len(ANN_err)):
    ANN_err[i] = ANN_err[i].item()



#%%
error_All = np.array([baseline_outer_err, linreg_outer_err, ANN_err]).squeeze()
np.savetxt("errors10K.csv", error_All, delimiter=",")

#%%
plot(error_test_baseline)
plot(error_test_linreg)
plot(ANN_outer_MSE)
legend(['Baseline', 'Lin Reg','ANN'])
