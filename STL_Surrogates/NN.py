# Neural Network based surrogate of the Sound Transmission Loss analyses
# Author: Barbara Zaparoli Cunha
# barbara.zaparoli-cunha@ec-lyon.fr

# %% Import Libraries

import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from pathlib import Path
import copy
import time
import pandas as pd
import numpy as np
import gc
    
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import TransformedTargetRegressor
from keras.wrappers.scikit_learn import KerasRegressor

from functions.evaluation import plot_hist, evaluate_ML
from functions.prepro import Stiffness_Term, Transf_mulp,FeatureSelector, separateXY, Resonant_Term_Real
from functions.model import model_NN

# Set seeModel for reproducibility  
np.random.seed(100);tf.compat.v1.random.set_random_seed(100);seed = 100


# %% Import Files

# Import files from finite plates without 1/3 octave band average
def importar_finite_no_avg():
    # Import file with inputs and outputs from STL analyses
    url = './data/'+Model+'/df_STL.txt'
    df_or = pd.read_csv(url, sep=",", header=None)
    df_or.rename(columns={0: 'h', 1: 'rho', 2: 'E', 3: 'v', 4: 'neta', 
                       5: 'a', 6: 'b', 7: 'iD'}, inplace=True)
    df = df_or.copy(deep=True)

    # Import file with frequencies in which outputs were calculated
    url = './data/'+Model+'/df_freq.txt'    
    omega = pd.read_csv(url, sep=",", header=None)
    return df, omega

# Import files from finite plates with 1/3 octave band average
def importar_finite():
    # Import file with inputs and outputs from STL analyses
    url = './data/'+Model+'/df_STL_avg.txt'
    df_or = pd.read_csv(url, sep=",", header=None)
    df_or.rename(columns={0: 'h', 1: 'rho', 2: 'E', 3: 'v', 4: 'neta', 
                       5: 'a', 6: 'b', 7: 'iD'}, inplace=True)
    df = df_or.copy(deep=True)

    # Import file with frequencies in which outputs were calculated
    url = './data/'+Model+'/df_freq_avg.txt'    
    omega = pd.read_csv(url, sep=",", header=None)
    return df, omega

# Import files from infinite plates model
def importar_infinite():    
    # Import file with inputs and outputs from STL analyses
    url = './data/'+Model+'/df_STL.txt'
    df_or = pd.read_csv(url, sep=",", header=None)
    df_or.rename(columns={0: 'h', 1: 'rho', 2: 'E', 3: 'v', 4: 'neta', 
                       5: 'a', 6: 'b', 7: 'iD'}, inplace=True)
    df = df_or.copy(deep=True)

    # Import file with frequencies in which outputs were calculated
    url = './data/'+Model+'/df_freq.txt'    
    omega = pd.read_csv(url, sep=",", header=None)

    return df, omega



# %% NN Regressors
    
def regressor_history():
    start_time = time.time()
    history = regressor.fit(X_train_T,y_train_T)
    plot_hist(results_path, history)
    evaluate_ML(results_path, pre_name, regressor, X_test, y_test, 'STL','dB',
                   N_samples, start_time, file_report, writer,  omega.astype(int).values[0],
                   preprocess_pipe = Pipe_Preproc, output_transf = std_y,
                   curve = True, curve_var = 'STL')  
    
    regressor.model.summary(print_fn=lambda x: file_report.write(x + '\n'))
    return

def regressor_pipe():
    start_time = time.time()
    
    # Pipeline with Preprocessing + Regressor
    Pipe_PP_M = Pipeline([('pre_proc', Pipe_Preproc),
                          ('regressor', regressor)])
    #Pipeline with Preprocessing + Regressor + Output Transformation
    Pipe_Final = TransformedTargetRegressor(regressor = Pipe_PP_M,
                                            transformer = std_y,
                                            check_inverse = False)

    # Fit pipeline and evaluate performance
    Pipe_Final.fit(X_train,y_train)    
    evaluate_ML(results_path, pre_name, Pipe_Final, X_test, y_test, 'STL','dB',
                   N_samples, start_time, file_report, writer,  omega.astype(int).values[0],
                   curve = True, curve_var = 'STL')
    
    #Perform Cross Validation
    if Cross_val ==True:
        kfold = KFold(n_splits=n_kfold, shuffle=True)
        results = cross_val_score(Pipe_Final, X, Y, scoring='neg_root_mean_squared_error',cv=kfold,n_jobs=-1)
        print("\n RMSE: %.2f dB (%.2f dB)" % (-results.mean(), results.std()))
        file_report.write("\n RMSE: %.2f dB (%.2f dB) \n" % (-results.mean(), results.std()))
        writer_kfold.writerow([N_samples,-results.mean(),results.std()])
        
    return



########################################################################################################
# %% Run Analysis
Super_start_time = time.time()


# %% Define STL Model and ML method used


#'Model' can be defined as one of the followings: 'Infinite_Isotropic', 'Correction_Factor', 'Modal_Summation', 'Modal_Summation_avg','Comsol', 'Comsol_avg'
Model = 'Correction_Factor'

#'Method' can be defined as one of the followings: 'NN', 'XGB', 'RF', 'GPR'
Method = 'NN'

#Define if physics-guided features are used
physics_features = True

#Set title
title = 'NN with physicsc-guided features'

#Define model category
infinite = True if Model == 'Infinite_Isotropic' else False
avg = True if (Model == 'Comsol_avg' or Model == 'Modal_Summation_avg') else False


# %% Define ML Hyperparameters and preprocessing
l2 = 1e-7  #regularization l2
n1 = 32     #number of nodes in each hidden layer
drop = 0    #dropout rate
if (Model == 'Comsol' or Model == 'Modal_Summation'):
    epochs = 2500
else:
    epochs = 1500
batch_size = 32
test_size = 0.2
validation_split = 0.1

## Data scaling methods
stand_transf = StandardScaler()
output_transf = MinMaxScaler()

## Feature engineering
resonant_feature = False if (Model == 'Correction_Factor' or infinite or physics_features == False) else True

# Feature names in dataset
var = ['h', 'rho', 'E', 'v', 'neta','a','b', 'iD']

# Define features to be used
features_used =  ['h', 'rho', 'E', 'v', 'neta']
features_latex = ['$h$', '$\\rho$', '$E$', '$\\nu$', '$\\eta$']
if not infinite:
    features_used.append('a');features_latex.append('a')
    features_used.append('b');features_latex.append('b')
if physics_features == True:        
    features_latex.append('$m$')
    features_latex.append('$D$')
if resonant_feature ==  True: features_latex.append('$\\frac{D}{m(a^4 b^4)}$') 


# %% Define which analysis to run

with_history =  False    # "with_history =  True" run analysis without pipeline and plot the convergence history
with_pipe =     True     # "with_pipe =  True" run analysis with pipeline. Use this case to run Cross Validation

# Cross Validation. Bolean with_pipe should be True
Cross_val =     True     # "Cross_val = True" run Cross validation with n_kfold
n_kfold = 5


# %% Create files to register performance

#Set variable names and folder to save results
pre_name = Model+'_'+Method
folder = 'Results/'+Model+'/'+Method+'/'
Path(folder).mkdir(parents=True, exist_ok=True)    
file_csv = open('Results/'+'error_'+pre_name+'.csv','w')
writer = csv.writer(file_csv)

if with_pipe == True and Cross_val ==True:
    file_kfold = open('Results/'+'kfold_'+pre_name+'.csv','w')
    writer_kfold = csv.writer(file_kfold) 
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score 
    

# %% Import Database
if infinite == True:
    df_or, omega = importar_infinite() 
elif avg == True:
    df_or, omega = importar_finite() 
else:
    df_or, omega = importar_finite_no_avg() 
if omega.shape[1] == 1: omega = omega.T




# %% Loop of evaluation for different number of supporing points

for N_samples in [50,100,250,500,1000,1500,2000]:    
    
    ## File Name to save figures
    pre_file = pre_name+'_'+str(N_samples)+'_points' 
    results_path = folder+'/'+pre_file
    Path(results_path).mkdir(parents=True, exist_ok=True)

    ## Write Report
    file_reportn = pre_file + '_Report'
    file_report = open(results_path+'/'+file_reportn+".txt","a")
    comments = ["\n"+Model+"\n"
                "\n NN DETAILS: "+
                "\n With Pipe :" +str(with_pipe)+
                "\n With Cross_val :" +str(Cross_val)+ " - k folders:" +str(n_kfold)+
                "\n Preprocessing Method:" +str(stand_transf)+
                "\n Optimizer:"+"Adam"+
                "\n Epochs:" +str(epochs)+
                "\n batch_size:" +str(batch_size)+
                "\n l2 regularization:" +str(l2)+
                "\n Dropout rate:" +str(drop)+
                "\n Number of supporting points:" +str(N_samples)+"\n"]       
    file_report.write(' '.join(comments))
    

    ###########################################################
    
    # %% Prepare Dataset
    df = df_or.copy(deep=True)
    
    # Just use 'N_samples' from the dataset
    df = df.drop(labels = df_or.index[N_samples-1:-1], axis=0) 
    X,Y = separateXY(df, var)
    omega_shape = omega.shape   
    if len(Y.columns) != omega.shape[1]:
        import sys
        sys.exit("Error: Number of frequencies are not the same as number of outputs")

    #Split Data (Obs: no need to split when using KFold Validation)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


    # %% Pipeline

    Pipe1 = Pipeline([
            ('feat_sel', FeatureSelector(features_used)),
            ('standardize', stand_transf)       
            ])
    Pipe2 = Pipeline([
            ('feat_mulpt', Transf_mulp('h', 'rho')),
            ('standardize', StandardScaler())
            ])
    Pipe3 = Pipeline([                
            ('feat_stiff', Stiffness_Term('h', 'E', 'v')),
            ('standardize', StandardScaler())
            ])
    Pipe4 = Pipeline([                
            ('Resonant_Term_Real', Resonant_Term_Real('h', 'E', 'v', 'rho', 'a', 'b')),
            ('standardize',StandardScaler())                
            ])
    if physics_features == True and resonant_feature == False:
        Pipe_Preproc = Pipeline([
            ('feat_union', FeatureUnion(transformer_list=[
                  ('pipe1', Pipe1),
                  ('pipe2', Pipe2),
                  ('pipe3', Pipe3)]))
            ])
    elif physics_features == True and resonant_feature == True:  
        Pipe_Preproc = Pipeline([
            ('feat_union', FeatureUnion(transformer_list=[
                ('pipe1', Pipe1),
                ('pipe2', Pipe2),
                ('pipe3', Pipe3),
                ('pipe4', Pipe4)]))
            ])        
    else:
        Pipe_Preproc = Pipe1

    inpt_var = copy.copy(features_used);
    if (physics_features == True): inpt_var.append('Mass'), inpt_var.append('Stiffness')
    if (resonant_feature == True): inpt_var.append('Resonant_Term')
    
    # %% Preprocessing
    std_y = output_transf
    if not with_pipe:            
        y_train_T = std_y.fit_transform(np.asarray(y_train))

        Pipe_Preproc.fit(X_train)
        X_train_T = Pipe_Preproc.transform(X_train)

        inp_sh = X_train_T.shape[1]
        out_sh = y_train_T.shape[1]
        inp_sh_train = X_train_T.shape
        out_sh_train = y_train_T.shape
    else:
        inp_sh = len(inpt_var)
        out_sh = Y.shape[1]
    
    n_samples = X.shape[0]        
    print('Input Shape:',inp_sh, 'and Output Shape:', out_sh)
    print('Number of Samples:',n_samples) 
    file_report.write('Features used:'+', '.join(inpt_var)+ '\n')
    file_report.write('Input Shape:'+str(inp_sh)+ 'and Output Shape:'+ str(out_sh)+ '\n')
    file_report.write('Number of Samples:'+str(n_samples)+ '\n')
    file_report.write('Batch Size:'+str(batch_size)+ ' and Epochs:'+ str(epochs)+ '\n')
    if len(features_latex) != inp_sh:
        import sys
        sys.exit("Number of input features does not match!")


    # %% ML Model Construction and Training  

    regressor = KerasRegressor(build_fn = model_NN, inp_sh = inp_sh, out_sh = out_sh,
                               epochs = epochs, batch_size = batch_size, validation_split = validation_split,
                               verbose = 0, shuffle = False, n1 = n1, l2 = l2, drop = drop)

    if with_pipe == True:
        regressor_pipe()
        
    if with_history == True:
        regressor_history()
            
    print("Total Elapsed Time: {:7.2f}".format(time.time()-Super_start_time), "s \n")
    file_report.write(' '.join(["\n Total Time: {:7.2f}".format(time.time()-Super_start_time), "s \n"]) )    
    file_report.write("#################################################################### \n")
    file_report.close()
    
    #Clean memory
    gc.collect()
    
file_csv.close()        
if with_pipe == True and Cross_val == True:
    file_kfold.close()
    
    