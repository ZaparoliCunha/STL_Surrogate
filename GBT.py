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

import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import TransformedTargetRegressor

from functions.evaluation import evaluate_ML
from functions.prepro import Nothing, Stiffness_Term, Transf_mulp,FeatureSelector, separateXY, Resonant_Term_Real

# Set seeModel for reproducibility  
np.random.seed(10);tf.compat.v1.random.set_random_seed(10);seed = 10


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
Model = 'Infinite_Isotropic'

#'Method' can be defined as one of the followings: 'NN', 'XGB', 'RF', 'GPR'
Method = 'RF'

#Define if physics-guided features are used
physics_features = True

#Set title
title = 'GPR with physicsc-guided features'

#Define model category
infinite = True if Model == 'Infinite_Isotropic' else False
avg = True if (Model == 'Comsol_avg' or Model == 'Modal_Summation_avg') else False
# Cross Validation
Cross_val =     True     # "Cross_val = True" run Cross validation with n_kfold
n_kfold = 5

# %% Define ML Hyperparameters and preprocessing

objective = 'reg:squarederror'
learning_rate = 0.05
max_depth = 10
n_estimators = 125
colsample_bytree = 1
subsample = 1
test_size = 0.2

## Data scaling methods
stand_transf = Nothing()
output_transf = Nothing()

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


# %% Create files to register performance

#Set variable names and folder to save results
pre_name = Model+'_'+Method
folder = 'Results/'+Model+'/'+Method+'/'
Path(folder).mkdir(parents=True, exist_ok=True)    
file_csv = open('Results/'+'error_'+pre_name+'.csv','w')
writer = csv.writer(file_csv)

if Cross_val ==True:
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

for N_samples in [50,100,250,500,1000,1500,2000] :
    
    
    ## File Name to save figures
    pre_file = pre_name+'_'+str(N_samples)+'_points' 
    results_path = folder+'/'+pre_file
    Path(results_path).mkdir(parents=True, exist_ok=True)

    ## Write Report
    file_reportn = pre_file + '_Report'
    file_report = open(results_path+'/'+file_reportn+".txt","a")
    comments = ["\n"+Model+"\n"
                "\n DETAILS: "+
                "\n With Cross_val :" +str(Cross_val)+ " - k folders:" +str(n_kfold)+
                "\n Preprocessing Method:" +str(stand_transf)+
                "\n n_estimators:"+str(n_estimators)+
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


    # %% Preprocessing Pipeline

    Pipe1 = Pipeline([
            ('feat_sel', FeatureSelector(features_used)),
            ('standardize', stand_transf)       
            ])
    Pipe2 = Pipeline([
            ('feat_mulpt', Transf_mulp('h', 'rho')),
            ('standardize', Nothing())
            ])
    Pipe3 = Pipeline([                
            ('feat_stiff', Stiffness_Term('h', 'E', 'v')),
            ('standardize', Nothing())
            ])
    Pipe4 = Pipeline([                
            ('Resonant_Term_Real', Resonant_Term_Real('h', 'E', 'v', 'rho', 'a', 'b')),
            ('standardize',Nothing())                
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
    
    std_y = output_transf
    inp_sh = len(inpt_var)
    out_sh = Y.shape[1]
    
    n_samples = X.shape[0]        
    print('Input Shape:',inp_sh, 'and Output Shape:', out_sh)
    print('Number of Samples:',n_samples) 
    file_report.write('Features used:'+', '.join(inpt_var)+ '\n')
    file_report.write('Input Shape:'+str(inp_sh)+ 'and Output Shape:'+ str(out_sh)+ '\n')
    file_report.write('Number of Samples:'+str(n_samples)+ '\n')
    if len(features_latex) != inp_sh:
        import sys
        sys.exit("Number of input features does not match!")


    # %% ML Model Construction and Training  

    regressor = MultiOutputRegressor(xgb.XGBRegressor(objective = objective, learning_rate = learning_rate, 
                                                      max_depth = max_depth, n_estimators = n_estimators,
                                                      colsample_bytree = colsample_bytree, subsample = subsample,n_jobs = -1)
                                     ,n_jobs = -1)
    
    regressor_pipe()
        
  
    print("Total Elapsed Time: {:7.2f}".format(time.time()-Super_start_time), "s \n")
    file_report.write(' '.join(["\n Total Time: {:7.2f}".format(time.time()-Super_start_time), "s \n"]) )    
    file_report.write("#################################################################### \n")
    file_report.close()
    
    #Clean memory
    gc.collect()
    
file_csv.close()        
if Cross_val == True: file_kfold.close()
    
    