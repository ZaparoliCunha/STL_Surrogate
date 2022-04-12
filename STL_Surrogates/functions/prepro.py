from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


# %% Prepare Data

def separateXY(df, var):
    size = len(var);
    X = df.drop(df.columns[size:],axis=1,inplace=False).copy()
    X.columns = var
    Y = df.drop(columns = X.columns, inplace = False).copy()
    return X,Y


class Nothing(BaseEstimator, TransformerMixin): 
        
    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        X_ = X.copy() 
        return X_
    
    def inverse_transform(self, X, y = None):
        X_ = X.copy() 
        return X_

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
        
    def fit(self, X, y=None):        
        print('Data Processing', end ="\n")
        return self
        
    def transform(self, X):        
        return X[self.feature_names]

# %% Feature Engineering
    
class Transf_mulp(BaseEstimator, TransformerMixin):

      def __init__(self, feature_1, feature_2):  
        self.feature_1 = feature_1
        self.feature_2 = feature_2
    
      def fit(self, X, y = None):
        return self

      def transform(self, X, y = None):
        X_ = X.copy() 
        X_new = (X_[self.feature_1] * X_[self.feature_2]).values.reshape(-1,1)
        return X_new

class Resonant_Term_Real(BaseEstimator, TransformerMixin):

    def __init__(self, h, E, v, rho, a,b):  
        self.h = h
        self.E = E
        self.v = v
        self.rho = rho
        self.a = a
        self.b = b

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        X_ = X.copy()  
        mp = X_[self.h]*X_[self.rho]
        D = ((X_[self.h]**3) * X_[self.E]/(1-X_[self.v]**2))    
        X_new = (D/(mp*(X_[self.a]**4)*(X_[self.b]**4))).values.reshape(-1,1)
        return X_new  
    
class Stiffness_Term(BaseEstimator, TransformerMixin):

    def __init__(self, h, E, v):  
        self.h = h
        self.E = E
        self.v = v

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        X_ = X.copy()         
        X_new = ((X_[self.h]**3) * X_[self.E]/(1-X_[self.v]**2)).values.reshape(-1,1)
        return X_new