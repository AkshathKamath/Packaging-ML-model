import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin

class MedianImputer(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None):
        self.variables = variables
    
    def fit(self,X,y=None):
        self.median_dict = {}
        for col in self.variables:
            self.median_dict[col] = X[col].median()
        return self
    
    def transform(self,X):
        X = X.copy()
        for col in self.variables:
            X[col] = X[col].fillna(self.median_dict[col])
        return X
    
## --------------------------------------------------------- ##

class ModeImputer(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None):
        self.variables = variables
    
    def fit(self,X,y=None):
        self.mode_dict = {}
        for col in self.variables:
            self.mode_dict[col] = X[col].mode()[0]
        return self
    
    def transform(self,X):
        X = X.copy()
        for col in self.variables:
            X[col] = X[col].fillna(self.mode_dict[col])
        return X

## --------------------------------------------------------- ##

class DropColumns(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None):
        self.variables = variables
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X = X.copy()
        X = X.drop(columns = self.variables)
        return X

## --------------------------------------------------------- ##

class AddColumns(BaseEstimator,TransformerMixin):
    def __init__(self,col1=None, col2=None):
        self.col1 = col1
        self.col2 = col2
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X = X.copy()
        X[self.col1] = X[self.col1] + X[self.col2]
        return X
    
## --------------------------------------------------------- ##

class LabelEncoder(BaseEstimator,TransformerMixin):
    def __init__(self, variables=None):
        self.variables=variables
    
    def fit(self, X,y):
        self.label_dict = {}
        for var in self.variables:
            t = X[var].value_counts().sort_values(ascending=True).index 
            self.label_dict[var] = {k:i for i,k in enumerate(t,0)}
        return self
    
    def transform(self,X):
        X=X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.label_dict[feature])
        return X

## --------------------------------------------------------- ##

class LogTransformer(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None):
        self.variables = variables

    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X = X.copy()
        X[self.variables] = np.log(X[self.variables])
        return X
    
## --------------------------------------------------------- ##

