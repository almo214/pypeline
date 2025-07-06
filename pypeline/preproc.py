from sklearn import preprocessing
import numpy as np
import pandas as pd


def scale_and_impute(x, impute_method: str = "mean", skip: list= None, drop_nan: bool= True):
    """Impute missing and normalize all x values of a pandas DataFrame or series object."""
    new_x = x.copy()

    #Replace any string NaNs with true numpy NaN
    new_x = new_x.replace('NaN', np.nan)



    #Establish names of numeric columns for imputation.
    if isinstance(new_x, pd.Series):
        new_x = impute(new_x, impute_method)
        #Scale
        maxx = new_x.max()
        minx = new_x.min()
        new_x = (new_x - minx) / (maxx- minx)

    else:    
        if drop_nan:
            new_x.dropna(axis=1, how='all', inplace=True)
        numeric_cols = list(new_x.select_dtypes(include=[np.number]).columns)
        new_x[numeric_cols] = impute( new_x[numeric_cols], impute_method)
        #Remove any columns the user wants to skip imputation on
        if skip is not None:
            for item in skip:
                if item in numeric_cols:
                    numeric_cols.remove(item)
        #Scale 
        min_max_scaler = preprocessing.MinMaxScaler()
        new_x[numeric_cols] = min_max_scaler.fit_transform(new_x[numeric_cols])




    #Output normalized and imputed x values
    return new_x


def impute(x1,  impute_method):
    """Impute pandas DataFrame"""
    #Impute using indicated method
    if impute_method in ['zero',0, '0']:
        imp_x= x1.fillna(0)
    elif impute_method in ['mean','mu','mn']:
        imp_x= x1.fillna(x1.mean())
    else:
        raise ValueError("Unknown imputation method. Please use 'zero' or 'mean'.")
    return imp_x
