# /usr/local/bin/python3

# python3 -W ignore user_segmentation.py -nclusters 3 -a .05


import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from matplotlib.backends.backend_pdf import PdfPages
import datetime
import os
import itertools
import argparse
import pickle

# Seaborn visualization library
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier

import scipy

import random
random.seed( 0 )

def test_print(f):
    return print(1)

def results_summary_to_dataframe(results):
    '''take the result of an statsmodel results table and transforms it into a dataframe'''
    pvals = results.pvalues
    tvals = results.tvalues
    coeff = results.params
    conf_lower = results.conf_int()[0]
    conf_higher = results.conf_int()[1]

    results_df = pd.DataFrame({"pvals":pvals,
                               "coeff":coeff,
                               "tvals":tvals,
                               "conf_lower":conf_lower,
                               "conf_higher":conf_higher
                                })

    #Reordering...
    results_df = results_df[["coeff","pvals","tvals","conf_lower","conf_higher"]]
    return results_df


def dataprep_X (df, varmap, valmap, varlist_X, samsung_only=False, ohdict=None):
    """
    Data Prep for attribute variables for linear regression independent 
    
    - one hot encoding of select vars
    - standardization
    - replaces missing values with 0
    

    Parameters
    ----------
    df : dataframe
        data to be prepped
    varmap : dataframe
        varmap to be modified
    valmap : dataframe
        valmap to pull value Labels from
    varlist : list
        column names for Z 
    ohdict : dictionary
             {string : list}
        variables for one-hot-encoding
        if value==None, encode each value
        

    Returns
    -------
    X : dataframe
    varmap : dataframe

    """
    
    if samsung_only:
        df = df[df.hbrand==1]   
    
    
    print(varlist_X)  
    
    if ohdict:
        # one-hot encode variables
        for key, value in ohdict.items():
            if key in varlist_X:
                if not value:
                    varlist_X.remove(key) 
                    for v in df[key].unique() :
                        df[key+'_'+str(v)] = np.where(df[key]==v, 1, 0)
                        varlist_X.extend([key+'_'+str(v)])
                        
                        try:
                            l1 = varmap.loc[varmap.Variable==key].Label.values[0]
                            l2 = valmap.loc[(valmap.key==key)&(valmap.id==v)].Label.values[0]
                            Label = l1 +" "+l2
                            keys = ['Variable', 'Label', 'Short Label', 'Type', 
                                    'Minimum', 'Maximum', 'Description']
                            values = [key+'_'+str(v), Label, '', '', '', '', '']
                            varmap=varmap.append(pd.DataFrame(dict(zip(keys, values)), 
                                                              index=[0]))
                        except:
                            print ("(key, value) not in X: (%s,%d), Skipping. "%(key, v))
                    
                
    print(varlist_X)               
    df = df[varlist_X]
    
    # drop highly correlated female with male
    if 'd1_2' in df.columns:
        df = df.drop(['d1_2'], axis=1)
    
    # coerce non-int columns to numeric
    coerce_cols = df.dtypes[df.dtypes!='int64'].index.to_list()
    df[coerce_cols] = df[coerce_cols].apply(pd.to_numeric, errors='coerce')
    
    # drop columns where are values are missing 
    cols_allmissing = df.isna().sum()[df.isna().sum()==3254].index.to_list()
    df = df.drop(cols_allmissing, axis = 1) 
    
    # Standardize the qx_list columns
    # NaNs are treated as missing values: 
    # disregarded in fit, and maintained in transform.
#     scaler = StandardScaler().fit(df)
    X = df.copy()
#     X[df.columns] = scaler.transform(df)
          
    # fill NA's with 0 
    X.fillna(0, inplace=True)
    
    # Feature Engineering
#     if any('activitiesxrecency' in sublist for sublist in varlist_X) :
        
    # … feature activities recency - Basic vs. Advanced
    # … feature activities recency - Photo, Social, Productivity, Settings
          
   
       
    
    return X, varmap


def dataprep_Y (df, var, samsung_only=False):
    """
    Data Prep for attribute variables for linear regression dependent
    
    - standardization 
    - replaces missing values with 0

    Parameters
    ----------
    df : dataframe
        data to be prepped
        

    Returns
    -------
    Y : dataframe

    """
    
    
    if samsung_only:
        df = df[df.hbrand==1]   
    
    df = pd.DataFrame(df[var])
    
    # coerce non-int columns to numeric
    coerce_cols = df.dtypes[df.dtypes!='int64'].index.to_list()
    df[coerce_cols] = df[coerce_cols].apply(pd.to_numeric, errors='coerce')
    
    # drop columns where are values are missing 
    cols_allmissing = df.isna().sum()[df.isna().sum()==3254].index.to_list()
    df = df.drop(cols_allmissing, axis = 1) 
    
    # Standardize the qx_list columns
    # NaNs are treated as missing values: disregarded in fit, and maintained in transform.
    scaler = StandardScaler().fit(df)
    Y = df.copy()
    Y[df.columns] = scaler.transform(df)
          
    # fill NA's with 0 
    Y.fillna(0, inplace=True)
    
    # Feature Engineering
    # if any('activitiesxrecency' in sublist for sublist in varlist_X) :
        
    # … feature activities recency - Basic vs. Advanced
    # … feature activities recency - Photo, Social, Productivity, Settings
          
   
       
    
    return Y
# def dataprep_X (df, varlist=None, bystring=None):
    
#     """
#     Data Prep for Model Inputs
    
#     - standardization
#     - replaces missing values with 0

#     Parameters
#     ----------
#     df : dataframe
#         data to be prepped
#     elbow : [True, False]
#         should a plot of WCSS be printed
#     varlist : list
#         column names  for input

#     Returns
#     -------
#     X : dataframe
#         subset of input df, prepped for analysis
#     elbow graph (printed)
        

#     """
    
#     if not bystring:
#         bystring='{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())
    
#     if varlist:
#         # subset dataframe
#         df = df[varlist]

#     # coerce non-int columns to numeric
#     coerce_cols = df.dtypes[df.dtypes!='int64'].index.to_list()
#     df[coerce_cols] = df[coerce_cols].apply(pd.to_numeric, errors='coerce')
    
#     # drop columns where are values are missing 
#     cols_allmissing = df.isna().sum()[df.isna().sum()==3254].index.to_list()
#     df = df.drop(cols_allmissing, axis = 1) 
    
#     # Standardize the qx_list columns
#     # NaNs are treated as missing values: disregarded in fit, and maintained in transform.
#     scaler = StandardScaler().fit(df)
#     X = df.copy()
#     X[df.columns] = scaler.transform(df)
          
#     # fill NA's with 0 
#     X.fillna(0, inplace=True)
    
    
#     # … drivers - trust, feeling function, simplicity
#     # .... drivers - - Basic vs. Advanced

    
#     return X
