# /usr/local/bin/python3

# python3 -W ignore user_segmentation.py -nclusters 3 -a .05
import os
import sys

import pandas as pd
import glob
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from matplotlib.backends.backend_pdf import PdfPages

import seaborn as sns

import sklearn

from sklearn.preprocessing import StandardScaler
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier

import scipy

import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics

import statsmodels.api as sm
from scipy import stats

import lxml
from io import StringIO 
import pickle

from fuzzywuzzy import process


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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

def OLS_loop (var_y_list, ohdict, X, df_data_Y, df_varmap):
    tp_list = []
    df_list = []
    for var_y in var_y_list:
        Y = dataprep_Y (df_data_Y, var_y)
        regressor = LinearRegression()  
        regressor.fit(X, Y)
        X2 = sm.add_constant(X)
        est = sm.OLS(Y, X2)
        est2 = est.fit()

        df_est = results_summary_to_dataframe(est2)
        df_est[(df_est.pvals<.05)].sort_values('coeff').round(5)

        df_temp = df_varmap[['Variable','Label']].merge(df_est[['coeff','pvals']][
            (df_est.index=='const')|(df_est.pvals<.05)].sort_values('pvals').round(5), left_on='Variable', right_index=True, 
                                          how='right')
        df_temp['var_y'] = var_y

        df_list.append(df_temp)
        tp_list.append((var_y, est2.rsquared))

    df_lmresults = pd.concat(df_list)
    df_lmresults = df_lmresults.drop_duplicates()
    
    df_lmresults.Label = df_lmresults.Label.str.replace(
    ' - Which of the following best describes your current employment status\?','',
                                                              regex=True)

    df_lmresults.Label = df_lmresults.Label.str.replace(r'D2. What is your age\?',r'Age',
                                                                regex=True)

    df_lmresults.Label = df_lmresults.Label.str.replace(r' - What is your race\?',r'',
                                                                regex=True)
    
    df_lmresults.Label = df_lmresults.Label.str.replace(r'D3. ',r'',
                                                                regex=True)
    df_lmresults.Label = df_lmresults.Label.str.replace(r'D4. What is the highest level of education you have completed\? ',r'',
                                                                regex=True)
    df_lmresults.Label = df_lmresults.Label.str.replace(r'D7. ',r'',
                                                                regex=True)
    df_lmresults.Label = df_lmresults.Label.str.replace(r'D1. Are you\?',r'Gender - ',
                                                                regex=True)
    df_lmresults.Label = df_lmresults.Label.str.replace(r'hBrand. Hidden Question - ',r'',
                                                                regex=True)
    df_lmresults.Label = df_lmresults.Label.str.replace(r'D6. Are you of Hispanic, Latino or Spanish origin\?',r'',
                                                                regex=True)
    
    df_lmresults.Label = df_lmresults.Label.str.replace(r'Retired or otherwise unable to work',r'Retired',
                                                                regex=True)
   

    


    return tp_list, df_lmresults

def PC1_loop(SuperLabels, SuperList, df_data, df_varmap):
    pc1_exvr_list = []
    pc1_comp_list = []
    for label, varlist in zip(SuperLabels,SuperList):

        df = df_data[varlist]        
        coerce_cols = df.dtypes[df.dtypes!='int64'].index.to_list()
        df[coerce_cols] = df[coerce_cols].apply(pd.to_numeric, errors='coerce')
        cols_allmissing = df.isna().sum()[df.isna().sum()==3254].index.to_list()
        df = df.drop(cols_allmissing, axis = 1) 

        scaler = StandardScaler().fit(df)
        X = df.copy()
        X[df.columns] = scaler.transform(df)
        X.fillna(0, inplace=True)

        pca = PCA(n_components=5)
        principalComponents = pca.fit_transform(X)

        pc1_comp = pd.DataFrame(pca.components_[0],X.columns)
        pc1_comp = pc1_comp.merge(df_varmap[['Variable','Label','Short Label']], 
                                  left_index=True, right_on='Variable')
        pc1_comp.loc[:,'SuperLabel'] = label
        pc1_comp_list.append(pc1_comp)
        pc1_exvr_list.append((label,pca.explained_variance_ratio_[0]))
        pickle.dump( pca, open( '../data/output/'+label+'_pca.p', "wb" ) )

    pc1_MASTER = pd.concat(pc1_comp_list)

    index = df_data.index
    columns = [label+'_PC1' for label in SuperLabels]
    df_data_PC1 = pd.DataFrame(index=index, columns=columns)
    df_data_PC1 = df_data_PC1.fillna(0) 

    for label, varlist in zip(SuperLabels,SuperList):
        df_temp = df_data.filter(like=label, axis=1).apply(
            pd.to_numeric, errors='coerce').multiply(pc1_MASTER[pc1_MASTER.SuperLabel==label].set_index('Variable')[0]
                                                ).filter(like=label, axis=1).sum(axis=1, skipna=True)
        df_data_PC1.loc[:,label+'_PC1']=df_temp

    return pc1_exvr_list, df_data_PC1
