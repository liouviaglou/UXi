# /usr/local/bin/python3

# python3 -W ignore latent_variables.py 

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
import numpy

# Seaborn visualization library
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier

import scipy

import random
random.seed( 0 )


def dataprep_X (df, elbow=True, varlist=None, bystring=None):
    
    """
    Data Prep for Model Inputs
    
    - standardization
    - replaces missing values with 0

    Parameters
    ----------
    df : dataframe
        data to be prepped
    elbow : [True, False]
        should a plot of WCSS be printed
    varlist : list
        column names  for input

    Returns
    -------
    X : dataframe
        subset of input df, prepped for analysis
    elbow graph (printed)
        

    """
    
    if not bystring:
        bystring='{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())
    
    if varlist:
        # subset dataframe
        df = df[varlist]
#         print("%d Columns loaded of %d attempted" % (df.shape[0], len(varlist)))

    # coerce non-int columns to numeric
    coerce_cols = df.dtypes[df.dtypes!='int64'].index.to_list()
    df[coerce_cols] = df[coerce_cols].apply(pd.to_numeric, errors='coerce')
#     print("%d Columns coerced to numeric" % (len(coerce_cols)) )
    
    # drop columns where are values are missing 
    cols_allmissing = df.isna().sum()[df.isna().sum()==3254].index.to_list()
    df = df.drop(cols_allmissing, axis = 1) 
#     print("%d Columns dropped due to all missing values" % (len(cols_allmissing)) )
    
    # Standardize the qx_list columns
    # NaNs are treated as missing values: disregarded in fit, and maintained in transform.
    scaler = StandardScaler().fit(df)
    X = df.copy()
    X[df.columns] = scaler.transform(df)
#     print("%d Columns scaled to mean=0 stdv=1" % (X.shape[0] ))
          
    # fill NA's with 0 
    X.fillna(0, inplace=True)
          
#     if elbow:
#         # Using WCSS-based (sum of dists b/w centroids & points for all clusters)...
#          # ... Elbow Method to find optimal number of clusters
#         wcss = []
#         for i in range(1, 11):
#             kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, 
#                             n_init=10, random_state=0)
#             kmeans.fit(X)
#             wcss.append(kmeans.inertia_)
#         plt.plot(range(1, 11), wcss)
#         plt.title('Elbow Method')
#         plt.xlabel('Number of clusters')
#         plt.ylabel('WCSS')
#         plt.savefig( outroot+'/Seg1_KNN'+str(nclusters)+'_'+bystring+'.pdf')
    
    return X

def latvar_pca (X, n_components, bystring=None, plots=False):
#     if not bystring:
#         bystring='{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())
    # Fitting KMeans w/ Optimal number of Clusters
    
    
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponents
             , columns = [f'principal component {i}' for i in range(1,1+n_components)])
    
    df_data_pca = pd.concat([principalDf, df_data], axis = 1)
            
    return df_data_pca, pca 

def pc1_analysis(pca, features, bystring, plot=False):
    
#     if not bystring:
#         bystring='{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())
    
    pc1_comp = pd.DataFrame(pca.components_[0],features)
    pc1_comp['abs'] = abs(pc1_comp[0])
    pc1_comp = pc1_comp.merge(df_varmap[['Variable','Label','Short Label']], left_index=True, right_on='Variable')
    pc1_comp = pc1_comp.sort_values(by='abs', ascending=False)
    pc1_comp['sign'] = np.sign(pc1_comp[0])
    pc1_comp['Variable_grp'] = [x.split('_')[0] for x in pc1_comp['Variable']] 
    
    grouped_abs = pc1_comp.groupby('Variable_grp')['abs'].agg(['sum','count','mean']).reset_index()
    grouped_abs = grouped_abs.sort_values('sum', ascending=False)
    grouped_abs = grouped_abs.set_index('Variable_grp')
    grouped_abs

    pc1_comp_vargrp = pc1_comp.groupby('Variable_grp').sign.value_counts().unstack().merge(grouped_abs,
                                           left_index=True, right_index=True).sort_values('mean', ascending=False)
        
    if plot:
        plt.plot(range(0, len(features)), pc1_comp['abs'])
        plt.title('Feature Importance for PC 1')
        plt.xlabel('Feature')
        plt.ylabel('Abs Weight')
        plt.savefig( outroot+'/latvar1_PCA1.pdf')
        
    return pc1_comp, pc1_comp_vargrp


if __name__ == "__main__":
    
    # arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--nclusters", "-nc", type=int,
#                         help="number of clusters for KNN")
#     parser.add_argument("--alpha", "-a", type=float,
#                         help="tolerance threshold for stat sig of attributes across clusters")
#     args = parser.parse_args()
#     print("set number of clusters for KNN %s" % args.nclusters)
#     print("tolerance threshold for stat sig of attributes across clusters %s" % args.alpha)
    
#     nclusters = args.nclusters
#     alpha =  args.alpha

    inroot = r"../data/input/07 Samsung UX Index - Web App Implementation/"
    fname_data = inroot + r"Samsung UX Index Survey_Data.csv"
    df_data = pd.read_csv(fname_data)
    fname_vaxmap = inroot + r"Samsung UX Index Survey_Datamap.xlsx"
    df_varmap = pd.read_excel(fname_vaxmap, header=1, sheet_name=0)
    df_valmap = pd.read_excel(fname_vaxmap, header=1, sheet_name=1)
    
    SuperLabels = ['upgradetrans','ECGexpect']
        'drivers', 'brandlovetrust', 'loyaltymetrics', 'overallquality',
                  'usagemetrics', 'activitiesximportance','activitiexsquality',
                  'activitiesxsatisfaction','activitiesxdrivers','upgradetrans','ECGexpect']

    
    SuperList = [[ 'att01_1','att01_2','att01_3','att02_1','att02_2','soc03','soc04_1',
               'soc04_2','soc04_3','ret06_1','qxexpectations' ], 
                 ['qxupgrade01_1', 'qxupgrade01_2', 'qxupgrade01_3', 'qxupgrade01_4', 
                  'qxupgrade01_5', 'qxupgrade01_6', 'qxupgrade01_7', 'qxupgrade01_8', 
                  'qxtransition_1']]
    
#     [ ['qxdrivers_'+str(i+1) for i in range(34)],
#                      [ x for x in df_data.columns if 'qxbrandx' in x ],
#                     ['qxadvocacy01_1',    'qxadvocacy02_1',    'qxretention_1',    'qxenrichment_1'],
#                     ['qxoverallxqualityxindicators_'+str(i+1) for i in range(4)],
#                     ['qxtime', 'qxcurrentxos',    'qxcurrentxstorage',    'qxcurrentxcarrier',    'qxunlocking',  'qxpreviousxbrand',    'qxtransition_1',],
#                     [ x for x in df_data.columns if 'qxactivitiesximportance' in x ],
#                     [ x for x in df_data.columns if 'qxactivitiesxqualityxindicators' in x ],
#                     [ x for x in df_data.columns if 'qxactivitiesxsatisfaction' in x ],
#                     [ x for x in df_data.columns if 'qxactivitiesxdrivers' in x ] ,
#                     ]

                 
    
    for label, varlist in zip(SuperLabels,SuperList):
        # Create Output Folder
        timestamp='{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())
        outroot = r"../data/output/"+timestamp+"_"+label

        try:
            os.mkdir(outroot)
        except OSError:
            print ("Creation of the directory %s failed" % outroot)
        else:
            print ("Successfully created the directory %s " % outroot)

        X = dataprep_X (df_data, elbow=True, varlist=varlist, bystring='elbow')
        features = X.columns

        df_data_pca, pca = latvar_pca (X, n_components=min(len(features),5), bystring=None, plots=False)
        
        print (pca.explained_variance_ratio_)
        
        numpy.savetxt( outroot+'/latvar1_expvar.csv', pca.explained_variance_ratio_, delimiter=",")

        pc1_comp, pc1_comp_vargrp = pc1_analysis(pca, features, bystring='x', plot=False)

        pc1_comp.to_csv(outroot+'/pc1_comp.csv', index=False)
        pc1_comp.to_csv(outroot+'/pc1_comp_vargrp.csv', index=False)
#         pc1_comp.to_csv(outroot+'/grouped_abs.csv', index=False)

        favorite_color = { "lion": "yellow", "kitty": "red" }

        pickle.dump( pca, open( outroot+'/pca.p', "wb" ) )

