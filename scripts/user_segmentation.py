# /usr/local/bin/python3
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from matplotlib.backends.backend_pdf import PdfPages
import datetime
import os
import itertools


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
        print("%d Columns loaded of %d attempted" % (df.shape[0], len(varlist)))

    # coerce non-int columns to numeric
    coerce_cols = df.dtypes[df.dtypes!='int64'].index.to_list()
    df[coerce_cols] = df[coerce_cols].apply(pd.to_numeric, errors='coerce')
    print("%d Columns coerced to numeric" % (len(coerce_cols)) )
    
    # drop columns where are values are missing 
    cols_allmissing = df.isna().sum()[df.isna().sum()==3254].index.to_list()
    df = df.drop(cols_allmissing, axis = 1) 
    print("%d Columns dropped due to all missing values" % (len(cols_allmissing)) )
    
    # Standardize the qx_list columns
    # NaNs are treated as missing values: disregarded in fit, and maintained in transform.
    scaler = StandardScaler().fit(df)
    X = df.copy()
    X[df.columns] = scaler.transform(df)
    print("%d Columns scaled to mean=0 stdv=1" % (X.shape[0] ))
          
    # fill NA's with 0 
    X.fillna(0, inplace=True)
          
    if elbow:
        # Using WCSS-based (sum of dists b/w centroids & points for all clusters)...
         # ... Elbow Method to find optimal number of clusters
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, 
                            n_init=10, random_state=0)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
        plt.plot(range(1, 11), wcss)
        plt.title('Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.savefig( outroot+'/Seg1_KNN'+str(nclusters)+'_'+bystring+'.pdf')
    
    return X

# Standardization of these columns?
def dataprep_Z (df, varlist, ohdict=None):
    """
    Data Prep for attribute variables for statistical significance testing 
    
    - 0 one hot encoding select varables
    (- standardization (?))
    - replaces missing values with 0

    Parameters
    ----------
    df : dataframe
        data to be prepped
    varlist : list
        column names for Z 
    ohdict : dictionary
             {string : list}
        variables and break points 
        for one-hot-encoding
        if None, encode each value
        

    Returns
    -------
    Z : dataframe

    """
    
    if ohdict:
        # one-hot encode variables
        for key, value in ohdict.items():
            if key in varlist:
                if not value:
                    for v in df[key].unique() :
                        df[key+'_'+str(v)] = np.where(df[key]==v, 1, 0)
                        varlist.extend([key+'_'+str(v)])
                else:
                    print (key, value)
                    df[key+'_X'] = ~pd.Series(df[key].isin(value))*1
                    varlist.extend([key+'_X'])
                    for v in value:
                        df[key+'_'+str(v)] = np.where(df[key]==v, 1, 0)
                        varlist.extend([key+'_'+str(v)])

                varlist.remove(key)

    Z = df[varlist]
    
    # fill NA's with 0 
    Z.fillna(0, inplace=True)
       
    
    return Z
    

def segm_kmeans (X, n_clusters, bystring=None):
    if not bystring:
        bystring='{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())
    # Fitting KMeans w/ Optimal number of Clusters
    kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
    pred = kmeans.fit_predict(X)
    
    with PdfPages(outroot+'/Seg1_KNN'+str(n_clusters)+'_'+bystring+'.pdf') as pdf:
        plt.ioff()
#         if 'pred' in X.columns:
#             cols = X.columns.drop(['pred'])
#         else:
#             cols = 
        for col in X.columns:
            plt.figure()
            fig=X.groupby(pred)[col].plot(kind='kde', 
                                            title=col, legend=True)[0].get_figure()
            pdf.savefig(fig)   
            
    return pred
    
def varimp (X, varlist, pred, c):
    """
    Calculation of Variable Importance for cluster memebership c (in pred) and data X

    Parameters
    ----------
    X : dataframe
        clustered to determine pred array
    c : int
        value in c indicating cluster 
        for calculating embership variable importance
   

    Returns
    -------
    Z : dataframe

    """
    
    # drop pred columns if they exist
    X = X[varlist]
    print(X.shape)
    
    # one-hot encoding of pred == c
    ohcluster = np.where(pred==c, 1, 0)
    print(ohcluster)
    X_vars = X.columns.values.tolist()
    y = ohcluster
    print(y.shape)
    
    # Logistic Regression to calculate coefficients
    m = LogisticRegression()
    m.fit(X, y)
    
    coefs = pd.DataFrame(list(zip(X.columns, m.coef_[0])))
    coefs.columns = ['Variable', 'coeff']

    # Fitt a random forest classifier
    rfc = RandomForestClassifier(random_state = 0, n_jobs = -1)
    rfc.fit(X, y)
    
    varimp = pd.DataFrame({'Variable':X.columns,
                           'imp': list(rfc.feature_importances_)}
                         ).sort_values('imp', ascending=False)
    
    varimp = varimp.merge(coefs[['Variable','coeff']], on='Variable', how='left')
    
    return varimp

def varimp_moreinfo (vi, df_varmap):

    varimp = vi.merge(df_varmap[['Variable', 'Label']], 
                          on='Variable', how='left')

    varimp['imp_rank'] = varimp['imp'].rank(ascending=False)
    varimp['coeff_rank'] = [abs(x) for x in varimp['coeff']]
    varimp['coeff_rank'] = varimp['coeff_rank'].rank(ascending=False)
    varimp['coeff_sign'] = np.where(varimp['coeff']<0,-1,1)

    varimp = varimp[['Variable','Label','imp','coeff','imp_rank','coeff_rank']]

    return varimp

def Zclust_stats (X, pred, Z, alpha, bystring=None):
    # # Examine statistical significant differences in demographics amongst segments. 
    
    clusters = np.unique(pred)
    if not bystring:
        bystring='{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())
    
    with PdfPages(outroot+'/Seg1_KNN'+str(len(clusters))+'_'+bystring+'.pdf') as pdf:
        plt.ioff()
        for col in Z.columns:
            if pd.api.types.is_numeric_dtype(df_data[col]):
                try:
                    plt.figure()
                    fig=Z.groupby(pred)[col].plot(kind='kde', title=col, legend=True)[0].get_figure()
                    pdf.savefig(fig)
                except:
                    print("Error: "+col)

    # Testing whether the populations have statistically sig diff in means (assuming N distr)
    # For two independent samples w/ potentially !=  variance, use Welch's t-test
    # scipy.stats.ttest_ind(cat1['values'], cat2['values'], equal_var=False)

    keys = ['clusterA','clusterB','stat','pvalue','interp']
    df_results = pd.DataFrame(columns=keys)
    for col in Z.columns:
        if pd.api.types.is_numeric_dtype(Z[col]):
            for x in itertools.product(clusters, clusters):
                s0 = pred==x[0]
                s1 = pred==x[1]

                g0 = Z[s0].dropna()[col]
                g1 = Z[s1].dropna()[col]
                
                # in testing that g0>g1, cA>cB
                # we'd like to reject null hypothesis of
                # i.e. we'd like pval/2 < alpha
                t_test_results = scipy.stats.ttest_ind(
                    g0,g1, equal_var=False)
                stat = t_test_results.statistic
                pvalue = t_test_results.pvalue
                
                if stat < 0 and pvalue < alpha/2:
                    interp = 'Reject H0 in favor of Ha: cluster %d < cluster %d' % (x[0],x[1] ) 
                elif stat < 0 and pvalue > alpha/2:
                    interp = 'Accept H0: cluster %d >= cluster %d' % (x[0],x[1] ) 
                elif stat > 0 and pvalue < alpha/2:
                    interp = 'Reject H0 in favor of Ha: cluster %d > cluster %d' % (x[0],x[1] ) 
                elif stat > 0 and pvalue > alpha/2:
                    interp = 'Accept H0: cluster %d <= cluster %d' % (x[0],x[1] ) 
                else:
                    interp = ''
            
                values = [ x[0], x[1], stat, pvalue, interp]
                
                
                df_results = df_results.append(pd.DataFrame(dict(zip(keys, values)), index=[0]))

                
                
    return df_results

           

if __name__ == "__main__":
    
    nclusters = 3
    
    # Create Output Folder
    datetime='{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())
    outroot = r"../data/output/"+datetime
    
    try:
        os.mkdir(outroot)
    except OSError:
        print ("Creation of the directory %s failed" % outroot)
    else:
        print ("Successfully created the directory %s " % outroot)

    inroot = r"../data/input/07 Samsung UX Index - Web App Implementation/"
    fname_data = inroot + r"Samsung UX Index Survey_Data.csv"
    df_data = pd.read_csv(fname_data)
    fname_vaxmap = inroot + r"Samsung UX Index Survey_Datamap.xlsx"
    df_varmap = pd.read_excel(fname_vaxmap, header=1, sheet_name=0)
    df_valmap = pd.read_excel(fname_vaxmap, header=1, sheet_name=1)
    
    varlist = ['qxdrivers_'+str(i+1) for i in range(34)]
    
    X = dataprep_X (df_data, elbow=True, varlist=varlist, bystring='elbow')
    
    # Geographic Grouping of state vars? d5x1
    varlist_Z  = [ 'd1',    'd3_1',    'd3_2',    'd3_3',    'd3_4',    'd4', 'd6',    'd7_1',    
                'd7_2',    'd7_3',    'd7_4',    'd7_5',    'd7_97',    'd7_99',]
    
    ohdict = {'d4' :  [7], 
              'd1' : None, 
              'hidagemodels' : None,
            'qxcurrentxmodel': None,
             'hbrand' : None,
             'hmodelquota' : None,
             'd3_1': None,
             'hmodelquota_reordered' : None,
             'Empowered_Customer_Groups' : None}

    Z = dataprep_Z (df_data, varlist=varlist_Z, ohdict=ohdict)
     
    pred = segm_kmeans (X, nclusters, bystring='byattitudes')
  
    vi = varimp (X, varlist, pred, 2)

    vi_mi = varimp_moreinfo (vi, df_varmap)
    vi_mi.to_csv(outroot+'/Seg1_KNN'+str(nclusters)+'_vimi.csv', index=False)
    
    Zcst = Zclust_stats(X, pred, Z, alpha=.05, bystring='byattributes')
    Zcst.to_csv(outroot+'/Seg1_KNN'+str(nclusters)+'_zclust'+str(alpha)+'.csv', index=False)
