# PPM Utilities Module

#from ..spc import *
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import itertools
import pandas as pd
import zipfile
from scipy.cluster import hierarchy as hc
import scipy
from pdpbox import pdp
from plotnine import *
from collections import Counter
import random

# from pandas.plotting import scatter_matrix
from scipy.stats.stats import pearsonr

from sklearn_pandas import DataFrameMapper
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer as Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import make_scorer


# from scipy.stats import randint
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import PolynomialFeatures

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model.base import *
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import statsmodels.formula.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from statsmodels.stats.diagnostic import normal_ad as ntest

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import datetime
from collections import namedtuple

# For outlier analysis
from pyod.models.cblof import CBLOF
# For dimention reduction
from sklearn.manifold import TSNE
from sklearn import manifold
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

#For TreeInterpreter
from io import StringIO
from sklearn.tree import export_graphviz
import pydotplus
import os
import json

from .utils import *

os.environ["PATH"] += os.pathsep + 'D:/Python/graphviz-2.38/release/bin'

random_state = np.random.RandomState(42)

def analyze_patterns(ppmdata, directory,
                       vif_thresh, outliers_fraction, build_model, save=True):

    continue_execution = True
    x_feat_nom = []
    x_feat_ord = []
    x_feat_num = []
    y_feat = []
    row_id = []
    warning_messages = []
    outlier_rows = np.empty(shape=(0, 0))
    outlier_id = 0


     #Added for outlier functionality    
    ################################
    
    # Define seven outlier detection tools to be compared
    classifiers = {
        'Cluster-based Local Outlier Factor (CBLOF)':CBLOF(contamination=outliers_fraction,check_estimator=False, random_state=42)}
    x_feat_nc =[]
    x_data_outlier =[]
    rowid_outlier =[]    
    
    ################################
    for col in ppmdata.columns:  # Classify X's into Numerical/Nominal
        # or Categorical data
        if col[:3] == "XN_":
            x_feat_nom.append(col)
        elif col[:3] == "XO_":
            x_feat_ord.append(col)
        elif col[:2] == "X_":
            x_feat_num.append(col)
            x_data_outlier.append(col) # for outlier analysis   
        elif col[:2] == "Y_":
            y_feat.append(col)
            x_data_outlier.append(col) # for outlier analysis   
        elif col[:3] == "ID_":
            row_id.append(col)

        if col[:3] != "ID_":
            if is_string_dtype(ppmdata[col]):
                if col[:3] != "XN_" and col[:3] != "XO_":
                    warning_messages.append(col + " has wrong data type! Execution will be aborted.")
                    continue_execution = False
            elif is_numeric_dtype(ppmdata[col]):
                if col[:2] != "X_" and col[:2] != "Y_":
                    warning_messages.append(col + " has wrong data type! Execution will be aborted.")
                    continue_execution = False

    if len(x_feat_nom) == 0:
        warning_messages.append("No Nominal X's found in the dataset")
    if len(x_feat_ord) == 0:
        warning_messages.append("No Ordinal X's found in the dataset")
    if len(x_feat_num) == 0:
        warning_messages.append("No Numerical X's found in the dataset")
    if len(y_feat) == 0:
        warning_messages.append("No Y found in the dataset")
    if len(row_id) == 0:
        warning_messages.append("No ID found in the dataset")

    if continue_execution:
        for col in ppmdata.columns:  # For Missing Values
            if ppmdata[col].isnull().values.ravel().sum() > 0:
                if col in x_feat_num:
                    imputer = Imputer(missing_values='NaN', strategy='mean')
                    ppmdata[col] = imputer.fit_transform(ppmdata[[col]]).ravel()
                elif col in y_feat:
                    imputer = Imputer(missing_values='NaN', strategy='mean')
                    ppmdata[col] = imputer.fit_transform(ppmdata[[col]]).ravel()
                else:
                    mapper = DataFrameMapper([([col], Imputer())])
                    ppmdata[col] = mapper.fit_transform(ppmdata)

    if len(x_feat_num) > 0 and continue_execution and build_model:
        # For plotting joint plots, histograms, pair plots
        sns.set(font_scale=2)
        #sns.set()
        sns.set_context('poster')
        for dep_feat in x_feat_num:
            sns.jointplot(ppmdata[dep_feat],
                          ppmdata[y_feat[0]], kind="kde", dropna=False)
            if save:
                save_fig(directory,"UD_Jointplot_for_" + dep_feat)

        sns.set_context("paper", font_scale=3, rc={"font.size": 8, "axes.labelsize": 5})

        plt.style.use('ggplot')
        orig_columns = ppmdata.columns
        ppmdata.columns = cleanup_header(list(ppmdata.columns))

        sns.set(font_scale=1.3)
        sns.pairplot(ppmdata[ppmdata.columns], size=2.5)
        plt.tight_layout()
        if save:
            save_fig(directory,'UD_Scatter Plots', tight_layout=False)
        ppmdata.columns = orig_columns

        sns.set_context("paper", font_scale=3, rc={"font.size": 8, "axes.labelsize": 5})
        heatmap_features = x_feat_num + y_feat
        cm = np.corrcoef(ppmdata[heatmap_features].values.T)
        sns.set(font_scale=1.5)
        sns.heatmap(cm, annot=True, cbar=False,cmap="Blues",fmt='.2f',annot_kws={'size':15},
                    xticklabels=heatmap_features, yticklabels=heatmap_features)
        plt.yticks(rotation=0)
        plt.xticks(rotation=90)
        if save:
            save_fig(directory,'UD_Correlation_Plot', tight_layout=True)
    plt.close()

    x_feat_num_corr = []
    if len(x_feat_num) > 1 and continue_execution:
        variables = np.arange(len(x_feat_num))
        d = [x_feat_num[ix] for ix in list(variables)]
        vif_ = [vif(ppmdata[d].values, ix) for ix in range(len(d))]
        vif_report_data = list(zip(d, vif_))
        dropped = True
        while dropped:
            dropped = False
            d = []
            d = [x_feat_num[ix] for ix in list(variables)]
            if len(d) > 1:
                vif_ = [vif(ppmdata[d].values, ix) for ix in range(len(d))]
                maxloc = vif_.index(max(vif_))
                if float(max(vif_)) > float(vif_thresh):
                    x_feat_num_corr.append(x_feat_num[variables[maxloc]])
                    variables = np.delete(variables, maxloc)
                    dropped = True
        x_feat_num_nc = [x_feat_num[ix] for ix in list(variables)]
    else:
        x_feat_num_nc = x_feat_num.copy()
        
        #for outlier functionality
    ################################    
     
    x_data_outlier = pd.DataFrame(ppmdata[x_data_outlier])          
    rowid_outlier = pd.DataFrame(ppmdata[row_id])
   
    if len(x_feat_ord) > 0:  # To transform Ordinal X
        ord_x_mapper = [(x_feat_ord[i], LabelEncoder()) for i, col in enumerate(x_feat_ord)]    
        ord_encoder = DataFrameMapper(ord_x_mapper)
        x_data_ord = ord_encoder.fit_transform(ppmdata)                                
        x_feat_nc = np.append(x_feat_nc, x_feat_ord)        
        x_data_ord_outlier_df = pd.DataFrame(x_data_ord)        
        x_data_ord_outlier_df.columns=x_feat_ord
        x_data_outlier= x_data_outlier.join(x_data_ord_outlier_df)       
        
    if len(x_feat_nom) > 0:  # To transform Nominal X
        x_data_nom = np.empty((len(ppmdata[x_feat_nom[0]]), 1))
        nom_encoder = []
        for i, col in enumerate(x_feat_nom):        
            dummies = pd.get_dummies(ppmdata[x_feat_nom[i]]).rename(columns=lambda x: x_feat_nom[i] + ":" + str(x))           
            #x_data_nom = np.append(x_data_nom, np.array(dummies), 1)
            x_data_outlier= x_data_outlier.join(dummies)
            x_feat_nc = np.append(x_feat_nc, dummies.columns.values)    
    
    X = x_data_outlier
    
    ################################

    #X = ppmdata[x_feat_num].values
    
    #if len(x_feat_num_nc) > 0 and continue_execution:
    if len(x_data_outlier.columns) > 0 and continue_execution:
        # For finding outliers
        n_neighbours = int(np.ceil(np.sqrt(len(X))))
        clf = LocalOutlierFactor(n_neighbors=n_neighbours)
        clf.fit(X)
        x_outlierscore1 = clf.negative_outlier_factor_
        outlier_rows = np.array(np.where(x_outlierscore1 <= -1.5), dtype=int)
        if len(outlier_rows.T) > outliers_fraction * len(X):
            clf = LocalOutlierFactor(n_neighbors=n_neighbours,
                                     contamination=outliers_fraction)
            outlier_rows = clf.fit_predict(X)
            outlier_rows = np.array(np.where(outlier_rows < 0), dtype=int)
        outlier_rows = np.array(outlier_rows[0], dtype=int)
        
        if len(outlier_rows) == 0:
            print("** No Outliers Found**")
        else:
            #rowid_outlier
            #print("** Inside Plot Logic**")
            n_samples, n_features = X.shape
            X_tsne = manifold.TSNE(n_components=2, init='pca').fit_transform(X)
            feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
            df = pd.DataFrame(X,columns=feat_cols)
            df['label'] = rowid_outlier
            rndperm = np.random.permutation(df.shape[0])
            n_sne=len(df)
            df_tsne = None
            df_tsne= df.loc[rndperm[:n_sne],:].copy()
            df_tsne['x-tsne'] = X_tsne[:,0]
            df_tsne['y-tsne'] = X_tsne[:,1]
            scaler = MinMaxScaler(feature_range=(0, 1))
            df_tsne[['x-tsne','y-tsne']] = scaler.fit_transform(df_tsne[['x-tsne','y-tsne']])
            
            X1_Plot = df_tsne['x-tsne'].values.reshape(-1,1)
            X2_Plot = df_tsne['y-tsne'].values.reshape(-1,1)
            X_Plot = np.concatenate((X1_Plot,X2_Plot),axis=1)            
            
            #Outlier Plot
            xx , yy = np.meshgrid(np.linspace(0,1 , 200), np.linspace(0, 1, 200))
    
            for i, (clf_name, clf) in enumerate(classifiers.items()):
            
                clf.fit(X_Plot)
                
                # predict raw anomaly score
                scores_pred = clf.decision_function(X_Plot) * -1
                
                d = pd.DataFrame(np.zeros((len(X_Plot), 1)),dtype=int)
                
                for i in range(0, len(outlier_rows)):        
                    d.loc[outlier_rows[i],0]=1
                
                # prediction of a datapoint category outlier or inlier
                y_pred = clf.predict(X_Plot)    
                n_inliers = len(y_pred) - np.count_nonzero(y_pred)
                n_outliers = np.count_nonzero(y_pred == 1)                                    
                
                plt.figure(figsize=(10, 10))

                # copy of dataframe
                dfx = df_tsne
                dfx['outlier'] = y_pred.tolist()                
                dfx['trueoutlier'] = d                
                              
                if (len(dfx['label'][dfx['trueoutlier'] == 1])) > (len(dfx['label'][dfx['outlier'] == 1])):                   
                    # Logic 1 - More outliers
                    Global = (len(dfx['label'][dfx['trueoutlier']==1])) - (len(dfx['label'][dfx['outlier'] == 1]))                    
                    
                   # for i in xrange(len(dfx),0,-1):
                    #    if Global !=0:
                     #       if dfx['trueoutlier'][i] == 1:
                      #          dfx['trueoutlier'][i] = 2 #Global
                       #         Global = Global -1
                else:
                    # Logic 2 - Less outliers
                    Global = (len(dfx['label'][dfx['outlier']==1])) - (len(dfx['label'][dfx['trueoutlier'] == 1]))                    
                    for i in range (0,len(dfx)):   
                        if Global !=0:
                            if dfx['outlier'][i] == 1:                                
                                dfx['outlier'][i] = 2 #Global
                                Global = Global -1                        
                        
                # IX1 - inlier feature 1,  IX2 - inlier feature 2
                IX1 =  np.array(dfx['x-tsne'][dfx['outlier'] == 0]).reshape(-1,1)
                IX2 =  np.array(dfx['y-tsne'][dfx['outlier'] == 0]).reshape(-1,1)
                IID =  np.array(dfx['label'][dfx['trueoutlier'] == 0]).reshape(-1,1)

                # OX1 - outlier feature 1, OX2 - outlier feature 2
                OX1 =  dfx['x-tsne'][dfx['outlier'] == 1].values.reshape(-1,1)
                OX2 =  dfx['y-tsne'][dfx['outlier'] == 1].values.reshape(-1,1)
                OID =  np.array(dfx['label'][dfx['trueoutlier'] == 1]).reshape(-1,1)

                # threshold value to consider a datapoint inlier or outlier
                threshold = stats.scoreatpercentile(scores_pred,100 * outliers_fraction)


                # decision function calculates the raw anomaly score for every point
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1

                Z = Z.reshape(xx.shape)

                # fill blue map colormap from minimum anomaly score to threshold value
                plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),cmap=plt.cm.Blues_r)

                # draw red contour line where anomaly score is equal to thresold
                a = plt.contour(xx, yy, Z, levels=[threshold],linewidths=2, colors='red')

                # fill orange contour lines where range of anomaly score is from threshold to maximum anomaly score
                plt.contourf(xx, yy, Z, levels=[threshold, Z.max()],colors='orange')

                b = plt.scatter(IX1,IX2, c='white',s=50, edgecolor='k')

                c = plt.scatter(OX1,OX2, c='black',s=50, edgecolor='k')     
                
                #plt.axis('tight')
                                                                                
                for i, label in enumerate(OID):                      
                    label = np.array2string(label)
                    label = label.strip('[')
                    label = label.strip(']')
                    label = label.strip("'")
                    plt.annotate(label, (OX1[i], OX2[i]))
                    
                # loc=2 is used for the top left corner 
                plt.legend(
                    [a.collections[0], b,c],
                    ['learned decision function', 'inliers','outliers'],
                    prop=matplotlib.font_manager.FontProperties(size=15),
                    loc=2)
                
                plt.xlim((0, 1))
                plt.ylim((0, 1))
                plt.title(clf_name+"\n")
                
                plt.tight_layout()
                plt.gcf().set_size_inches(20,15)
                
                Outlier_Rows=[]
                Outlier_Rows = pd.DataFrame(df_tsne.loc[df_tsne.trueoutlier==1,['label']])
                outlierdata_report=pd.DataFrame(columns=ppmdata.columns)
                colname = ppmdata.columns[-1]
                Outlier_Rows= pd.DataFrame(Outlier_Rows['label'].values,columns=['label'])
                
                for i in range(len(Outlier_Rows)):                
                    outlierdata_report = outlierdata_report.append(ppmdata.loc[ppmdata[colname]==Outlier_Rows['label'][i]])
                    print("Dbug...... Outlier logic -- -Colname", colname)
                    print("Dbug...... Outlier logic -- -Colname", outlierdata_report[colname])
                    #outlierdata_report[colname] = outlierdata_report[colname].str.wrap(15)
                
                if save:
                    save_fig(directory,"Outliers Graph - " + clf_name,tight_layout=False)
                    file = os.path.join(directory, "Outliers_table.jpg")
                    #save_table(outlierdata_report, file, font_size=15, size='not default',header_columns=0, col_width=4.0, row_height=1.5,text_message='',text_title='List of outliers in the given dataset')
                    
    featgroup = namedtuple('featgroup', 'all,all_nc, all_model,num,num_nc,num_corr,ord,nom,target')
    feature_group = featgroup(None,None, None,x_feat_num,x_feat_num_nc, x_feat_num_corr, x_feat_ord, x_feat_nom, y_feat)

    return warning_messages, feature_group, outlier_rows, continue_execution


