#from ..spc import *
import matplotlib as mpl
import numpy as np

mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# from pandas.plotting import scatter_matrix
from scipy.stats.stats import pearsonr

from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder

# from scipy.stats import randint

from collections import namedtuple

# For outlier analysis
# For dimention reduction

#For TreeInterpreter
import os

from .utils import *
#---TODO Below graphviz path needs to be replaced with variable ---
#os.environ["PATH"] += os.pathsep + 'E://graphviz//release//bin'

random_state = np.random.RandomState(42)


def select_model(ppmdata, featgroup, datagroup, summary_data_file,
                 directory, options, save=True):

    model_to_try = namedtuple('model_to_try', 'model_name,func,params,sensitive_to_correlation')
    model_search_space = []
    model_selection_ind = False

    model_name = 'Random Forest'
    params = {'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10], 'n_estimators': [10, 20, 30, 40, 60]}
    func = RandomForestRegressor()
    sensitive_to_correlation = False
    model_search_space.append(model_to_try(model_name, func, params, sensitive_to_correlation))

    model_name = 'Linear Regression'
    func = LinearRegression()
    params = {}
    sensitive_to_correlation = True
    model_search_space.append(model_to_try(model_name, func, params, sensitive_to_correlation))

    #if options.try_2d_polynomial:
    #    model_name = '2d-Poly Regression'
    #    func = PolynomialRegression(degree=2)
    #    params = {}
    #    sensitive_to_correlation = True
    #    model_search_space.append(model_to_try(model_name, func, params, sensitive_to_correlation))

    if options.try_ridge:
        model_name = 'Ridge'
        params = {'alpha': [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 100, 300, 1000]}
        sensitive_to_correlation = False
        func = Ridge()
        model_search_space.append(model_to_try(model_name, func, params, sensitive_to_correlation))

    if options.try_lasso:
        model_name = 'Lasso'
        params = {'alpha': [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 100, 300, 1000]}
        func = Lasso()
        sensitive_to_correlation = False
        model_search_space.append(model_to_try(model_name, func, params, sensitive_to_correlation))

    if options.try_decision_trees:
        model_name = 'Decision Tree'
        params = {'max_depth': [2, 3, 4, 5, 6, 8, 10]}
        func = DecisionTreeRegressor()
        sensitive_to_correlation = False
        model_search_space.append(model_to_try(model_name, func, params, sensitive_to_correlation))

        #Added for TreeInterpreter
        get_TreeInterpreter(datagroup, featgroup, directory, save)


    if options.try_adaboost:
        model_name = 'AdaBoost'
        params = {'n_estimators': [10, 20, 30, 50, 70, 100, 120, 150, 200, 300]}
        func = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5))
        sensitive_to_correlation = False
        model_search_space.append(model_to_try(model_name, func, params, sensitive_to_correlation))

    if options.try_support_vector:
        model_name = 'SVR'
        params = {'gamma': [0.01, 0.03, 0.1, 0.3, 0.6, 1, 3, 5, 10, 30, 100],
                    'C': [1e5, 1e3, 1e2, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001],
                    'kernel': ['rbf']}
        func = SVR()
        sensitive_to_correlation = False
        model_search_space.append(model_to_try(model_name, func, params, sensitive_to_correlation))

    if options.try_neural_networks:
        model_name = 'Neural Network'
        #params = {'activation': ['relu'], 'solver': ['sgd', 'adam'],
        #          'learning_rate': ['constant', 'invscaling', 'adaptive']}
        #func = MLPRegressor(hidden_layer_sizes=1, max_iter=1000)

        #---Modified for Neural network tuning ---
        params = {'activation': ['logistic'], 'solver': ['lbfgs','sgd', 'adam'],
                            'learning_rate': ['constant', 'invscaling', 'adaptive']}
        func = MLPRegressor(hidden_layer_sizes=(2), max_iter=200, random_state=9)
        sensitive_to_correlation = False
        model_search_space.append(model_to_try(model_name, func, params, sensitive_to_correlation))

    if options.try_bayesian:
        model_name = 'Bayesian Regression'
        func = BayesianRidge()
        params = {}
        sensitive_to_correlation = False
        model_search_space.append(model_to_try(model_name, func, params, sensitive_to_correlation))

    fitted_model, fitment_summary = fit_models(model_search_space, datagroup,featgroup,directory)

    fitment_report = pd.DataFrame.from_records(fitment_summary, columns=['Model', 'MSE_Train', 'MSE_Test', 'R_Square_Train', 'R_Square_Test'])

    if float(fitment_report.iloc[fitment_report.MSE_Test.values.argmin()]['R_Square_Test']) > 0.75:
        best_model = fitted_model[fitment_report.Model[fitment_report.MSE_Test.values.argmin()]]
        model_selection_ind = True
        best_model_name = fitment_report.iloc[fitment_report.MSE_Test.values.argmin()]['Model']
        text_message = "Selected Model : " + best_model_name
        plot_residuals_predictions(best_model, directory, save=True)
    else:
        text_message = "Could not Select a Model with Adequate Fit"
        best_model = fitted_model[fitment_report.Model[fitment_report.MSE_Test.values.argmin()]]

    print(text_message)

    if save:
        file = os.path.join(directory, summary_data_file)
        save_table(fitment_report, file, header_columns=0, col_width=4.0, bbox=[0, 0.1, 1, 1], text_message=text_message)


    return best_model, model_selection_ind

def fit_models(model_search_space,datagroup,featgroup,directory):

    fitment_summary = []
    fitted_models = {}
    splits = datagroup.y_train.shape[0]
    search_space_list = [(model_search_space[i].model_name, model_search_space[i].func, model_search_space[i].params,
                          model_search_space[i].sensitive_to_correlation) for i in range(len(model_search_space))]
    feature_imporances = []

    for model_name, func, params, sensitive_to_correlation in search_space_list:
        print("Fitting ", model_name)
        if sensitive_to_correlation:
            x_train = datagroup.x_train_nc
            y_train = datagroup.y_train_nc
            x_test = datagroup.x_test_nc
            y_test = datagroup.y_test_nc
            x_all = datagroup.x_all_nc
            y_all = datagroup.y_all
        else:
            x_train = datagroup.x_train
            y_train = datagroup.y_train
            x_test = datagroup.x_test
            y_test = datagroup.y_test
            x_all = datagroup.x_all
            y_all = datagroup.y_all

        if len(params) == 0:
            mdl = func.fit(x_train, y_train)
        else:
            gs = GridSearchCV(func, param_grid=params, cv=splits,
                              scoring=make_scorer(get_map_error, greater_is_better=False))
            gs.fit(x_train, y_train)
            mdl = gs.best_estimator_
        mdl.model_name = model_name
        errors = evaluate_reg(model_name, mdl, x_train, y_train, x_test, y_test, directory)
        fitment_summary.append((model_name, errors.mse_train, errors.mse_test, errors.r2_train, errors.r2_test))
        if model_name == 'Random Forest':
            feature_importances = list(zip(featgroup.all, gs.best_estimator_.feature_importances_))
        mdl = mdl.fit(x_all, y_all)
        mdl = update_model_attributes(mdl, model_name, featgroup, x_all,
                                y_all, x_train, y_train, x_test, y_test,feature_importances,sensitive_to_correlation)
        print("@2 overall mape..", get_map_error(y_all,predict(mdl,x_all)))


        fitted_models[model_name] = mdl

    return fitted_models, fitment_summary

def update_model_attributes(model,name,featgroup,x_all,y_all,x_train,y_train,x_test,y_test,feature_importances,
                            sensitive_to_correlation):

    model = model.fit(x_all,y_all)
    model.name = name
    if sensitive_to_correlation:
        model.features = featgroup.all_nc
        model.features_num = featgroup.num_nc
    else:
        model.features = featgroup.all
        model.features_num = featgroup.num
    model.features_ord = featgroup.ord
    model.features_nom = featgroup.nom
    model.target = featgroup.target
    model.x_all = x_all
    model.y_all = y_all
    model.x_train = x_train
    model.y_train = y_train
    model.x_test = x_test
    model.y_test = y_test

    if len(feature_importances) > 0 :
        model.feature_importances = np.array([i for f,i in feature_importances if f in model.features])

    return model

def plot_residuals_predictions(best_model, directory, save=True):


    SMALL_SIZE = 24
    MEDIUM_SIZE = 36
    BIGGER_SIZE = 48
    model_name = best_model.model_name
    x_train = best_model.x_train
    y_train = best_model.y_train
    x_test = best_model.x_test
    y_test = best_model.y_test
    x_all = best_model.x_all
    y_all = best_model.y_all
    y_train_predicted = predict(best_model, x_train)
    y_test_predicted = predict(best_model, x_test)
    y_all_predicted = predict(best_model,x_all)

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.style.use('ggplot')

    plt.figure()
    plt.gcf().set_size_inches(20,15)
    plt.subplot(211)
    plt.scatter(y_train_predicted, y_train_predicted - y_train,
                c='steelblue', marker='o', s=80, edgecolor='white',
                label='Training data')
    plt.xlabel('Predicted Train values',fontsize='xx-large')
    plt.ylabel('Residuals',fontsize='xx-large')
    xmn = 0.9 * np.min(y_train_predicted)
    xmx = 1.1 * np.max(y_train_predicted)
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=xmn, xmax=xmx, color='black', lw=0.5)
    plt.xlim([xmn, xmx])

    plt.subplot(212)

    plt.scatter(y_test_predicted, y_test_predicted - y_test,
                c='limegreen', marker='s', s=80, edgecolor='white',
                label='Test data')
    xmn = 0.9 * np.min(y_test_predicted)
    xmx = 1.1 * np.max(y_test_predicted)
    plt.xlabel('Predicted Test values',fontsize='xx-large')
    plt.ylabel('Residuals',fontsize='xx-large')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=xmn, xmax=xmx, color='black', lw=0.5)
    plt.xlim([xmn, xmx])
    if save:
        save_fig(directory,"MS_Residuals_of_" + model_name)

    plt.figure()
    plt.gcf().set_size_inches(20, 15)
    plt.xlabel('Predicted values', fontsize='xx-large')
    plt.ylabel('Actual Values', fontsize='xx-large')
    sns.regplot(y=y_all, x=y_all_predicted)
    if save:
        save_fig(directory,"MS_Prediction_vs_Actuals_of_" + model_name)

    return

def evaluate_reg(model_name, mdl, ppmtrx_prepared, ppmtr_y, ppmtsx_prepared,
                 ppmts_y, directory):
    np.random.seed(42)

    if model_name == '2d-Poly Regression':
        poly = PolynomialFeatures(degree=2)
        ppmtsx_prepared = poly.fit_transform(ppmtsx_prepared)
        ppmtrx_prepared = poly.fit_transform(ppmtrx_prepared)
    ppmtry_predicted = mdl.predict(ppmtrx_prepared)
    ppmtsy_predicted = mdl.predict(ppmtsx_prepared)

    error_summary = namedtuple('error_summary','mse_train,mse_test,r2_train,r2_test,'
                                               'mape_train,mape_test')
    #error_summary = namedtuple('error_summary', 'mse_train,mse_test,r2_train,r2_test')

    error_summary.mse_train = format(mean_squared_error(ppmtr_y, ppmtry_predicted),'2.9f')
    error_summary.mse_test = format(np.clip(mean_squared_error(ppmts_y, ppmtsy_predicted),-100,100),'2.9f')
    error_summary.r2_train = format(np.clip(r2_score(ppmtr_y, ppmtry_predicted),-100,100),'2.2f')
    error_summary.r2_test = format(np.clip(r2_score(ppmts_y, ppmtsy_predicted),-100,100),'2.2f')
    error_summary.mape_train = format(get_map_error(ppmtr_y, ppmtry_predicted),'2.6f')
    error_summary.mape_test = format(get_map_error(ppmts_y, ppmtsy_predicted),'2.6f')

    return error_summary

def mlr_regression(x_data_train, y_data_train, x_data_test):
    # Building optimal model using Backward Elimination
    x_data_train = np.append(arr=np.ones((len(x_data_train[:, 0]), 1)).astype(int), values=x_data_train, axis=1)
    x_temp = x_data_train[:, :]
    count = 0  # Since we do not wish to eliminate any columns in the first run
    model_final = 0  # Boolean variable to exit the loop
    max_index = len(x_data_train[0, :]) + 1
    while model_final == 0:
        if count > 0:
            x_temp = np.delete(x_temp, max_index, axis=1)
        ols_regressor = sm.OLS(endog=y_data_train, exog=x_temp)
        regressor_ols = ols_regressor.fit()
        count += 1
        regressor_pvalues = regressor_ols._results.pvalues
        regressor_pvalues = regressor_pvalues[1:]  # Ignore the constant
        max_value = regressor_pvalues.max()
        if max_value > 0.05:
            max_index = regressor_pvalues.argmax() + 1
        else:
            model_final = 1
    regressor_ols.summary()
    indices = []
    for i in range(len(list(x_temp.T))):
        for j in range(len(list(x_data_train.T))):
            if (list(x_temp.T)[i] == list(x_data_train.T)[j]).all():
                indices.append(j)
    x_data_test = np.append(arr=np.ones((len(x_data_test[:, 0]), 1)).astype(int), values=x_data_test, axis=1)
    x_data_test = x_data_test.T[indices].T
    return regressor_ols, x_temp, x_data_test


class PolynomialRegression():
    def __init__(self, fit_intercept=True, normalize=False, copy_X=True,
                 n_jobs=1, degree=2):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.degree = degree

    def fit(self, X, y, sample_weight=None):
        poly = PolynomialFeatures(degree=self.degree)
        X = poly.fit_transform(X)
        n_jobs_ = self.n_jobs
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
                         y_numeric=True, multi_output=True)

        if sample_weight is not None and np.atleast_1d(sample_weight).ndim > 1:
            raise ValueError("Sample weights must be 1D array or scalar")

        X, y, X_offset, y_offset, X_scale = self._preprocess_data(
            X, y, fit_intercept=self.fit_intercept, normalize=self.normalize,
            copy=self.copy_X, sample_weight=sample_weight)

        if sample_weight is not None:
            # Sample weight can be implemented via a simple rescaling.
            X, y = _rescale_data(X, y, sample_weight)

        if sp.issparse(X):
            if y.ndim < 2:
                out = sparse_lsqr(X, y)
                self.coef_ = out[0]
                self._residues = out[3]
            else:
                # sparse_lstsq cannot handle y with shape (M, K)
                outs = Parallel(n_jobs=n_jobs_)(
                    delayed(sparse_lsqr)(X, y[:, j].ravel())
                    for j in range(y.shape[1]))
                self.coef_ = np.vstack(out[0] for out in outs)
                self._residues = np.vstack(out[3] for out in outs)
        else:
            self.coef_, self._residues, self.rank_, self.singular_ = \
                linalg.lstsq(X, y)
            self.coef_ = self.coef_.T

        if y.ndim == 1:
            self.coef_ = np.ravel(self.coef_)
        self._set_intercept(X_offset, y_offset, X_scale)
        return self

def significant_sp_x(best_model):

    sp_ix = [i for i, s in enumerate(best_model.features) if "_SP_" in s];
    sp_name = [n for n in best_model.features if "_SP_" in n]
    sp_imp = best_model.feature_importances[sp_ix]

    return sp_name[sp_imp.argmax()]

def check_marginal_dependency(best_model, directory, feat=None, clusters=None, save=True):

    if not save:
        return

    if best_model.name == "2d-Poly Regression":
        print("Marginal Depndency is not available for Poly Regression")
        return

    x_data_all_df = pd.DataFrame(best_model.x_all, columns=best_model.features)
    x_names_all = best_model.features

    for f in best_model.features_num:
        feat_name = f
        p = pdp.pdp_isolate(best_model, x_data_all_df, x_names_all, feat_name)
        SMALL_SIZE = 25
        MEDIUM_SIZE = 20
        BIGGER_SIZE = 25

        plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        pdp.pdp_plot(p, f, plot_lines=True,
                            cluster=clusters is not None, n_cluster_centers=clusters)
        file = os.path.join(directory,"MI_Partial_Dependency_Plot_For_" + f + ".jpg")
        plt.gcf().set_size_inches(20,15)
        plt.savefig(file, format='jpg', dpi=None, orientation='landscape')
        plt.close()

    return


def check_feature_importance(best_model, feature_importance_data_file, directory, save=True):
    if not save:
        return
    x_names_all = cleanup_header(list(best_model.features))
    fi = pd.DataFrame({'Parameter': x_names_all, 'importance': best_model.feature_importances}).sort_values('importance', ascending=False)
    file = os.path.join(directory,feature_importance_data_file)
    fi = np.round(fi,decimals=4)
    save_table(fi,file,col_width=8)


    return

def check_feature_redundancy(best_model, feature_redundancy_chart_file, directory, save=True):

    if not save:
        return

    x_data_all_df = pd.DataFrame(best_model.x_all, columns=best_model.features)

    corr = np.round(scipy.stats.spearmanr(x_data_all_df).correlation, 4)
    corr_condensed = hc.distance.squareform(1 - corr)
    z = hc.linkage(corr_condensed, method='average')
    fig = plt.figure(figsize=(40, 30))
    x_names_all = cleanup_header(list(best_model.features))
    hc.dendrogram(z, labels=x_names_all, orientation='left', leaf_font_size=30)

    file = os.path.join(directory,feature_redundancy_chart_file)
    plt.gcf().set_size_inches(40, 30)
    plt.savefig(file, format='jpg', dpi=None, orientation='landscape')
    plt.close()

    return

	
def predict_for_new_data(best_model, directory, featgroup,
                         ppmdata, newdata, sigma, prediction_output_file, save=True):

    x_names_nom, x_names_ord, x_names_num_model = \
        (best_model.features_nom, best_model.features_ord, best_model.features_num)

    X_original = ppmdata[x_names_num_model].values
    encoder = StandardScaler()
    X_original = encoder.fit_transform(X_original)
    X_pred = newdata[x_names_num_model].values
    X_pred = encoder.transform(X_pred)

    if len(x_names_ord) > 0:
        ord_x_mapper = [(x_names_ord[i], LabelEncoder()) for i, col in enumerate(x_names_ord)]
        mapper = DataFrameMapper(ord_x_mapper)
        mapper.fit_transform(ppmdata)
        Ord_X = mapper.transform(newdata)
        X_pred = np.append(X_pred, Ord_X, 1)

    if len(x_names_nom) > 0:
        Nom_X = np.empty((len(newdata[x_names_nom[0]]), 1))
        for i, col in enumerate(x_names_nom):
            dummies = pd.get_dummies(ppmdata[x_names_nom[i]]).rename(columns=lambda x: 'Category_' + str(x))
            new_dummies = pd.get_dummies(newdata[x_names_nom[i]]).rename(columns=lambda x: 'Category_' + str(x))
            new_dummies = new_dummies.reindex(columns=dummies.columns, fill_value=0)
            X_pred = np.append(X_pred, np.array(new_dummies), 1)

    Y_Pred_Header = []
    Y_pred = predict(best_model, X_pred).reshape(-1, 1)
    Y_pred_limits = np.append(Y_pred - 1.96 * sigma, Y_pred + 1.96 * sigma, axis=1)
    Y_pred = np.append(Y_pred, Y_pred_limits, axis=1)
    Y_pred = np.clip(Y_pred,a_min=0,a_max=np.max(Y_pred))

    newdata['Y_Predicted'] = Y_pred[:, 0]
    newdata['Lower Limit for 95% Prediction Interval'] = Y_pred[:, 1]
    newdata['Upper Limit for 95% Prediction Interval'] = Y_pred[:, 2]

    if save == True:
        newdata.to_csv(os.path.join(directory, prediction_output_file), encoding='utf-8', index=False)

    return newdata


def cleanup_header(header,drop_category=False):

    for i in range(len(header)):
        if header[i][0:10] == 'X_SP_CNTS_':
            header[i] = header[i][10:]
        elif header[i][0:5] == 'X_SP_':
            header[i] = header[i][5:]
        elif header[i][0:5] == 'XN_M_':
            header[i] = header[i][5:]
        elif header[i][0:3] == 'XO_':
            header[i] = header[i][3:]
        elif header[i][0:3] == 'XN_':
            header[i] = header[i][3:]
        elif header[i][0:2] == 'X_':
            header[i] = header[i][2:]
        elif header[i][0:2] == 'Y_':
            header[i] = header[i][2:]
        elif (header[i][0:9] == 'Category_') and drop_category:
            header[i] = header[i][9:]

    return header

def strip_feature_name(name,drop_category=False):
    if name[0:10] == 'X_SP_CNTS_':
        name = name[10:]
    elif name[0:5] == 'X_SP_':
        name = name[5:]
    elif name[0:5] == 'XN_M_':
        name = name[5:]
    elif name[0:3] == 'XO_':
        name = name[3:]
    elif name[0:3] == 'XN_':
        name = name[3:]
    elif name[0:2] == 'X_':
        name = name[2:]
    elif name[0:2] == 'Y_':
        name = name[2:]
    elif (name[0:9] == 'Category_') and drop_category:
        name = name[9:]


def load_csv_data(data_file):
    """ To load PPM Data"""
    return pd.read_csv(data_file)


def save_fig(directory,fig_id, tight_layout=True):
    """ To save figure"""
    path = os.path.join(directory,fig_id + ".jpg")
    if tight_layout:
        plt.tight_layout()
        plt.gcf().set_size_inches(20,15)
    plt.savefig(path, format='jpg', dpi=None, orientation='landscape')
    plt.clf()
    return


def save_table(data, file, col_width=20.0, row_height=2, font_size=40,
               header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
               bbox=[0, 0, 1, 1],text_message="",text_title="", header_columns=0,size='default',
               ax=None,**kwargs):
    if ax is None:
        if size != 'default':
            size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
            fig, ax = plt.subplots(figsize=size)
        else:
            fig, ax = plt.subplots(figsize=(45,30))
        ax.axis('off')
        
    if text_title != "":
        plt.title(text_title, fontsize=font_size*1.5,color='blue')
        
    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

# --- Commented the below for loop before of NameError: name six not define ---
    '''
    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
    '''
    if text_message != "":
        ax.text(0.5, 0,text_message, fontsize=font_size*1.5,color='blue', verticalalignment='bottom', horizontalalignment='center')

    fig.savefig(file, format='jpg')
    plt.close()

    return ax

def predict(best_model,x_data):

    try:
        feature_length = len(best_model.features)
        if best_model.name == "2d-Poly Regression" or "Linear Regression":
            x_data = x_data[:,-feature_length:]

        if best_model.name == "2d-Poly Regression":
            featurizer = PolynomialFeatures(degree=2)
            x_data_transformed = featurizer.fit_transform(x_data)
        else:
            x_data_transformed = x_data
    except:
        x_data_transformed = x_data

    return best_model.predict(x_data_transformed)

def get_map_error(y_actual,y_predicted,**kwargs):
    error = np.zeros(len(y_actual))
    for i in range(len(y_predicted)):
        if y_actual[i] != 0:
            error[i] = np.absolute(y_predicted[i] - y_actual[i]) / y_actual[i]
        else:
            error[i] = np.absolute(y_predicted[i] - y_actual[i])
    mape = error.mean()

    return mape


def get_wmap_error(y_actual,y_predicted,**kwargs):

    return np.sum(np.absolute(y_predicted-y_actual))/np.sum(y_actual)


## Added for TreeInterpreter - Start
def get_TreeInterpreter(datagroup, featgroup, directory, save):

    x_train = datagroup.x_train
    y_train = datagroup.y_train
    print("Inside getTreeinterpreter method...1")
    t_params = {'max_depth': [2, 3, 4, 5]}
    tr = DecisionTreeRegressor()

    gs = GridSearchCV(tr, param_grid=t_params, cv=5,
                      scoring=make_scorer(get_map_error, greater_is_better=False))

    #print("========= X input for gs === ", x_train )
    #print("========= Y input for gs === ", y_train)

    gs.fit(x_train, y_train)

    dot_data = StringIO()
    feature_importances = []

    for col in featgroup.all:
        feature_importances.append(col)
    print("========= feature_importances", feature_importances)

    export_graphviz(gs.best_estimator_, out_file=dot_data,
                    filled=True, rounded=True, impurity=False,
                    special_characters=True, feature_names=feature_importances,
                    label="all"
                    )

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.set_simplify("True")
    print(graph)
    if save:
        file = os.path.join(directory, "MI_TreeInterpreter.jpg")
        print("file full path === ", file)
        #graph.write_png('treeInterpreter.png')
        graph.write_png(file)
         #Added for TreeInterpreter - end

#--- Start - Code added for Classification algorithms ---#

# Fitting classifier to the Training set
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def select_classification_model(ppmdata, featgroup, datagroup, summary_data_file,
                 directory, options, save=True):

    model_to_try = namedtuple('model_to_try', 'model_name,func,params,sensitive_to_correlation')
    model_search_space = []
    model_selection_ind = False

    model_name = 'Random Forest'
    params = {'n_estimators': [10], 'criterion': ['entropy'], 'random_state':[0]}
    func = RandomForestClassifier()
    sensitive_to_correlation = False
    model_search_space.append(model_to_try(model_name, func, params, sensitive_to_correlation))


    model_name = 'Decision Tree'
    params = {'criterion': ['entropy'], 'random_state':[0]}
    func = DecisionTreeClassifier()
    #sensitive_to_correlation = False
    model_search_space.append(model_to_try(model_name, func, params, sensitive_to_correlation))

    model_name = 'LogisticRegression'
    func = LogisticRegression()
    params = {'random_state':[0]}
    #sensitive_to_correlation = True
    model_search_space.append(model_to_try(model_name, func, params, sensitive_to_correlation))

    model_name = 'KNeighbors'
    func = KNeighborsClassifier()
    params = {'n_neighbors':[5], 'metric': ['minkowski'],'p':[2]}
    #sensitive_to_correlation = True
    model_search_space.append(model_to_try(model_name, func, params, sensitive_to_correlation))

    model_name = 'SVC(linear)'
    params = {'kernel': ['linear'], 'random_state':[0]}
    func = SVC()
    #sensitive_to_correlation = False
    model_search_space.append(model_to_try(model_name, func, params, sensitive_to_correlation))

    model_name = 'NaiveBayesGaussian'
    params = {}
    func = GaussianNB()
    # sensitive_to_correlation = False
    model_search_space.append(model_to_try(model_name, func, params, sensitive_to_correlation))

    model_name = 'Neural Network'
    params = {'activation': ['logistic'], 'solver': ['lbfgs', 'sgd', 'adam'],
              'learning_rate': ['constant', 'invscaling', 'adaptive']}
    func = MLPClassifier(hidden_layer_sizes=(2), max_iter=200, random_state=9)
    #sensitive_to_correlation = False
    model_search_space.append(model_to_try(model_name, func, params, sensitive_to_correlation))

    fitted_model, fitment_summary = fit_classification_models(model_search_space, datagroup,featgroup,directory, save)
    print(fitted_model)

    fitment_report = pd.DataFrame.from_records(fitment_summary, columns=['Model', 'Accuracy_Score', 'Precision', 'Recall', 'F1_Score'])

    if float(fitment_report.iloc[fitment_report.F1_Score.values.argmax()]['Accuracy_Score']) > 0.75:
        best_model = fitted_model[fitment_report.Model[fitment_report.F1_Score.values.argmax()]]
        model_selection_ind = True
        best_model_name = fitment_report.iloc[fitment_report.F1_Score.values.argmax()]['Model']
        text_message = "Selected Model : " + best_model_name
        plot_residuals_predictions(best_model, directory, save=True)
    else:
        text_message = "Could not Select a Model with Adequate Fit"
        best_model = fitted_model[fitment_report.Model[fitment_report.F1_Score.values.argmax()]]

    print(text_message)

    if save:
        file = os.path.join(directory, summary_data_file)
        save_table(fitment_report, file, header_columns=0, col_width=4.0, bbox=[0, 0.1, 1, 1], text_message=text_message)



    return best_model, model_selection_ind

def fit_classification_models(model_search_space, datagroup, featgroup, directory, save):

    fitment_summary = []
    fitted_models = {}
    splits = datagroup.y_train.shape[0]
    search_space_list = [
        (model_search_space[i].model_name, model_search_space[i].func, model_search_space[i].params,
         model_search_space[i].sensitive_to_correlation) for i in range(len(model_search_space))]

    feature_imporances = []
    data = []
    for model_name, func, params, sensitive_to_correlation in search_space_list:
        print("Fitting ", model_name)
        if sensitive_to_correlation:
            x_train = datagroup.x_train_nc
            y_train = datagroup.y_train_nc
            x_test = datagroup.x_test_nc
            y_test = datagroup.y_test_nc
            x_all = datagroup.x_all_nc
            y_all = datagroup.y_all
        else:
            x_train = datagroup.x_train
            y_train = datagroup.y_train
            x_test = datagroup.x_test
            y_test = datagroup.y_test
            x_all = datagroup.x_all
            y_all = datagroup.y_all

        if len(params) == 0:
            mdl = func.fit(x_train, y_train)
            y_pred = func.predict(x_test)
        else:
            gs = GridSearchCV(func, param_grid=params, cv=5, refit=True, error_score=0,
                              scoring=make_scorer(get_map_error, greater_is_better=False))
            gs.fit(x_train, y_train)
            y_pred = gs.predict(x_test)
            mdl = gs.best_estimator_
        mdl.model_name = model_name

        print("y_pred value .... ", y_pred)
        # Analysing the classification performance metrics
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        TP = cm[0][0]
        FP = cm[0][1]
        FN = cm[1][0]
        TN = cm[1][1]
        #er = cm[0][1] + cm[1][0]
        er = FP + FN
        co = TN + TP

        #co = cm[1][1] + cm[0][0]
        tot = er + co
        accuracy_score = (float(co) / float(tot))
        precision = (float(TP) / float (TP + FP))
        recall = (float(TP) / float(TP + FN))
        F1_score = (float(2*precision*recall)/float(precision+recall))
        accuracy_score = format(np.clip(accuracy_score, -100, 100), '2.2f')
        F1_score = format(np.clip(F1_score, -100, 100), '2.2f')
        precision = format(np.clip(precision, -100, 100),'2.2f')
        recall = format(np.clip(recall, -100, 100),'2.2f')

        print("Model name -- Classification ===", mdl.model_name)
        print("accuracy_score ===", accuracy_score)
        print("precision ===", precision)
        print("recall ===", recall)
        print("F1_score ===", F1_score)

        fitment_summary.append((model_name, accuracy_score, precision, recall, F1_score))
        if model_name == 'Random Forest':
            feature_importances = list(zip(featgroup.all, gs.best_estimator_.feature_importances_))
        mdl = mdl.fit(x_all, y_all)
        mdl = update_model_attributes(mdl, model_name, featgroup, x_all,
                                      y_all, x_train, y_train, x_test, y_test, feature_importances,
                                      sensitive_to_correlation)
        print("@2 overall mape..", get_map_error(y_all, predict(mdl, x_all)))

        fitted_models[model_name] = mdl

        print("fitted_models....", fitted_models)
        print("fitment_summary...", fitment_summary)


    with open('output', 'w') as outfile:
        json.dump(data, outfile)
    print(data)

    return fitted_models, fitment_summary

def evaluate_classification(model_name, mdl, ppmtrx_prepared, ppmtr_y, ppmtsx_prepared,
                 ppmts_y, directory):
    np.random.seed(42)

    ppmtry_predicted = mdl.predict(ppmtrx_prepared)
    ppmtsy_predicted = mdl.predict(ppmtsx_prepared)

    error_summary = namedtuple('accuracy_summary','accuracy_score,mse_test,r2_train,r2_test,'
                                               'mape_train,mape_test')
    #error_summary = namedtuple('error_summary', 'mse_train,mse_test,r2_train,r2_test')

    error_summary.mse_train = format(mean_squared_error(ppmtr_y, ppmtry_predicted),'2.9f')
    error_summary.mse_test = format(np.clip(mean_squared_error(ppmts_y, ppmtsy_predicted),-100,100),'2.9f')
    error_summary.r2_train = format(np.clip(r2_score(ppmtr_y, ppmtry_predicted),-100,100),'2.2f')
    error_summary.r2_test = format(np.clip(r2_score(ppmts_y, ppmtsy_predicted),-100,100),'2.2f')
    error_summary.mape_train = format(get_map_error(ppmtr_y, ppmtry_predicted),'2.6f')
    error_summary.mape_test = format(get_map_error(ppmts_y, ppmtsy_predicted),'2.6f')

    return error_summary

#--- End - Code added for Classification algorithms ---#

