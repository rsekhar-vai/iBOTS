# PPM Utilities Module
#from ..spc import *
import matplotlib as mpl
mpl.use('Agg')

# from pandas.plotting import scatter_matrix


# from scipy.stats import randint

# For outlier analysis
# For dimention reduction

#For TreeInterpreter
from .predictions import *
#from .monitors import *
#from .recommendations import *

def prepare_data(featgroup, rows, ppmdata, outlier_flag):
    """ Prepare the data """

    y_feat = featgroup.target
    x_feat_nom = featgroup.nom
    x_feat_ord = featgroup.ord
    x_feat_num_nc = featgroup.num_nc
    x_feat_num_corr = featgroup.num_corr
    ppmdata_dropped = None

    if outlier_flag and len(rows) > 0:  # Remove Outliers
        #rows = rows - 1
        indexes_to_keep = set(range(ppmdata.shape[0])) - set(rows)
        ppmdata_dropped = ppmdata.take(rows)
        ppmdata = ppmdata.take(list(indexes_to_keep))

    y_data_all = ppmdata[y_feat].values.ravel()
    x_data_all_nc = ppmdata[x_feat_num_nc].values
    x_data_num_nc = ppmdata[x_feat_num_nc].values
    x_feat_nc = x_feat_num_nc
    x_data_num_corr = ppmdata[x_feat_num_corr].values

    num_encoder = StandardScaler()  # To transform Numerical X
    x_data_all_nc = num_encoder.fit_transform(x_data_all_nc)
    if len(x_feat_num_corr) > 0:
        x_data_num_corr = num_encoder.fit_transform(x_data_num_corr)
    x_data_ord = []
    x_data_nom = []

    if len(x_feat_ord) > 0:  # To transform Ordinal X
        ord_x_mapper = [(x_feat_ord[i], LabelEncoder()) for i, col in enumerate(x_feat_ord)]
        ord_encoder = DataFrameMapper(ord_x_mapper)
        x_data_ord = ord_encoder.fit_transform(ppmdata)
        x_data_all_nc = np.append(x_data_all_nc, x_data_ord, 1)
        x_feat_nc = np.append(x_feat_nc, x_feat_ord)

    if len(x_feat_nom) > 0:  # To transform Nominal X
        x_data_nom = np.empty((len(ppmdata[x_feat_nom[0]]), 1))
        nom_encoder = []
        for i, col in enumerate(x_feat_nom):
            #dummies = pd.get_dummies(ppmdata[x_feat_nom[i]]).rename(columns=lambda x: 'Category_' + str(x))
            dummies = pd.get_dummies(ppmdata[x_feat_nom[i]]).rename(columns=lambda x: x_feat_nom[i] + ":" + str(x))
            x_data_all_nc = np.append(x_data_all_nc, np.array(dummies), 1)
            x_feat_nc = np.append(x_feat_nc, dummies.columns.values)


    x_data_all = np.append(x_data_num_corr, x_data_all_nc, 1)

    x_data_train, x_data_test, y_data_train, y_data_test = \
                            train_test_split(x_data_all, y_data_all, test_size=0.25, random_state=42)

    x_data_train_nc, x_data_test_nc, y_data_train_nc, y_data_test_nc = \
                            train_test_split(x_data_all_nc, y_data_all, test_size=0.25, random_state=42)


    datagroup = namedtuple('datagroup','x_all,x_all_nc,x_num_nc,x_num_corr,x_ord,x_nom,x_train,x_test,x_train_nc,x_test_nc, \
                                       y_all,y_train,y_test,y_train_nc,y_test_nc')

    datagroup = datagroup(x_data_all,x_data_all_nc, x_data_num_nc,x_data_num_corr, x_data_ord, x_data_nom, x_data_train,
                           x_data_test, x_data_train_nc,x_data_test_nc,y_data_all, y_data_train, y_data_test,
                           y_data_train_nc,y_data_test_nc)

    x_feat_all = np.append(x_feat_num_corr, x_feat_nc)
    featgroup = featgroup._replace(all_nc=list(x_feat_nc),all=list(x_feat_all))

    return ppmdata, ppmdata_dropped, datagroup, featgroup

def publish_baselines(best_model,ppmdata, featgroup, ppb_data_file, directory, save=True):

    num_features = best_model.features_num
    ppbcols = list(best_model.features_num) + list(featgroup.target);

    pd.options.display.float_format = '{:,.3f}'.format
    non_normality_col = []
    ucl = np.zeros(len(ppbcols))
    lcl = np.zeros(len(ppbcols))
    for i in range(len(ppbcols)):
        nv, normp = ntest(ppmdata[ppbcols[i]].values)
        if normp >= 0.05:
            ucl[i] = np.mean(ppmdata[ppbcols[i]]) + 3 * np.std(ppmdata[ppbcols[i]])
            lcl[i] = np.mean(ppmdata[ppbcols[i]]) - 3 * np.std(ppmdata[ppbcols[i]])
            if lcl[i] < 0:
                lcl[i] = 0
        else:
            ucl[i] = 'NaN'
            lcl[i] = 'NaN'
    ppb = ppmdata[ppbcols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).append(
        pd.Series(lcl, index=list(ppmdata[ppbcols].describe().columns), name='lcl'), ignore_index=False)
    ppb = ppb.append(pd.Series(ucl, index=list(ppmdata[ppbcols].describe().columns), name='ucl'),
                     ignore_index=False)
    ppb.insert(loc=0,column='Statistic',value=ppb.index)
    ppb.reset_index(drop=True,inplace=True)
    file = os.path.join(directory,ppb_data_file)
    ppb.columns = cleanup_header(list(ppb.columns))
    ppb = np.round(ppb,decimals=6)
    if save:
        #save_table(ppb,file,header_columns=0,col_width=6.0)
        print("Baselines are saved.")

    return ppb


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

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])

    if text_message != "":
        ax.text(0.5, 0,text_message, fontsize=font_size*1.5,color='blue', verticalalignment='bottom', horizontalalignment='center')

    fig.savefig(file, format='jpg')
    plt.close()

    return ax
