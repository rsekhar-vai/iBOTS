import traceback
import zipfile
from shutil import rmtree

from flask import g, session, abort, current_app, send_from_directory
from flask_login import login_required

from . import api
#from .monitors import *
from .patterns import *
from .predictions import *
#from .recommendations import *
from .utils import *
from ..httpconfig import update_response_headers
from ..models import Agent, User


@api.before_request
def before_request():
    if session["user_id"]:
        user = User.get_user_by_id(session["user_id"])
        g.username = user.username
        g.email = user.email
        g.org = user.org
        g.role = user.role
    return

@api.after_request
def after_request(response):
    response = update_response_headers(response)
    return response

@api.route('/api/createmodel/<model>.html', methods=['POST'])
@login_required
def create_model(json_create_model):
    response_code = 0
    json_create_model['author_id'] = g.username
    success, agent = agent.create_from_json(json_create_model)
    if not success:
        abort(500)
    id = agent.id
    print("working with id...", id)

    datafile = json_create_model['datafile']
    for_what_if_data = json_create_model['for_what_if_data']

    directory = os.path.join(current_app.config['OUT_FOLDER'], str(agent.id))
    filename = os.path.join(directory,agent.data)
    filename4 = os.path.join(directory, 'whatif_input.csv')
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
            datafile.save(filename)
            infile = open(filename)
            if for_what_if_data is not None:
                for_what_if_data.save(filename4)
                what_if_file = open(filename4)
            else:
                what_if_file = None
        except:
            cleanup(id, infile, what_if_file, directory)
            response_code = 'BuildError01'
            print('Error in accessing input files.. ')
            traceback.print_exc()
            return response_code, id
    else:
        cleanup(id, None, None, directory)
        response_code = 'BuildError02'
        print('Duplicate Datastore Exists. Please contact Support')
        return 0, id

    options = json_create_model.get('options')
    try:
        response_code = _create_model(infile,options,directory,what_if_file)
    except:
        if response_code == 0:
            response_code = 'BuildError03'
        print("Error in Creating Model, Please check your data")
        traceback.print_exc()
        return response_code, id

    if response_code != 0 and response_code != 'Warn01':
        cleanup(id, infile, what_if_file, directory)

    if response_code == 0:
        try:
            agentnew = get_agent_by_id(id)
            success = agent.update_agent_as_success(agentnew)
            if not success:
                abort(500)
        except:
            cleanup(id, infile, what_if_file, directory)
            response_code = 'BuildError04'
            print("Error in Saving feature list, please contact Support")
            traceback.print_exc()

    return response_code, id

def cleanup(id,infile,what_if_file,directory):
    try:
        infile.close()
    except:
        abort(500)
    try:
        if what_if_file is not None:
            what_if_file.close()
    except:
        abort(500)
    try:
        rmtree(directory)
    except:
        abort(500)
    try:
        delete_agent(id)
    except:
        abort(500)

    return

def _create_model(infile,options,directory,what_if_file=None):

    success = True
    response_code = 0
    build_model = True

    vif_threshold = options.vif_threshold
    outlier_limit_max = options.outlier_limit
    remove_outliers = options.remove_outliers
    y_optimization_goal = options.optimization_goal
    sub_group_size = options.subgroup_size
    model_type = namedtuple('model_type','regression,classification')
    model_type = model_type(options.model_type_regression,options.model_type_classification)

    summary_data_file = current_app.config['SUMMARY_DATA']
    ppb_data_file = current_app.config['PPB']
    recommended_features_data_file = current_app.config['REC_FEATURES']
    risk_quantification_file = current_app.config['RISK_QUANT']
    best_model_file = current_app.config['BEST_MODEL']
    data_in_file = current_app.config['DATA_IN']
    feature_importance_data_file = current_app.config['FEATURE_IMPORTANCE']
    feature_redundancy_chart = current_app.config['FEATURE_REDUNDANCY']

    agentdata = load_csv_data(infile)
    agentdata.to_csv(os.path.join(directory,data_in_file), encoding='utf-8', index=False)

    try:
        messages, featgroup, outlier_rows, success = analyze_patterns(agentdata, directory, vif_threshold,
                                                                        outlier_limit_max, build_model,model_type)
    except:
        response_code = 'BuildError11'
        print("Error in Data Preprocessing.. Please check your input")
        traceback.print_exc()
        return response_code

    if not success:
        print("Error in Data Preprocessing.. Please check your input")
        response_code = 'BuildError12'
        return response_code

    try:
        agentdata, agentdata_dropped, datagroup, featgroup = \
                            prepare_data(featgroup, outlier_rows, agentdata, remove_outliers)
    except:
        print("Error in preparing data.. Please check your input")
        response_code = 'BuildError13'
        traceback.print_exc()
        return response_code

    try:
        x_data_all = np.append(datagroup.x_num_corr, datagroup.x_all_nc, 1)
        x_data_all_df = pd.DataFrame(x_data_all, columns=featgroup.all)

        # Added for Classification
        if model_type.classification:
            best_model, model_selected = select_classification_model(agentdata, featgroup, datagroup, summary_data_file,
                                                                     directory, options)
        else:
            best_model, model_selected = select_model(agentdata, featgroup, datagroup, summary_data_file, directory,
                                                      options)

    except:
        response_code = 'BuildError14'
        print("Error in Selecting Model.. Please check your input")
        traceback.print_exc()
        return response_code

    joblib.dump(best_model, os.path.join(directory,best_model_file))
    if model_selected:
        best_model = joblib.load(os.path.join(directory,best_model_file))
    else:
        print(" None of the applied models fit the data. ")
        response_code = 'Warn01'
        return response_code

    try:
        publish_baselines(best_model, agentdata, featgroup, ppb_data_file, directory)
    except:
        response_code = 'BuildError16'
        print("Error is deriving baselines.. please check your data")
        traceback.print_exc()
        return response_code

    x_names_all = list(x_data_all_df.columns)

    try:
        check_marginal_dependency(best_model, directory)
    except:
        response_code = 'BuildError17'
        print("Error in checking marginal dependency, please contact support")
        traceback.print_exc()
        return response_code

    try:
        if model_type.regression:
            check_feature_redundancy(best_model, feature_redundancy_chart, directory)
    except:
        response_code = 'BuildError18'
        print("Error in Checking feature redundancy, please conatact support")
        traceback.print_exc()
        return response_code

    try:
        check_feature_importance(best_model, feature_importance_data_file, directory)
    except:
        response_code = 'BuildError19'
        print("Error in deriving feature importances, please contact support")
        traceback.print_exc()
        return response_code

    try:
        if what_if_file is not None:
            trials_data = load_csv_data(what_if_file)
        else:
            trials_data = None
        x_at_optimum_y_df, x_at_optimum_y, x_headers, y_optimum = what_if_analysis(best_model, y_optimization_goal,
                                                                                   agentdata, datagroup, featgroup,
                                                                                   trials_data,
                                                                                   recommended_features_data_file,
                                                                                   directory)
        print("What-if analysis output is saved.")
    except:
        response_code = 'BuildError1A'
        print("Error in What-if analysis .. please check your input data.")
        traceback.print_exc()
        return response_code

    try:
        if model_type.regression:
            sigma = np.std(best_model.y_all - predict(best_model,best_model.x_all))
            method_data, risk_score = risk_quantification(agentdata, datagroup, featgroup, y_optimization_goal, y_optimum,
                                                      x_at_optimum_y_df,
                                                      best_model, sigma, directory, risk_quantification_file)
            print("Risk Quantification output is saved.")
    except:
        print("Error in Risk Quantification .. please check your input data.")
        traceback.print_exc()
        response_code = 'BuildError1C'
        return response_code

    return response_code

def get_features(file):
    agentdata = load_csv_data(file)
    features = {}
    for col in agentdata.columns:  # Classify X's into Numerical/Nominal
        # or Categorical data
        if col[:3] == "XN_":
            #values = list(np.unique(agentdata[col]))
            values = list(agentdata[col].dropna().unique())
            features[col] = values
        elif col[:3] == "XO_":
            #values = list(np.unique(agentdata[col]))
            values = list(agentdata[col].dropna().unique())
            features[col] = values
        elif col[:2] == "X_":
            values = [np.min(agentdata[col]), np.max(agentdata[col])]
            features[col] = values
    return features

@api.route('/api/old_model/<id>.html', methods=['GET'])
@login_required
def work_with_model(json_work_with_model):
    agent = get_agent_by_id(json_work_with_model['id'])

    response_code = 0; message = ''

    prediction_data = json_work_with_model.get('prediction_data',None)
    controlcharts_data = json_work_with_model.get('controlcharts_data',None)
    calibration_data = json_work_with_model.get('calibration_data',None)
    save = json_work_with_model.get('save',True)
    what_if_data = None

    directory = os.path.join(current_app.config['OUT_FOLDER'], str(agent.id))
    filename = agent.data
    base_data = os.path.join(current_app.config['OUT_FOLDER'], str(agent.id),filename)
    base_model = os.path.join(current_app.config['OUT_FOLDER'],str(agent.id),current_app.config['BEST_MODEL'])

    try:
        if controlcharts_data is not None:
            filename = os.path.join(directory,'controlcharts_input.csv')
            controlcharts_data.save(filename)
            controlcharts_data = open(filename)
            filename4 = os.path.join(directory, 'whatif_input.csv')
            if os.path.exists(filename4):
                what_if_data = open(filename4)
        if calibration_data is not None:
            filename = os.path.join(directory,'calibration_input.csv')
            calibration_data.save(filename)
            calibration_data = open(filename)
        #if prediction_data is not None:
        #    filename = os.path.join(directory, 'prediction_input.csv')
        #    prediction_data.save(filename)
        #    prediction_data = open(filename)

    except:
        response_code = 'WWError01'
        print("Error in Accessing Your file.. Please contact support")
        traceback.print_exc()
        return response_code, message

    options = json_work_with_model.get('options')
    try:
        response_code, message = _work_with_model(base_data, base_model, options, directory, prediction_data=prediction_data,
                        calibration_data=calibration_data,controlcharts_data=controlcharts_data, whatif_data=None,save=save)
    except:
        if response_code == 0:
            response_code = 'WWError02'
            print("Error in Accessing your Model, please check your data")
            traceback.print_exc()
        return response_code, None

    return response_code, message


def _work_with_model(infile,best_model_pkl,options,directory,prediction_data=None,calibration_data=None,
                       controlcharts_data=None,whatif_data=None,save=True):

    response_code = 0
    message = ''
    newmodelflag = False
    utilize_existing_agent_execute = True
    agent_prediction_execute = options.do_predictions
    what_if_analysis_execute = options.do_what_if
    control_charts_execute = options.do_control_charts
    agent_calibration_execute = options.check_calibration
    vif_threshold = options.vif_threshold
    outlier_limit_max = options.outlier_limit
    remove_outliers = options.remove_outliers
    y_optimization_goal = options.optimization_goal

    #Start - Added to include chart-type
    chart_type = options.chart_type
    print("---- Inside _work_with_model chart_type is---")
    print(chart_type)

    target_value = options.target_value
    print("---- Inside _work_with_model target_value is---")
    print(target_value)

    sub_group_size = options.subgroup_size
    minimum_rows = 2
    MAPE_Tolerance = options.mape_tolerance

    new_data_file = prediction_data
    trials_data_file = whatif_data
    control_chart_data_file = controlcharts_data
    current_data_file = calibration_data
    baseline_data_file = infile
    data_file = infile

    recommended_features_data_file = current_app.config['REC_FEATURES']
    prediction_output_file = current_app.config['PREDICTIONS']
    summary_data_file = current_app.config['SUMMARY_DATA']
    best_model_file = current_app.config['BEST_MODEL']
    data_in_file = current_app.config['DATA_IN']
    best_model = joblib.load(best_model_pkl)

    agentdata = load_csv_data(data_file)
    print("Data is successfully loaded.")

    try:
        build_model = False
        messages, featgroup, outlier_rows, success = analyze_patterns(agentdata, directory, vif_threshold,
                                                                    outlier_limit_max, build_model)
    except:
        response_code = 'WWError11'
        print("Error in Data Preprocessing.. Please check your input")
        traceback.print_exc()
        return response_code, message

    if not success:
        print("Error in Data Preprocessing.. Please check your input")
        response_code = 'WWError12'
        return response_code, message

    try:
        agentdata, agentdata_dropped, datagroup, featgroup = prepare_data(featgroup, outlier_rows, agentdata, remove_outliers)
    except:
        response_code = 'WWError13'
        print("Error in preparing Data. Please check your input")
        traceback.print_exc()
        return response_code, message

    best_model = joblib.load(best_model_pkl)

    if agent_prediction_execute:
        try:
            sigma = np.std(best_model.y_all - predict(best_model, best_model.x_all))
            prediction = predict_for_new_data(best_model, directory, featgroup, agentdata, new_data_file, sigma,
                                 prediction_output_file,save=save)
            print("Predictions are saved.")
            if save == True:
                message = prediction_output_file
            else:
                message = prediction
        except:
            print("Error in doing Predictions .. Contact Support.")
            traceback.print_exc()
            response_code = 'WWError17'
        return response_code, message

    print("--- before control_charts_execute ---")
    print(control_charts_execute)
    if control_charts_execute:
        '''
        try:
            x_names_significant_subprocess = significant_sp_x(best_model)
        except:
            print("Error in Determining Significant Sub Process X")
            traceback.print_exc()
            response_code = 'WWError18'
            return response_code, message
        '''
        #try:
            #print("Executing What-if Analysis")
            #if trials_data_file is not None:
             #   trials_data = load_csv_data(trials_data_file)
            #else:
             #   trials_data = None
            #x_at_optimum_y_df, x_at_optimum_y, x_headers, y_optimum = what_if_analysis(best_model, y_optimization_goal,
             #                                                                          agentdata, datagroup, featgroup,
              #                                                                         trials_data,
               #                                                                        recommended_features_data_file,                                                                                       directory)
        #except:
         #   print("Error in What-if analysis .. Contact Support.")
          #  traceback.print_exc()
           # response_code = 'WWError14'
            #return response_code, message

        try:
            ccdata = load_csv_data(control_chart_data_file.name)

            print("-----chart type received from input from...")
            print(chart_type)

            if chart_type == CHART_X_MR_X or chart_type == CHART_X_MR_MR:
                # Modifying existing logic to take the column with CC_ as X instead of significant_subprocess
                # for plotting controlchart for XMR/IMR
                x_names_significant_subprocess = [n for n in ccdata.columns if "CC_" in n]
                print("--- CC controlchart field name is ---")
                print(x_names_significant_subprocess)

                if (x_names_significant_subprocess == []):
                    print("Controlchart input file format issue.. Column should start with CC_")

                # Logic for XMR & IMR
                sigma = np.std(ccdata[x_names_significant_subprocess].values.ravel())
                target_mean = target_value
                cc_data = np.array(ccdata[x_names_significant_subprocess].values.ravel().reshape(-1, 1))
                message_cc = control_chart(chart_type, x_names_significant_subprocess, directory, cc_data, target_mean,
                                           sigma,
                                           sub_group_size)
            elif  chart_type == CHART_U or chart_type == CHART_P:

                x_name_numerator = "NR"
                cc_data = np.array(ccdata[x_name_numerator].values.ravel().reshape(-1, 1))
                print("input nr data...")
                print(cc_data)
                x_name_denominator = "DR"
                drdata = np.array(ccdata[x_name_denominator].values.ravel().reshape(-1, 1))
                print("input dr data...")
                print(drdata)
                target_mean = target_value
                sigma = 0
                message_cc = control_chart(chart_type, x_name_numerator, directory, cc_data,target_mean,
                                           sigma,
                                           drdata)
        except:
            print("Error in Control Charts .. Contact Support.")
            traceback.print_exc()
            response_code = 'WWError15'
            return response_code, message

    if agent_calibration_execute:
        try:
            currentdata = load_csv_data(current_data_file)
            message = check_calibration(best_model, agentdata,currentdata, minimum_rows, MAPE_Tolerance,featgroup,
                                            directory,prediction_output_file)

            print(message)
        except:
            print("Error in Recalibration .. Contact Support.")
            traceback.print_exc()
            response_code = 'WWError16'
            return response_code, message

    return response_code, message


@api.route('/api/uploadmodel/<model>.html', methods=['POST'])
@login_required
def upload_model(json_upload_model):

    json_create_model = {
        'project': json_upload_model['project'],
        'desc': json_upload_model['desc'],
        'options': json_upload_model['options'],
        'data': json_upload_model['datafile'].filename,
        'reports': current_app.config['OUT_FILE'],
        'author_id': current_app.config['USERID'],
        'datafile': json_upload_model['datafile'],
    }
    success, agent = agent.create_from_json(json_create_model)
    #db.session.add(agent)
    #db.session.commit()
    if not success:
        abort(500)

    directory = os.path.join(current_app.config['OUT_FOLDER'], str(agent.id))
    outfile = os.path.join(directory,json_create_model['reports'])

    os.makedirs(directory)

    data_filename = json_upload_model['datafile'].filename
    data_in = load_csv_data(json_upload_model['datafile'])
    data_in.to_csv(os.path.join(directory,data_filename), encoding='utf-8', index=False)


    model_filename = current_app.config['BEST_MODEL']
    model_in = joblib.load(json_upload_model['model'])
    joblib.dump(model_in, os.path.join(directory,model_filename))

    return 1


def get_agent_by_id(id):
    success, agent = agent.get_agent_by_id(id)
    if not success:
        abort(500)
    if agent is None:
        abort(404)

    return agent


def delete_agent(id):
    success, agent = agent.delete_agent(id)
    if not success:
        abort(500)
    if agent is None:
        abort(404)

    return

@api.route('/api/listmodels/<page>', methods=['GET', 'POST'])
@login_required
def list_models(page,search_project):
    success, agentlist = agent.get_agent_list(search_project,page, current_app.config['POSTS_PER_PAGE'])
    if not success:
        abort(500)

    return agentlist

@api.route('/api/model/<id>.html', methods=['GET'])
@login_required
def download_reports(id):
    agent = get_agent_by_id(id)
    directory = os.path.join(current_app.config['OUT_FOLDER'], str(id))
    outfile = os.path.join(directory,agent.reports)
    if os.path.exists(outfile):
        print("Downloading existing zip")
    else:
        print("creating New Zip")
        filenames = os.listdir(path=directory)
        outzip = zipfile.ZipFile(outfile,'w')
        for filename in filenames:
            outzip.write(os.path.join(directory,filename),arcname=filename,compress_type=zipfile.ZIP_DEFLATED)
        outzip.close()
    return send_from_directory(directory,agent.reports,as_attachment=True)

def get_model_features(id):
    agent = get_agent_by_id(id)

    model_pkl = os.path.join(current_app.config['OUT_FOLDER'],str(agent.id),current_app.config['BEST_MODEL'])
    data_file = os.path.join(current_app.config['OUT_FOLDER'], str(agent.id),agent.data)
    model = joblib.load(model_pkl)
    agentdata = load_csv_data(data_file)

    cat_names = model.features_ord + model.features_nom
    cat_data = agentdata[cat_names].values
    cat_choices_t = []
    for i in range(len(cat_names)):
        cat_choices_t.append(list(np.unique(cat_data[:, i])))
    cat_choices = []
    for choice_t in cat_choices_t:
        choice = []
        for sub_choice in choice_t:
            choice.append((sub_choice, sub_choice))
        cat_choices.append(choice)

    features_cat = dict(zip(cat_names, cat_choices))

    return model.features_num, features_cat

def check_model_version(id):

    response_code = 0
    agent = get_agent_by_id(id)

    try:
        model_pkl = os.path.join(current_app.config['OUT_FOLDER'], str(agent.id), current_app.config['BEST_MODEL'])
        model = joblib.load(model_pkl)
    except:
        print("Model Not present")
        response_code = "WWError02"
        return response_code

    try:
        test1 = model.features
        test2 = model.features_num
        test3 = model.features_ord
        test4 = model.features_nom
        test5 = model.target
        test6 = model.x_all
        test7 = model.y_all
        test8 = model.x_train
        test9 = model.y_train
        test10 = model.x_test
        test11 = model.y_test
    except:
        print("Using Old Model")
        response_code = "WWError00"

    return response_code


def get_target_name(id):
    agent = get_agent_by_id(id)
    model_pkl = os.path.join(current_app.config['OUT_FOLDER'],str(agent.id),current_app.config['BEST_MODEL'])
    model = joblib.load(model_pkl)
    return list(model.target)[0]
