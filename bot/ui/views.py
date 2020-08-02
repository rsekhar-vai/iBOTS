import os
import traceback
from collections import namedtuple

import numpy as np
import pandas as pd
import regex as re
import werkzeug
from werkzeug.utils import secure_filename
from flask import g, render_template, session, request, redirect, url_for, current_app, abort, flash, \
    send_from_directory
from flask_images import resized_img_src
from flask_login import login_required
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField
from wtforms import ValidationError
from wtforms.validators import DataRequired

from . import ui
from .forms import NewAgentForm, ExistingAgentsForm, UserGuideForm, \
    ViewReportsForm, CalibrationForm, ControlChartForm, BatchPredictionForm, DeleteAgentForm
from .. import api
from ..api.agent import list_models, get_agent_by_id, get_model_features, check_model_version, get_target_name
from ..api.utils import load_csv_data
from ..httpconfig import update_response_headers
from ..models import Agent, User


@ui.before_request
def before_request():
    try:
        if session["user_id"]:
            user = User.get_user_by_id(session["user_id"])
            g.username = user.username
            g.email = user.email
            g.org = user.org
            g.role = user.role
        else:
            g.username = 'testuser'
            g.email = 'testuser@mail.com'
            g.org = 'testorg'
            g.role = 'tester'
    except:
        g.username = 'testuser'
        g.email = 'testuser@mail.com'
        g.org = 'testorg'
        g.role = 'tester'
    return

@ui.after_request
def after_request(response):
    response = update_response_headers(response)
    return response

@ui.route('/', methods=['GET', 'POST'])
@ui.route('/index.html', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@ui.route('/new_agent.html', methods=['GET', 'POST'])
#@login_required
def new_agent():
    form = NewAgentForm()
    if form.validate_on_submit():
        options = set_default_options()
        options = options._replace(build_agent = True)
        pattern = '[\w-_ ]+'
        error = False

        if  re.fullmatch(pattern,form.process_name.data) is None:
            flash('Invalid Input. Process name can have letters, numbers, hypens and underscores only')
            error = True

        if re.fullmatch(pattern,form.agent_name.data) is None:
            flash('Invalid Input. Agent Name can have letters, numbers, hypens and underscores only')
            error = True

        file = request.files['data']
        if file.filename == '' or not allowed_file(file.filename):
            flash('Invalid Input Data File, only .CSV allowed')
            error = True
        filename = secure_filename(file.filename)

        if error:
            return render_template('new_agent.html', form=form)

        if form.outlier_removal.data:
            options = options._replace(remove_outliers = True)
        print("********* Agent type is ", form.agent_type.data)
        options = options._replace(optimization_goal = form.optimization_goal.data)
        print("********* Agent Objective is ", form.optimization_goal.data)
        if form.agent_type.data == 'classification':
            options = options._replace(model_type_classification = True)
        else:
            options = options._replace(model_type_regression=True)
        json_new_agent = {
            'process_name': str(form.process_name.data),
            'agent_name' : str(form.agent_name.data),
            'options' : options,
            'features' : None,
            'data' : filename,
            'reports' : current_app.config['OUT_FILE'],
            'author_id' : g.username,
            'scope' : str(form.scope.data),
            'datafile' : file,
        }
        response_code = 0
        try:
            response_code, id = api.agent.create_model(json_new_agent)
        except:
            traceback.print_exc()
            if response_code == 0:
                flash('Error in Creating Model, please check your input data')
            return redirect(request.url)

        if response_code == 0:
            return redirect(url_for('ui.view_reports',id=id))
        else:
            flash(get_response_message(response_code))
            return redirect(request.url)
    else:
        return render_template('new_agent.html',form=form)

@ui.route('/existing_agents.html', methods=['GET', 'POST'])
#@login_required
def existing_agents():
    form = ExistingAgentsForm()
    page = request.args.get('page', 1, type=int)
    sp = request.args.get('sp',"",type=str)
    message = ""
    if form.search_project.data is None or form.search_project.data == "":
        if sp == '':
            search_project = "%"
        else:
            form.h_search_project.data = sp
            search_project = "%" + form.h_search_project.data + "%"
            message = "Searched : " + form.h_search_project.data
    else:
        search_project = "%" + form.search_project.data + "%"
        form.h_search_project.data = form.search_project.data
        message = "Searched : " + form.h_search_project.data
    pagination = list_models(page,search_project)
    agents = pagination.items

    return render_template('existing_agents.html', form=form, agents=agents,
                           pagination=pagination,search_message=message)

@ui.route('/utilities.html', methods=['GET', 'POST'])
#@login_required
def utilities():
    return render_template('utilities.html')


@ui.route('/apis.html', methods=['GET', 'POST'])
#@login_required
def apis():
    return render_template('apis.html')



@ui.route('/view_reports/<id>.html', methods=['GET'])
#@login_required
def view_reports(id):
    agent = get_agent_by_id(id)
    if agent is None:
        abort(404)
    form = ViewReportsForm()
    directory = os.path.join(current_app.config['OUT_FOLDER'], str(id))
    try:
        filenames = os.listdir(path=directory)
    except:
        abort(404)
    report_tuple = namedtuple('reports','filename,desc')
    ud_reports = []
    ms_reports = []
    mi_reports = []
    cc_reports = []
    for i in range(len(filenames)):
        if '.' in filenames[i] and filenames[i].rsplit('.', 1)[1].lower() == 'jpg' or 'jpeg' or 'png' or 'pdf':
            filename =  str(id) + "/" + filenames[i]
            desc = filenames[i].rsplit('.',1)[0].replace('_',' ')
            if desc[0:2] == "UD":
                ud_reports.append(report_tuple(filename,desc[2:]))
            elif desc[0:2] == "MS":
                ms_reports.append(report_tuple(filename, desc[2:]))
            elif desc[0:2] == "MI":
                mi_reports.append(report_tuple(filename, desc[2:]))
            elif desc[0:2] == "CC":
                cc_reports.append(report_tuple(filename, desc[2:]))

    return render_template('view_reports.html', form=form,id=id,ud_reports=ud_reports,ms_reports=ms_reports,
                           mi_reports=mi_reports,cc_reports=cc_reports,resized_img_src=resized_img_src)

@ui.route('/delete_model/<id>.html', methods=['GET','POST'])
#@login_required
def delete_model(id):
    agent = get_agent_by_id(id)
    if agent is None:
        abort(404)
    form = DeleteAgentForm()
    form.project = agent.project
    form.desc = agent.desc
    if form.validate_on_submit():
        success = agent.delete_agent(id)
        if success:
            flash('Successfully Deleted')
        else:
            flash('Not Authorized - Only creator can delete the model')
        return redirect(url_for('ui.existing_agents'))
    else:
        return render_template('delete_model.html',form=form)

@ui.route('/work_with_model/<id>.html', methods=['GET','POST'])
#@login_required
def work_with_model(id):
    response_code = check_model_version(id)
    if response_code != 0:
        flash(get_response_message(response_code))
        return redirect(url_for('ui.existing_agents'))

    agent = get_agent_by_id(id)
    form1 = get_interactive_predictionform(id)
    form1a = BatchPredictionForm()
    form2 = ControlChartForm()
    form3 = CalibrationForm()
    form1.project = agent.project
    form1.message1 = ''
    form1.message2 = ''
    form1.message3 = ''
    form1a.project = agent.project
    form1.desc = agent.desc
    form1a.desc = agent.desc
    form2.project = agent.project
    form2.desc = agent.desc
    form3.project = agent.project
    form3.desc = agent.desc
    message = ''
    active1 = ''
    active1a = ''
    active2 = ''
    active3 = ''
    in1 = ''
    in1a = ''
    in2 = ''
    in3 = ''
    response_code = 0
    if form1.submit1.data:
        active1 = 'active'
        in1 = 'in'
        if form1.validate():
            response_code, message = do_interactive_prediction(id,agent,form1)
            if response_code == 0:
                target = get_target_name(id)
                message1 = "Predicted " + target + " : "
                form1.message1 = message1 + str(np.round(message['Y_Predicted'].values[0],8))
                form1.message2 = "Lower Limit for 95% Prediction Interval : " + \
                                 str(np.round(message['Lower Limit for 95% Prediction Interval'].values[0], 8))
                form1.message3 = "Upper Limit for 95% Prediction Interval : " + \
                                 str(np.round(message['Upper Limit for 95% Prediction Interval'].values[0], 8))

            else:
                flash(get_response_message(response_code))
            return render_template('work_with_model.html', form1=form1, form1a=form1a,form2=form2, form3=form3, id=id,
                                   active1=active1, active1a=active1a, active2=active2, active3=active3,
                                    in1=in1, in1a=in1a, in2=in2, in3=in3)
        else:
            flash("Error in Choice of Inputs, Please check")
            return render_template('work_with_model.html', form1=form1, form1a=form1a,form2=form2, form3=form3, id=id,
                                   active1=active1, active1a=active1a, active2=active2, active3=active3,
                                    in1=in1, in1a=in1a, in2=in2, in3=in3)
    elif form1a.submit1a.data:
        active1a = 'active'
        in1a = 'in'
        file1 = request.files.get('for_prediction_data')
        if file1.filename == '' or not allowed_file(file1.filename):
            flash('Invalid File for Prediction')
            return render_template('work_with_model.html', form1=form1, form1a=form1a, form2=form2, form3=form3, id=id,
                                   active1=active1, active1a=active1a, active2=active2, active3=active3,
                                   in1=in1, in1a=in1a, in2=in2, in3=in3)
        else:
            filename1 = secure_filename(file1.filename)
        response_code, message = do_batch_prediction(id, file1, filename1, agent)
        if response_code != 0:
            flash(get_response_message(response_code))
            return render_template('work_with_model.html', form1=form1, form1a=form1a, form2=form2, form3=form3, id=id,
                                   active1=active1, active1a=active1a, active2=active2, active3=active3,
                                   in1=in1, in1a=in1a, in2=in2, in3=in3)
        else:
            directory = os.path.join(current_app.config['OUT_FOLDER'], id)
            return send_from_directory(directory, message, as_attachment=True)
            send_report(message)
            return render_template('work_with_model.html', form1=form1, form1a=form1a, form2=form2, form3=form3, id=id,
                                   active1=active1, active1a=active1a, active2=active2, active3=active3,
                                   in1=in1, in1a=in1a, in2=in2, in3=in3)

    elif form2.submit2.data:
        active2 = 'active'
        in2 = 'in'
        file2 = request.files.get('for_controlcharts_data')
        if file2.filename == '' or not allowed_file(file2.filename):
            flash('Invalid File for Control Charts')
            return render_template('work_with_model.html', form1=form1, form1a=form1a, form2=form2, form3=form3, id=id,
                                   active1=active1, active1a=active1a, active2=active2, active3=active3,
                                   in1=in1, in1a=in1a, in2=in2, in3=in3)
        else:
            filename2 = secure_filename(file2.filename)
        #Start -- Added code to include Chart_type as input
        charttype = dict(form2.chart_type.choices).get(form2.chart_type.data)
        target_value = form2.target_value.data
        print("In view.py.. chart type selected is ")
        print(charttype)
        print("In view.py.. Target Value entered from UI is ")
        print(target_value)
        response_code, message = build_control_chart(id,file2,filename2,agent, charttype, target_value)
        if response_code != 0:
            flash(get_response_message(response_code))
            return render_template('work_with_model.html', form1=form1, form1a=form1a, form2=form2, form3=form3, id=id,
                                   active1=active1, active1a=active1a, active2=active2, active3=active3,
                                   in1=in1, in1a=in1a, in2=in2, in3=in3)
        else:
            return redirect(url_for('ui.view_reports',id=id))
    elif form3.submit3.data:
        active3 = 'active'
        in3 = 'in'
        file3 = request.files.get('for_calibration_data')
        if file3.filename == '' or not allowed_file(file3.filename):
            flash('Invalid File for Calibration')
            return render_template('work_with_model.html', form1=form1, form1a=form1a, form2=form2, form3=form3, id=id,
                                   active1=active1, active1a=active1a, active2=active2, active3=active3,
                                   in1=in1, in1a=in1a, in2=in2, in3=in3)
        else:
            filename3 = secure_filename(file3.filename)
        response_code, message = check_calibration(id,file3,filename3,agent)
        if response_code != 0:
            flash(get_response_message(response_code))
            return render_template('work_with_model.html', form1=form1, form1a=form1a, form2=form2, form3=form3, id=id,
                                   active1=active1, active1a=active1a, active2=active2, active3=active3,
                                   in1=in1, in1a=in1a, in2=in2, in3=in3)
        else:
            form3.message = message
            return render_template('work_with_model.html', form1=form1, form1a=form1a, form2=form2, form3=form3, id=id,
                                   active1=active1, active1a=active1a, active2=active2, active3=active3,
                                   in1=in1, in1a=in1a, in2=in2, in3=in3)

    else:
        form1.project = agent.project
        form1.desc = agent.desc
        form2.project = agent.project
        form1a.project = agent.project
        form1a.desc = agent.desc
        form2.desc = agent.desc
        form3.project = agent.project
        form3.desc = agent.desc
        active1 = ''
        active1a = 'active'
        active2 = ''
        active3 = ''
        in1 = ''
        in1a = 'in'
        in2 = ''
        in3 = ''

        return render_template('work_with_model.html', form1=form1, form1a=form1a, form2=form2, form3=form3, id=id,
                               active1=active1, active1a=active1a, active2=active2, active3=active3,
                               in1=in1, in1a=in1a, in2=in2, in3=in3)

def get_interactive_predictionform(id):

    class InteractivePredictionForm(FlaskForm):
        pass

    model_features_num, model_features_cat = get_model_features(id)

    for feature in model_features_num:
        setattr(InteractivePredictionForm,feature,StringField(feature,validators=[DataRequired(),validate_numeric]))

    for feature in model_features_cat.items():
        setattr(InteractivePredictionForm,feature[0],SelectField(feature[0],validators=[DataRequired()],choices=feature[1]))

    setattr(InteractivePredictionForm,'submit1',SubmitField('Submit'))

    return InteractivePredictionForm()

def validate_numeric(form, field):
    try:
        data = float(field.data)
    except ValueError:
        field.message = "..Invalid Value! "
        raise ValidationError('Field must be numeric')

def do_interactive_prediction(id, agent, form):
    options = set_default_options()
    options = options._replace(do_predictions = True)
    if agent.options == None:
        options = options._replace(optimization_goal =  'minimize')
    else:
        options = options._replace(optimization_goal = agent.options.get('optimization_goal','minimize'))
    response_code = 0
    file1 = pd.DataFrame(dict(form.data),index=[0])
    json_work_with_model = {
        'id': id,
        'options' : options,
        'prediction_data' : file1,
        'controlcharts_data' : None,
        'calibration_data' : None,
        'whatif_data': None,
        'riskquant_data': None,
        'prediction_data_name': None,
        'controlcharts_data_name': None,
        'calibration_data_name': None,
        'whatif_data_name': None,
        'riskquant_data_name':None,
        'save':False,
    }
    try:
        response_code, message  = api.agent.work_with_model(json_work_with_model)
    except:
        if response_code !=0:
            response_code = 'DPError01'
            print("Error in doing Predictions, please check input data")
    return response_code, message



def do_batch_prediction(id, file1, filename1, agent):
    options = set_default_options()
    options = options._replace(do_predictions = True)
    if agent.options == None:
        options = options._replace(optimization_goal =  'minimize')
    else:
        options = options._replace(optimization_goal = agent.options.get('optimization_goal','minimize'))
    response_code = 0
    loaded_file1 = load_csv_data(file1)
    json_work_with_model = {
        'id': id,
        'options' : options,
        'prediction_data' : loaded_file1,
        'controlcharts_data' : None,
        'calibration_data' : None,
        'whatif_data': None,
        'riskquant_data': None,
        'prediction_data_name': filename1,
        'controlcharts_data_name': None,
        'calibration_data_name': None,
        'whatif_data_name': None,
        'riskquant_data_name':None,
    }
    try:
        response_code, message  = api.agent.work_with_model(json_work_with_model)
    except:
        if response_code !=0:
            response_code = 'DPError01'
            print("Error in doing Predictions, please check input data")
    return response_code, message

def build_control_chart(id,file2,filename2,agent, charttype, target_value):
    options = set_default_options()

    if agent.options == None:
        options = options._replace(optimization_goal =  'minimize')
    else:
        options = options._replace(optimization_goal = agent.options.get('optimization_goal','minimize'))

    #STart added the below line charttype
    options = options._replace(do_control_charts=True)
    options = options._replace(chart_type=charttype)
    options = options._replace(target_value=target_value)

    response_code = 0
    json_work_with_model = {
        'id': id,
        'options' : options,
        'prediction_data' : None,
        'controlcharts_data' : file2,
        'calibration_data' : None,
        'whatif_data': None,
        'riskquant_data': None,
        'prediction_data_name': None,
        'controlcharts_data_name': filename2,
        'calibration_data_name': None,
        'whatif_data_name': None,
        'riskquant_data_name':None,
    }
    print("In build_control_chart ---Options ---")
    print(options)
    print("---json-workwithmodel---")
    print(json_work_with_model)
    try:
        response_code, message  = api.agent.work_with_model(json_work_with_model)
    except:
        if response_code !=0:
            response_code = 'CCError01'
            print("Error in building Control Charts, please check input data")
    return response_code, message


def check_calibration(id,file3,filename3,agent):
    options = set_default_options()
    options = options._replace(check_calibration = True)
    if agent.options == None:
        options = options._replace(optimization_goal =  'minimize')
    else:
        options = options._replace(optimization_goal = agent.options.get('optimization_goal','minimize'))
    response_code = 0
    json_work_with_model = {
        'id': id,
        'options' : options,
        'prediction_data' : None,
        'controlcharts_data' : None,
        'calibration_data' : file3,
        'whatif_data': None,
        'riskquant_data': None,
        'prediction_data_name': None,
        'controlcharts_data_name': None,
        'calibration_data_name': filename3,
        'whatif_data_name': None,
        'riskquant_data_name':None,
    }
    try:
        response_code, message  = api.agent.work_with_model(json_work_with_model)
    except:
        if response_code != 0:
            response_code = 'RCError01'
            print("Error in Recalibration, please check input data")

    return response_code, message

@ui.route('/user_guide.html', methods=['GET'])
def user_guide():
    form=UserGuideForm()
    return render_template('user_guide.html',form=form)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

def allowed_modelfile(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pkl'



@ui.route('/shutdown')
def server_shutdown():
    if not current_app.testing:
        abort(404)
    shutdown = request.environ.get('werkzeug.server.shutdown')
    if not shutdown:
        abort(500)
    shutdown()
    return 'Shutting down...'

@ui.route('/Reports/<path:filename>', methods=['GET'])
def send_report(filename):
    directory = os.path.join(current_app.config['OUT_FOLDER'])
    return send_from_directory(directory,filename)

def set_default_options():
    options = namedtuple('options','build_agent,do_predictions,do_what_if,do_control_charts,check_calibration,do_risk_quantificaiton,   \
                                   try_simple_linear,try_support_vector,try_neural_networks,              \
                                   try_random_forests,try_decision_trees,try_adaboost,try_bayesian,try_ridge,try_lasso,     \
                                   vif_threshold,outlier_limit, remove_outliers, \
                                   optimization_goal,model_type_regression,model_type_classification,subgroup_size, \
								   mape_tolerance,chart_type,target_value')

    options = options(False,False,False,False,False,False,False,False,False,False,
                      False,False,False,False,False,2,0.05,True,'minimize',False,False,1,0.15,'X mR - X',0.0)

    return options

def get_data_paths():

    datapath = namedtuple('datapath','directory,reports_download,model_comparison,data_in, \
                            best_model,ppb,predictions,risk_quant,rec_features,feature_importance, \
                            feature_redundancy')

    datapath.directory = current_app.config['OUT_FOLDER']
    datapath.reports_download =  current_app.config['OUT_FILE']
    datapath.model_comparison = current_app.config['SUMMARY_DATA']
    datapath.data_in = current_app.config['DATA_IN']
    datapath.best_model = current_app.config['BEST_MODEL']
    datapath.ppb = current_app.config['PPB']
    datapath.predictions = current_app.config['PREDICTIONS']
    datapath.risk_quant = current_app.config['RISK_QUANT']
    datapath.rec_features = current_app.config['REC_FEATURES']
    datapath.feature_importance = current_app.config['FEATURE_IMPORTANCE']
    datapath.feature_redundancy = current_app.config['FEATURE_REDUNDANCY']

    return datapath

def get_response_message(response_code):

    message_map = {

        'Error': 'Error, please contact your support',
        'BuildError01': 'Unable to access input files',
        'BuildError02': 'Duplicate Datastore found for your model, please contact Support',
        'BuildError03': 'Error in Processing, please check your inputs',
        'BuildError04': 'Failed to save Feature List, please contact support',
        'BuildError11': 'Failed to prep-process model input data, please check your inputs',
        'BuildError12': 'Failed to pre-process model input data, please check your inputs',
        'BuildError13': 'Failed to prepare model input data, please check your inputs',
        'BuildError14': 'Error in Selecting Model, please check your inputs',
        'Warn01': 'Unable to Select a Model based on Data given, please View Reports for more details',
        'BuildError16': 'Failed to Create Baselines, please contact Support',
        'BuildError17': 'Failed to Create Marginal Dependencies, please contact Support',
        'BuildError18': 'Failed to Check Feature Redundancies, please contact Support',
        'BuildError19': 'Failed to Create Feature Importance Data, please contact Support',
        'BuildError1A': 'Failed during What-If Analysis, please check your input data',
        'BuildError1B': 'Failed during Control Charts, please check your input data',
        'BuildError1C': 'Failed during Risk Quantification, please contact Support',
        'WWError00': 'Model was built with old BOT Version, Please rebuild',
        'WWError01': 'Unable to access input files',
        'WWError02': 'Unable to access your model',
        'WWError11': 'Failed to pre-process model input data, please check your input data',
        'WWError12': 'Failed to pre-process model input data, please check your inputs',
        'WWError13': 'Failed to prepare model input data, please check your input data',
        'WWError14': 'Failed during What-If Analysis, please contact Support',
        'WWError15': 'Failed during Control Charts, please check your input data',
        'WWError16': 'Failed during Recalibration, please check your input data',
        'WWError17': 'Failed during Predictions, please check your input data',
        'WWError18': 'Could not determine Significant Subprocess, please check your input data',

    }

    return_message = str(response_code) + ":" + message_map.get(response_code,"Processing Error, Please contact your Support")

    return return_message