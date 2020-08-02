from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from wtforms import RadioField, FormField, FieldList
from wtforms import StringField, SubmitField, SelectField, TextAreaField, IntegerField, FloatField, BooleanField, \
    HiddenField
from wtforms import ValidationError
from wtforms.validators import DataRequired, Length, Email, Regexp

from ..models import User


class EditProfileForm(FlaskForm):
    name = StringField('Real name', validators=[Length(0, 64)])
    location = StringField('Location', validators=[Length(0, 64)])
    about_me = TextAreaField('About me')
    submit = SubmitField('Submit')


class EditProfileAdminForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Length(1, 64),
                                             Email()])
    username = StringField('Username', validators=[
        DataRequired(), Length(1, 64),
        Regexp('^[A-Za-z][A-Za-z0-9_.]*$', 0,
               'Usernames must have only letters, numbers, dots or '
               'underscores')])
    confirmed = BooleanField('Confirmed')
    role = SelectField('Role', coerce=int)
    name = StringField('Real name', validators=[Length(0, 64)])
    location = StringField('Location', validators=[Length(0, 64)])
    about_me = TextAreaField('About me')
    submit = SubmitField('Submit')

    def __init__(self, user, *args, **kwargs):
        super(EditProfileAdminForm, self).__init__(*args, **kwargs)
        self.role.choices = [(role.id, role.name)
                             for role in Role.query.order_by(Role.name).all()]
        self.user = user

    def validate_email(self, field):
        if field.data != self.user.email and \
                User.query.filter_by(email=field.data).first():
            raise ValidationError('Email already registered.')

    def validate_username(self, field):
        if field.data != self.user.username and \
                User.query.filter_by(username=field.data).first():
            raise ValidationError('Username already in use.')


class NewAgentForm(FlaskForm):
    process_name = StringField(validators=[DataRequired(),Length(max=30)])
    agent_name = StringField(validators=[DataRequired(),Length(max=30)])
    data = FileField(label='Upload the Data',validators=[FileRequired()])
    for_what_if_data = FileField(label='Upload What-If Inputs')
    agent_type = RadioField(choices=[('regression', 'Regression'), ('classification', 'Classification')], default='regression')
    outlier_removal = BooleanField(default="checked")
    optimization_goal = RadioField(choices=[('minimize', 'Minimize'), ('maximize', 'Maximize')], default='minimize')
    scope = RadioField(choices=[('user', 'User'), ('Dept', 'Dept'),('Org','Org')], default='user')


    submit = SubmitField('Submit')

class ExistingAgentsForm(FlaskForm):
    search_project = StringField(validators=[Length(max=30)])

    h_search_project = HiddenField()


    submit = SubmitField('Submit')


class UserGuideForm(FlaskForm):
    pass

class ViewReportsForm(FlaskForm):
    pass

#class InteractivePredictionForm(FlaskForm):
#    pass


class BatchPredictionForm(FlaskForm):

    project = StringField()
    desc = StringField()
    for_prediction_data = FileField(label='Upload the Data',validators=[FileRequired()])
    submit1a = SubmitField('Submit')


class ControlChartForm(FlaskForm):

    project = StringField()
    desc = StringField()
    for_controlcharts_data = FileField(label='Upload the Data',validators=[FileRequired()])
    target_value = FloatField('target_value', validators=[DataRequired()])
    chart_type = SelectField('chart_type', [DataRequired()],
                        choices=[('XmR Chart', 'X mR - X'),
                                 ('ImR Chart', 'X mR - mR'),
                                 ('P Chart', 'p'),
                                 ('U Chart', 'u')])
    submit2 = SubmitField('Submit')

class CalibrationForm(FlaskForm):

    project = StringField()
    desc = StringField()
    for_calibration_data = FileField(label='Upload the Data',validators=[FileRequired()])
    submit3 = SubmitField('Submit')

class DeleteAgentForm(FlaskForm):

    project = StringField()
    desc = StringField()
    submit = SubmitField('Confirm Deletion')

