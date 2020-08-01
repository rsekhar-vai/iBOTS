from datetime import datetime

from flask import current_app
from flask import g
from flask_login import UserMixin
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer
from werkzeug.security import generate_password_hash, check_password_hash

from . import db
from . import login_manager


class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(64), unique=True, index=True)
    username = db.Column(db.String(64), unique=True, index=True)
    role = db.Column(db.Integer)
    password_hash = db.Column(db.String(128))
    org = db.Column(db.String(64))

    def __init__(self, **kwargs):
        super(User, self).__init__(**kwargs)

    @property
    def password(self):
        raise AttributeError('password is not a readable attribute')

    @password.setter
    def password(self, password):
        self.password_hash = generate_password_hash(password)

    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)

    @staticmethod
    def reset_password(token, new_password):
        s = Serializer(current_app.config['SECRET_KEY'])
        try:
            data = s.loads(token.encode('utf-8'))
        except:
            return False
        user = User.query.get(data.get('reset'))
        if user is None:
            return False
        user.password = new_password
        db.session.add(user)
        return True


    def ping(self):
        self.last_seen = datetime.utcnow()
        db.session.add(self)

    def to_json(self):
        json_user = {
            'url': url_for('api.get_user', id=self.id),
            'username': self.username,
        }
        return json_user
    def __repr__(self):
        return '<User %r>' % self.username

    @staticmethod
    def get_user_by_email(email):
        return User.query.filter_by(email=email).first()

    @staticmethod
    def get_user_by_id(id):
        return User.query.filter_by(id=id).first()

    @staticmethod
    def update_user(user):
        success = True
        try:
            db.session.add(user)
            db.session.commit()
        except:
            success = False

        return success

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class PPM(db.Model):
    __tablename__ = 'PPMs'
    id = db.Column(db.Integer, primary_key=True)
    project = db.Column(db.Text)
    desc = db.Column(db.Text)
    options = db.Column(db.PickleType)
    features = db.Column(db.PickleType)
    data = db.Column(db.Text)
    model = db.Column(db.Text)
    reports = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    author = db.Column(db.Integer)
    scope = db.Column(db.String(64))
    status = db.Column(db.Text)

    def to_json(self):
        json_ppm = {
            'project': self.project,
            'desc' : self.desc,
            'options' : self.options,
            'features': self.features,
            'data' : self.data,
            'model' :self.model,
            'reports': self.reports,
            'timestamp' : self.timestamp,
            'author' : self.author,
            'scope' : self.scope,
            'status' : self.status
        }
        return json_ppm

    @staticmethod
    def create_from_json(json_ppm):
        project = json_ppm.get('project')
        desc = json_ppm.get('desc')
        options_tuple = json_ppm.get('options')
        options = options_tuple._asdict()
        features = json_ppm.get('features')
        data = json_ppm.get('data')
        model = json_ppm.get('model')
        reports = json_ppm.get('reports')
        author = json_ppm.get('author')
        scope = json_ppm.get('scope')
        status = json_ppm.get('status')
        if scope == 'Dept':
            scope = g.org.lower()
        author = g.username
        status = 0

        ppm = PPM(project=project, desc=desc, options=options, features=features, data=data, model=model, reports=reports,
                   author=author,scope=scope,status=status)
        success = True
        try:
            db.session.add(ppm)
            db.session.commit()
        except:
            success = False

        return success, ppm

    @staticmethod
    def get_ppm_by_id(id):
        success = True
        try:
            ppm = PPM.query.filter_by(id=id).first()
            if not (ppm.author == g.username or ppm.scope == g.org or ppm.scope == 'unit'):
                ppm = None
        except:
            success = False
        return success, ppm

    @staticmethod
    def delete_ppm(id):
        success = True
        try:
            ppm = PPM.query.filter_by(id=id).first()
            if ppm.author != g.username:
                success = False
            else:
                ppm = PPM.query.filter_by(id=id).delete()
                db.session.commit()
        except:
            success = False
        return success

    @staticmethod
    def get_ppm_list(project,page,per_page):
        success = True
        try:
            ppmlist = PPM.query.filter(PPM.project.like(project) &
                ((PPM.author == g.username) | (PPM.scope == g.org) | (PPM.scope == 'unit'))) \
                .order_by(PPM.timestamp.desc()) \
                .paginate(page, per_page,error_out=False)
        except:
            success = False
        return success, ppmlist

    @staticmethod
    def update_ppm_as_success(ppm):
        ppm.status = 1
        success = True
        try:
            db.session.add(ppm)
            db.session.commit()
        except:
            success = False

        return success
