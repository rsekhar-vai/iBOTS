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

class Agent(db.Model):
    __tablename__ = 'agents'
    id = db.Column(db.Integer, primary_key=True)
    process_name = db.Column(db.Text)
    agent_name = db.Column(db.Text)
    options = db.Column(db.PickleType)
    data = db.Column(db.PickleType)
    features = db.Column(db.PickleType)
    model = db.Column(db.PickleType)
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    author = db.Column(db.String(64))
    scope = db.Column(db.String(64))

    def to_json(self):
        json_agent = {
            'process_name': self.process_name,
            'agent_name' : self.agent_name,
            'options' : self.options,
            'features': self.features,
            'data' : self.data,
            'model' :self.model,
            'timestamp' : self.timestamp,
            'author' : self.author,
            'scope' : self.scope,
            'status' : self.status
        }
        return json_agent

    @staticmethod
    def create_from_json(json_agent):
        process_name = json_agent.get('process_name')
        agent_name = json_agent.get('agent_name')
        options_tuple = json_agent.get('options')
        options = options_tuple._asdict()
        data = json_agent.get('data')
        features = json_agent.get('features')
        model = json_agent.get('model')
        author = json_agent.get('author')
        scope = json_agent.get('scope')
        status = json_agent.get('status')
        if scope == 'Dept':
            scope = g.org.lower()
        author = g.username
        status = 0

        agent = Agent(process_name=process_name, agent_name=agent_name, options=options, features=features,
                      data=data, model=model, author=author,scope=scope)
        success = True
        try:
            db.session.add(agent)
            db.session.commit()
        except:
            success = False

        return success, agent

    @staticmethod
    def get_agent_by_id(id):
        success = True
        try:
            agent = Agent.query.filter_by(id=id).first()
            if not (agent.author == g.username or agent.scope == g.org or agent.scope == 'unit'):
                agent = None
        except:
            success = False
        return success, agent

    @staticmethod
    def delete_agent(id):
        success = True
        try:
            agent = Agent.query.filter_by(id=id).first()
            if agent.author != g.username:
                success = False
            else:
                agent = Agent.query.filter_by(id=id).delete()
                db.session.commit()
        except:
            success = False
        return success

    @staticmethod
    def get_agent_list(process_name,page,per_page):
        success = True
        try:
            agentlist = Agent.query.filter(Agent.process_name.like(process_name) &
                ((Agent.author == g.username) | (Agent.scope == g.org) | (Agent.scope == g.unit))) \
                .order_by(Agent.timestamp.desc()) \
                .paginate(page, per_page,error_out=False)
        except:
            success = False
        return success, agentlist

