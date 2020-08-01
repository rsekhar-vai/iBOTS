import matplotlib as mpl
mpl.use('Agg')

from flask import Flask
from flask_bootstrap import Bootstrap
from flask_mail import Mail
from flask_moment import Moment
from flask_sqlalchemy import SQLAlchemy
from config import config
from flask_login import LoginManager
from flask_images import Images


bootstrap = Bootstrap()
mail = Mail()
moment = Moment()
db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = 'auth.login'



def create_app(config_name):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)
    app.secret_key = 'try'
    images = Images(app)

    bootstrap.init_app(app)
    mail.init_app(app)
    moment.init_app(app)
    db.init_app(app)
    login_manager.init_app(app)

    from .ui import ui as ui_blueprint
    app.register_blueprint(ui_blueprint)
    from .api import api as api_blueprint
    app.register_blueprint(api_blueprint)
    from .auth import auth as auth_blueprint
    app.register_blueprint(auth_blueprint, url_prefix='/auth')
#    from .tools import tools as tools_blueprint
#    app.register_blueprint(tools_blueprint,url_prefix='/tools')


    return app

