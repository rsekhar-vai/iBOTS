import os
import getpass
basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    #USERID = getpass.getuser()
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard to guess string'
    MAIL_SERVER = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', '465'))
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', False)
    MAIL_USE_SSL = os.environ.get('MAIL_USE_SSL', True)
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME', 'ibots.ppa@gmail.com')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD', 'Secret$12')
    MAIL_SUBJECT_PREFIX = 'iPPA Admin'
    MAIL_SENDER = 'ibots.ppa@gmail.com'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    POSTS_PER_PAGE = 8
    CURRENT_VERSION = 3
    ALLOWED_EXTENSIONS = set(['csv'])
    REPORTS = 'OutRepository'
    OUT_FOLDER = os.path.join(basedir, REPORTS)
    OUT_FILE = 'Output.zip'
    SUMMARY_DATA = 'MS_Model_Comparison_Summary.jpg'
    DATA_IN = 'Data_in.csv'
    BEST_MODEL = 'best_model.pkl'
    PPB = 'UD_Feature_Baselines.jpg'
    PREDICTIONS = 'Predicted_Values.csv'
    RISK_QUANT = 'MI_Risk_Quantification.jpg'
    UPD_BASELINE = 'Updated_Baseline.csv'
    REC_FEATURES = 'MI_Recommended_Feature_Values.jpg'
    FEATURE_IMPORTANCE = 'MI_Feature Importance.jpg'
    FEATURE_REDUNDANCY = 'MI_Feature Redundancy.jpg'


    @staticmethod
    def init_app(app):
        pass


class DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'postgresql+psycopg2://reader:secret@localhost:5432/ippadev'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    FILES_URI = os.path.join(basedir, 'DataFiles')


class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'postgresql+psycopg2://reader:secret@localhost:5432/ippatest'
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class ProductionConfig(Config):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'postgresql+psycopg2://reader:secret@localhost:5432/ippaprod'
    SQLALCHEMY_TRACK_MODIFICATIONS = False


config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,

    'default': DevelopmentConfig
}
