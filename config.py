import os
import getpass
basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    #USERID = getpass.getuser()
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard to guess string'
    MAIL_SERVER = os.environ.get('MAIL_SERVER', 'smtp.googlemail.com')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', '587'))
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() in \
        ['true', 'on', '1']
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    MAIL_SUBJECT_PREFIX = '[inteliproc]'
    MAIL_SENDER = 'inteliproc Admin <inteliproc.admin@gmail.com>'
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
    SQLALCHEMY_DATABASE_URI = os.environ.get('DEV_DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'data-dev.sqlite')
    FILES_URI = os.path.join(basedir, 'DataFiles')


class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = os.environ.get('TEST_DATABASE_URL') or \
        'sqlite://'


class ProductionConfig(Config):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'data.sqlite')


config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,

    'default': DevelopmentConfig
}
