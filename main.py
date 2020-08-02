import os
from flask_migrate import Migrate
import unittest
from bot import create_app, db
from bot.models import Agent
from waitress import serve
import logging
from paste.translogger import TransLogger
import logging, logging.config, yaml
import flask

app = create_app(os.getenv('FLASK_CONFIG') or 'production')

#logging.basicConfig(filename='demo.log', level=logging.DEBUG)
#app.add_url_rule('/Reports/<path:filename>','Reports')

#logging.config.dictConfig(yaml.load(open('logging.cfg')))

migrate = Migrate(app, db)

if __name__ == '__main__':
#    app.config.update(
#        SESSION_COOKIE_SECURE=True,
#        REMEMBER_COOKIE_SECURE=True,
#        SESSION_COOKIE_HTTPONLY=True,
#        SESSION_COOKIE_SAMESITE='Lax',
#    )
    app.run(port='5001')
#    serve(TransLogger(app,setup_console_handler=False),host='0.0.0.0',port='5000')
    #app.run(host='0.0.0.0',port='5000')