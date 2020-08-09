import os
from flask_migrate import Migrate
import unittest
from bot import create_app, db
from bot.models import Agent, User
from waitress import serve
import logging
from paste.translogger import TransLogger
import logging, logging.config, yaml
import flask
import click
import unittest
from flask_cli import with_appcontext


app = create_app(os.getenv('FLASK_CONFIG') or 'testing')

#logging.basicConfig(filename='demo.log', level=logging.DEBUG)
#app.add_url_rule('/Reports/<path:filename>','Reports')
#logging.config.dictConfig(yaml.load(open('logging.cfg')))

migrate = Migrate(app, db)

@app.cli.command()
@click.argument('test_names', nargs=-1)
def test(test_names):
    """Run the unit tests."""
    if test_names:
        tests = unittest.TestLoader().loadTestsFromNames(test_names)
    else:
        tests = unittest.TestLoader().discover('tests')
    unittest.TextTestRunner(verbosity=2).run(tests)

@app.shell_context_processor
def make_shell_context():
    #return dict(db=db, User=User, Role=Role)
    return dict(db=db, User=User)

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