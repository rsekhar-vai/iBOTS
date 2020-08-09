import click
from flask_cli import with_appcontext
from flask_migrate import Migrate, MigrateCommand
from flask_migrate import init as _init
from flask_migrate import migrate as _migrate
from . import app

@click.group()
def db():
    pass


@db.command()
@with_appcontext
def init(directory, multidb):
    """Creates a new migration repository."""
    _init()


@db.command()
@with_appcontext
def migrate(directory, message, sql, head, splice, branch_label, version_path,
            rev_id, x_arg):
    # Autogenerate a new revision file (Alias for 'revision --autogenerate')"""
    _migrate()


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

