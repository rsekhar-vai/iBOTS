from flask import Blueprint

api = Blueprint('api', __name__)

from . import agent, errors
from .predictions import   *
from .patterns import   *
