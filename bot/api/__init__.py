from flask import Blueprint

api = Blueprint('api', __name__)

from . import ppm, errors
from .predictions import   *
from .patterns import   *
