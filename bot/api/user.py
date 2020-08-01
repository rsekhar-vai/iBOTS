from flask import current_app
from flask_login import current_user, login_required

from . import api
from .. import db
from ..models import User


@api.route('/register/<user>.html', methods=['POST'])
def registeruser(json_user):
    user = User.from_json(json_user)
    db.session.add(user)
    db.session.commit()
    return user

@api.route('/inquire/<user>.html', methods=['GET'])
def inquireuser(userid):
    user1 = User.query.filter_by(userid=current_app.config['USERID']).first()
    user = User.query.filter_by(userid=userid).first()
    return user

@api.route('/update/<user>.html', methods=['PUT'])
@login_required
def updateuser(json_user):
    current_user.username = json_user['username']
    current_user.mailpreference = json_user['mailpreference']
    db.session.add(current_user._get_current_object())
    db.session.commit()

    return
