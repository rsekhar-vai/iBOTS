import sys, os, re
import pandas as pd
from flask_migrate import Migrate
from bot import create_app, db
from bot.models import User

app = create_app(os.getenv('FLASK_CONFIG') or 'production')
migrate = Migrate(app, db)


if __name__ == '__main__':
    filename = sys.argv[1]
    print("****************** processing filename : ", filename)
    if filename == '':
        print("please enter valid file name")
        exit(99)
    if not os.path.isfile(filename):
        print("please enter valid file name")
        exit(99)
    userdf = pd.read_csv(filename)
    regex = '^\w+([\.-]?\w+)*@gmail.com'
    error = False
    try:
        for action in userdf['action'].unique():
            if action not in ['add', 'modify', 'delete']:
                print("Invalid Action :", action)
                error = True

        for org in userdf['org'].unique():
            if org not in ['A', 'B']:
                print("Invalid org :", org)
                error = True

        for username in userdf['username']:
            if not username.isalnum():
                print("invalid username", username)
                error = True

        for email in userdf['email']:
            if not (re.search(regex, email)):
                print("Invalid Email", email)
                error = True
    except:
        print("Invalid file, check column headers")

    if error:
        exit(99)

    with app.app_context():
        for index,request in userdf.iterrows():
            print("processing.. ", request)
            try:
                if request.action == 'add':
                    user = User(email=request.email.lower(),
                                username=request.username,
                                password='password',
                                org=request.org,
                                role=request.role
                                )
                    db.session.add(user)
                elif request.action == 'modify':
                    user = User.query.filter_by(username=request.username,email=request.email).first()
                    user.password = 'password'
                    user.org = request.org
                    user.role = request.role
                elif request.action == 'delete':
                    user = User.query.filter_by(username=request.username,email=request.email).first()
                    db.session.delete(user)
            except:
                print("error in update, please check data")
                exit(99)
            db.session.commit()