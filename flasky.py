from flask import Flask, jsonify
from flask_mail import Mail, Message

app = Flask(__name__)
mail = Mail()
mail_settings = {
    "MAIL_SERVER": 'smtp.gmail.com',
    "MAIL_PORT": 465,
    "MAIL_USE_TLS": False,
    "MAIL_USE_SSL": True,
    "MAIL_USERNAME": 'ibots.ppa@gmail.com',
    "MAIL_PASSWORD": 'Secret$12'
}
app.config.update(mail_settings)
mail.init_app(app)

@app.route('/')
def send_mail():
    if __name__ == '__main__':
        with app.app_context():
            subject = 'Test Mail from iBOTS'
            body = 'This is just a test mail'
            recipients = ['raja.valiveti.ai@gmail.com']
            sender = app.config.get("MAIL_USERNAME")
            msg = Message(subject=subject,body=body,recipients=recipients,sender=sender)
            mail.send(msg)
            return jsonify("Email Sent")

app.run()