from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired
from wtforms.widgets import TextArea

class LoginForm(FlaskForm):
    abstract_text = StringField('Enter Abstract', widget=TextArea(), validators=[DataRequired()])
    submit = SubmitField('Submit')