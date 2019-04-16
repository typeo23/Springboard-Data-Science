from flask import render_template, flash, redirect
from app import app
from app.forms import LoginForm
from app.model import Model


model = Model()

@app.route('/')
def root():
    return redirect('/index')


@app.route('/index', methods=['GET', 'POST'])
def index():
    form = LoginForm()
    if form.validate_on_submit():
        model.find_similar(form.abstract_text.data)
        return redirect('/index')
    return render_template('index.html', title='Sign In', form=form,
                           tables=[model.result_df.to_html(classes=["table-bordered", "table-striped", "table-hover"])],
                           titles=model.result_df.columns.values
                           )

