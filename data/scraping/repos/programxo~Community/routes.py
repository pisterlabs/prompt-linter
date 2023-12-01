# app/web/routes.py

from flask import render_template, flash, redirect, url_for, request, session, make_response
from flask_login import current_user, login_user, logout_user, login_required
from urllib.parse import urlparse
from domain.models import Project, User, Idea
from infrastructure.database import db
from web.forms import LoginForm, ProjectForm, RegistrationForm, IdeaForm

from flask import Blueprint, request, jsonify
from config import Config
from openai import OpenAI
client = OpenAI(api_key=Config.OPENAI_API_KEY)

web = Blueprint('web', __name__)

@web.route('/')
@web.route('/home')
@login_required
def home():
    return render_template('home.html', title='Home')

@web.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('web.home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password')
            return redirect(url_for('web.login'))
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or urlparse(next_page).netloc != '':
            next_page = url_for('web.home')
        return redirect(next_page)
    return render_template('login.html', title='Sign In', form=form)

@web.route('/logout')
@login_required
def logout():
    logout_user()
    
    #WIP Causing Problems
    #response = make_response("Logging out...")
    #response.set_cookie('session', '', expires=0, path='/')
    #session.clear()
    
    return redirect(url_for('web.login'))

@web.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('web.home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Congratulations, you are now a registered user!')
        return redirect(url_for('web.login'))
    return render_template('register.html', title='Register', form=form)

@web.route('/idea', methods=['GET', 'POST'])
@login_required
def new_idea():
    form = IdeaForm()
    if form.validate_on_submit():
        idea = Idea(content=form.content.data, user_id=current_user.id)
        db.session.add(idea)
        db.session.commit()
        return redirect(url_for('web.home'))  # Oder eine andere Zielseite
    return render_template('idea.html', title='New Idea', form=form)

@web.route('/send_message', methods=['POST'])
@login_required
def send_message():
    user_message = request.json['message']

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0613",
        messages=[{"role": "user", "content": user_message}]
    )
    reply_content = response.choices[0].message.content
    return jsonify(reply=reply_content) 

@web.route('/chat')
@login_required
def chat():
    return render_template('chat.html', title='Chat')

@web.route('/tools', methods=['GET'])
def tools():
    return render_template('tools.html')

# Add these imports to web/routes.py
from web.forms import MemberForm
from domain.models import Member

@web.route('/members', methods=['GET', 'POST'])
@login_required
def member_list():
    form = MemberForm()
    if form.validate_on_submit():
        member = Member(
            name=form.name.data,
            function=form.function.data,
            experience_level=form.experience_level.data,
            phone_number=form.phone_number.data
        )
        db.session.add(member)
        db.session.commit()
        flash('Member added successfully!')
        return redirect(url_for('web.member_list'))
    members = Member.query.all()
    return render_template('member_list.html', form=form, members=members )

@web.route('/projects', methods=['GET'])
@login_required
def projects():
    projects = Project.query.all()
    return render_template('projects.html', projects=projects)

@web.route('/project/<string:project_name>')
@login_required
def start_project(project_name):
    project = Project.query.filter_by(name=project_name).first_or_404()
    template_name = f"projects/{project.name}.html"
    return render_template(template_name, project=project)

@web.route('/create_project', methods=['GET', 'POST'])
@login_required
def create_project():
    form = ProjectForm()
    if form.validate_on_submit():
        project = Project(
            name=form.name.data,
            description=form.description.data,
            image_url=form.image_url.data
        )
        db.session.add(project)
        db.session.commit()
        flash('Your project has been created!', 'success')
        return redirect(url_for('web.projects'))
    return render_template('create_project.html', title='Create Project', form=form)

@web.route('/edit_project/<string:project_name>', methods=['GET', 'POST'])
@login_required
def edit_project(project_name):
    project = Project.query.filter_by(name=project_name).first_or_404()
    form = ProjectForm()

    if form.validate_on_submit():
        project.name = form.name.data
        project.description = form.description.data
        project.image_url = form.image_url.data
        db.session.commit()
        flash('Project updated successfully!', 'success')
        return redirect(url_for('web.projects'))

    elif request.method == 'GET':
        form.name.data = project.name
        form.description.data = project.description
        form.image_url.data = project.image_url

    return render_template('edit_project.html', title='Edit Project', form=form, project=project)

