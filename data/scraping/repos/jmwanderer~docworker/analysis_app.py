import os
import os.path
import io
import time
import functools
from threading import Thread
import flask
from flask import Flask
from flask import request
from flask import abort
from flask import render_template, url_for, redirect
from flask import Blueprint, g, current_app, session
from markupsafe import escape
from werkzeug.middleware.proxy_fix import ProxyFix
import werkzeug.utils
from logging.config import dictConfig
import logging
from . import doc_convert
from . import prompts
from . import document
from . import doc_gen
from . import analysis_util
from . import users
import openai
import sqlite3
import click


def create_app(test_config=None,
               fakeai=False,
               instance_path=None):
  BASE_DIR = os.getcwd()
  log_file_name = os.path.join(BASE_DIR, 'docworker.log.txt')
  FORMAT = '%(asctime)s:%(levelname)s:%(name)s:%(message)s'
  if instance_path is None:
    app = Flask(__name__, instance_relative_config=True)
  else:
    app = Flask(__name__, instance_relative_config=True,
                instance_path=instance_path)
    
  app.config.from_mapping(
    SECRET_KEY='DEV',
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY'),
    MAX_CONTENT_LENGTH = 16 * 1000 * 1000,
    DATABASE=os.path.join(app.instance_path, 'docworker.sqlite'),
    SMTP_USER = os.getenv('SMTP_USER'),
    SMTP_PASSWORD = os.getenv('SMTP_PASSWORD'),        
    SMTP_SERVER = os.getenv('SMTP_SERVER'),
    SMTP_FROM = os.getenv('SMTP_FROM'),        
    AUTO_CREATE_USERS = False,
    NO_USER_LOGIN = False,
  )
  if test_config is None:
    app.config.from_pyfile('config.py', silent=True)
  else:
    app.config.from_mapping(test_config)

  if not app.debug:
    # Configure logging
    logging.basicConfig(filename=log_file_name,
                        level=logging.INFO,
                        format=FORMAT,)
  else:
    logging.basicConfig(level=logging.INFO)
  
  doc_gen.FAKE_AI_COMPLETION=fakeai

  # If so configured, setup for running behind a reverse proxy.
  if app.config.get('PROXY_CONFIG'):
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1,
                            x_host=1, x_prefix=1)
    logging.info("set proxy fix")

  try:
    logging.info("instance path = %s", app.instance_path)
    os.makedirs(app.instance_path)
  except OSError:
    pass

  openai.api_key = app.config['OPENAI_API_KEY']

  app.register_blueprint(bp)

  @app.errorhandler(Exception)
  def handle_exception(e):
    logging.exception("Internal error")
    return e

  app.teardown_appcontext(close_db)

  @app.route('/favicon.ico')
  def favicon():
    return redirect(url_for('static', filename='favicon.ico'))
  return app


def get_db():
  if 'db' not in g:
    g.db = sqlite3.connect(
      current_app.config['DATABASE'],
      detect_types=sqlite3.PARSE_DECLTYPES)
    g.db.row_factory = sqlite3.Row
  return g.db

def close_db(e=None):
  db = g.pop('db', None)
  if db is not None:
    db.close()

def init_db():
  db = get_db()
  with current_app.open_resource('schema.sql') as f:
    db.executescript(f.read().decode('utf8'))


bp = Blueprint('analysis', __name__, cli_group=None)
    
@bp.cli.command('init-db')
def init_db_command():
  """Drop and recreate tables."""
  init_db()
  click.echo('Initialized database.')

@bp.cli.command('set-user')
@click.argument('name')
@click.argument('limit', default=100000)
def set_user_command(name, limit):
  """Create or update a user."""
  users.add_or_update_user(get_db(), name, limit)
  click.echo('Configured user.')

@bp.cli.command('set-user-key')
@click.argument('name')
@click.argument('key')
def set_user_command(name, key):
  """Update an access key for a user."""
  users.set_user_key(get_db(), name, key)
  click.echo('Updated user.')

@bp.cli.command('get-user')
@click.argument('name')
def get_user_command(name):
  """Dump details of given user."""
  users.report_user(get_db(), name)

@bp.cli.command('delete-user')
@click.argument('name')
def delete_user_command(name):
  """Delete a user."""
  users.delete_user(get_db(), name, current_app.instance_path)
  click.echo('Deleted user %s.' % name)

@bp.cli.command('list-users')
def list_command():
  """List the users in the DB."""
  users.list_users(get_db())

  
def get_doc_file_path(doc_name):
  """
  Given a document name, return the path to the pickle doc 
  on the local storage.
  """
  file_name = doc_name + '.daf'
  return os.path.join(current_app.instance_path, g.user, file_name)
  

def get_document(doc_name):
  """
  Load a document for the given doc name.
  If the name is None, or the session can not be loaded, return None
  """
  if doc_name is None:
    return None
  
  file_path = get_doc_file_path(doc_name)
  if not os.path.exists(file_path):
    return None
  doc = document.load_document(file_path)
  return doc

    
@bp.before_app_request
def load_logged_in_user():
  user_key = session.get('user_key')
  g.user = None
  g.user_init = False
  # Validate user_key
  if user_key is None or len(user_key) < 1:
      return

  user_name = users.get_user_by_key(get_db(), user_key)
  if user_name is None:
    return

  g.user = user_name
  users.note_user_access(get_db(), user_name)

  # Ensure initialized
  if users.is_initialized(get_db(), user_name):
    g.user_init = True
    return

  # Check number of users
  if users.count_users(get_db()) < users.MAX_ACCOUNTS:
    users.check_initialized_user(get_db(), 
                                 current_app.instance_path,
                                 user_name)
    g.user_init = True


def login_required(view):
  @functools.wraps(view)
  def wrapped_view(**kwargs):
    if g.user is None:
      return redirect(url_for('analysis.login'))
    if not g.user_init:
      flask.flash("User limit hit. No more available at this time.")
      return redirect(url_for('analysis.login', sent=True))
    return view(**kwargs)
  return wrapped_view

def set_logged_in_user(user_name):
  # For testing
  g.user = user_name
  users.check_initialized_user(get_db(), 
                               current_app.instance_path,
                               user_name)
  users.note_user_access(get_db(), user_name)


@bp.route("/", methods=("GET","POST"))
@login_required
def main():
  doc = None
  if request.method == "GET":  
    doc_id = request.args.get('doc')
    run_id = request.args.get('run_id')
    if run_id != None:
      run_id = int(run_id)
    doc  = get_document(doc_id)
    prompt_set = prompts.Prompts.get_initial_prompt_set()
    if doc is not None:
      # If a run is in progress, show that run
      if run_id is None and doc.is_running():
        run_id = doc.get_current_run_record().run_id
      prompt_set = doc.prompts.get_prompt_set()

    # Encode op type into prompt string
    encoded_prompt_set = []
    for prompt in prompt_set:
      if prompt[3]:
        value = prompt[2] + ':C'
      else:
        value = prompt[2] + ':T'
      encoded_prompt_set.append((prompt[1], value))

    # Don't pass in username when configured for no login
    username = None
    if not current_app.config.get('NO_USER_LOGIN'):
      username = g.user

    (depth, item_list) = (0, [])      
    if run_id:
      (depth, item_list) = doc.get_completion_family(run_id)
      
    return render_template("main.html",
                           doc=doc,
                           username=username,
                           run_id=run_id,
                           prompts=encoded_prompt_set,
                           depth=depth,
                           source_list=item_list,
                           process=request.args.get('process'))

  else:
    doc_id = request.form.get('doc')
    run_id = request.form.get('run_id')
    op_value = request.form.get('op_type')

    op_type = document.OP_TYPE_CONSOLIDATE    
    if op_value == "transform":
      op_type = document.OP_TYPE_TRANSFORM

    prompt = request.form['prompt'].strip()      
    doc = get_document(doc_id)
    # Catch case where there is no doc, or another process is running,
    # or there is no prompt.
    if doc is None or doc.is_running():
      return redirect(url_for('analysis.main', doc=doc_id, run_id=run_id))

    if prompt is None or len(prompt) == 0:
        flask.flash("Enter a prompt to run an operation.")
        return redirect(url_for('analysis.main', doc=doc_id, run_id=run_id))

    # Catch case where there is no content to process
    if run_id is not None and doc.get_result_item(run_id) is None:
      flask.flash("No result text on which to run.")
      return redirect(url_for('analysis.main', doc=doc_id, run_id=run_id))

    file_path = get_doc_file_path(doc_id)
    run_state = doc_gen.start_docgen(file_path, doc, prompt, run_id, op_type)
    new_run_id = run_state.run_id
    logging.info("Start doc run. doc_id = %s, run_id = %s, new_run_id = %s" %
                 (doc_id, run_id, new_run_id))
    document.save_document(file_path, doc)      
      
    # Check if there are clearly not  enough tokens to run the generation
    if (doc_gen.run_input_tokens(doc, run_state) >
        users.token_count(get_db(), g.user)):
      doc.mark_cancel_run("Not enough OpenAI tokens available.")
      document.save_document(file_path, doc)
    else:
      t = Thread(target=background_docgen,
                 args=[current_app.config['DATABASE'], g.user,
                       file_path, doc, run_state])
      t.start()
        
    return redirect(url_for('analysis.main',
                            doc=doc_id, run_id=new_run_id))

    
def background_docgen(db_config, username, file_path, doc, run_state):
  """
  Runs from a background thread to process the document and 
  update the consumed token accouting.
  """
  # Open database
  db = sqlite3.connect(db_config, detect_types=sqlite3.PARSE_DECLTYPES)
  db.row_factory = sqlite3.Row

  id = doc_gen.run_all_docgen(file_path, doc, run_state)
  if id is not None:
    family = doc.get_completion_list(run_state.run_id)
    tokens = sum(item.token_cost for item in family)
    users.increment_tokens(db, username, tokens)
    logging.info("updated cost for %s of %d tokens", username, tokens)
  
  # Close database
  db.close()


@bp.route("/doclist", methods=("GET","POST"))
@login_required
def doclist():
  if request.method == "GET":
    doc_id = request.args.get('doc')
    doc = get_document(doc_id)    
    user_dir = os.path.join(current_app.instance_path, g.user)  
    file_list = []
    for filename in os.listdir(user_dir):
      if filename.endswith('.daf'):
        file_list.append(filename[:-4]) 
    return render_template("doclist.html", files=file_list, doc=doc)

  else:
    if request.form.get('upload'):
      if ('file' not in request.files or
          request.files['file'].filename == ''):
        return redirect(url_for('analysis.doclist'))

      file = request.files['file']
      filename = werkzeug.utils.secure_filename(file.filename)
      user_dir = os.path.join(current_app.instance_path, g.user)

      doc_id = None
      try:    
        doc_id = document.find_or_create_doc(user_dir, filename, file)
      except doc_convert.DocError as err:
        flask.flash("Error loading file: %s" % str(err))

      if doc_id is not None:
        return redirect(url_for('analysis.main', doc=doc_id))
    return redirect(url_for('analysis.doclist'))      
      


@bp.route("/runlist", methods=("GET",))
@login_required
def runlist():
  doc_id = request.args.get('doc')  
  doc = get_document(doc_id)
  if doc is None:
    return redirect(url_for('analysis.main'))
  return render_template("runlist.html", doc=doc)

@bp.route("/about", methods=("GET",))
def about():
  doc_id = request.args.get('doc')  
  doc = get_document(doc_id)
  return render_template("about.html", doc=doc)
    
@bp.route("/segview", methods=("GET",))
@login_required
def segview():
  doc_id = request.args.get('doc')
  doc = get_document(doc_id)  
  if doc is None:
    return redirect(url_for('analysis.main'))

  run_id = request.args.get('run_id')
  item_name = request.args.get("item")
  item = doc.get_item_by_name(run_id, item_name)
  if run_id is None:
    return redirect(url_for('analysis.main', doc=doc_id))

  if item is not None:
    item_id = item.id()
  else:
    item_id = None
  
  # Source items for a completion (returns empty for docseg)
  (depth, item_list) = doc.get_completion_family(run_id, item_id)

  # Remove first item if we are showing a specific item.
  if item_id is not None and (len(item_list) > 0):
    item_list.pop(0)

  # Generate next and prev items
  next_item = None
  prev_item = None
  parent_item = None

  # Get a list of all completions
  (x, entries) = doc.get_completion_family(run_id)
  level = 0

  # Find the level of the current item
  for entry in entries:
    if entry[1] == item:
      level = entry[0]
      break

  if level > 0:
    # Build list of items under the same parent.
    sibling_list = []
    parent_candidate = None
    sibling_candidate = []

    for entry in entries:
      if entry[0] == level:
        sibling_candidate.append(entry[1])
      if entry[0] == level - 1:
        sibling_candidate = []
        parent_candidate = entry[1]
      if entry[1] == item:
        parent_item = parent_candidate
        sibling_list = sibling_candidate
        
    if item in sibling_list:
      item_index = sibling_list.index(item)
      if item_index > 0:
        prev_item = sibling_list[item_index - 1]
      if item_index < len(sibling_list) - 1:
        next_item = sibling_list[item_index + 1]
  
  return render_template("segview.html",
                         doc=doc,
                         run_id=run_id,
                         depth=depth,
                         source_list=item_list,
                         item=item, prev_item=prev_item,
                         next_item=next_item, parent_item=parent_item)


          
@bp.route("/export", methods=("POST",))
@login_required          
def export():
  doc_id = request.form.get('doc')
  run_id = request.form.get('run_id')    
  doc = get_document(doc_id)
  item_names = request.form.getlist('items')
  if doc is None:
    return redirect(url_for('analysis.main'))
    
  out_file = io.BytesIO()
  if run_id is None:
    # Export document test
    out_file.write(doc.get_doc_text().encode('utf-8'))
  else:
    # Export items from run_id record
    for name in item_names:
      out_file.write(doc.get_item_by_name(run_id, name).text().encode('utf-8'))
      out_file.write('\n\n'.encode('utf-8'))      

  out_file.seek(0, 0)    
  return flask.send_file(out_file, mimetype='text/plain;charset=utf-8',
                         as_attachment=True,
                         download_name='%s.txt' %
                         os.path.basename(doc.name()))

  
  

EMAIL_TEXT = """
Hello %s,

Your access link for DocWorker is %s?authkey=%s

If you didn't request this email, don't worry, your email address
may have been entered by mistake. You can ingore and delete this email.

"""

  
@bp.route("/login", methods=("GET", "POST"))
def login():
  if request.method == "GET":
    # Handle a login action
    auth_key = request.args.get("authkey")
    if auth_key is not None:
      user_name = users.get_user_by_key(get_db(), auth_key)
      if user_name is not None:
        # set up cookie on client and go to the main view
        session.permanent = True
        session['user_key'] = auth_key
        return redirect(url_for('analysis.main'))

      # clear cookie session
      session.permanent = False      
      session['user_key'] = None
      flask.flash("Bad access key")
      return render_template("login.html", sent=True)

    # If we sent an email, show the sent message.
    if request.args.get('sent'):
      return render_template("login.html", sent=True)

    # If we are not configured for auto-login, display
    # the request form
    if not current_app.config.get('NO_USER_LOGIN'):
      return render_template("login.html", sent=None)

    # Auto-login enabled. Check if we already have a user.
    if g.user is not None:
      return redirect(url_for('analysis.main'))

    # Support the auto-login by creating a user key.
    name = users.add_or_update_user(get_db(), None,
                                    users.DEFAULT_TOKEN_COUNT,
                                    request.remote_addr)
    return redirect(url_for('analysis.login', authkey=name))      

  else: # POST
   # TODO:
    # - track emails per time unit - rate limit
    # - track emails to target address - limit by time
    # - limit number of accounts
    address = escape(request.form.get('address'))
    if address is None or len(address) < 1:
        return redirect(url_for('analysis.login'))      

    sent = None
    key = users.get_user_key(get_db(), address)
    if key is None:
      # User does not exist. Create if we are configured and not at max
      if not current_app.config.get('AUTO_CREATE_USERS'):
        flask.flash("User %s does not exist." % address)
      elif users.count_users(get_db()) >= users.MAX_ACCOUNTS:
        flask.flash("User limit hit. No more available at this time.")
      else:
        # Create or get the user entry.
        users.add_or_update_user(get_db(), 
                                 address, users.DEFAULT_TOKEN_COUNT,
                                 request.remote_addr)
        key = users.get_user_key(get_db(), address)

    if key is not None:
      email = EMAIL_TEXT % (address,
                            url_for('analysis.login', _external=True),
                            key)
      logging.info("Login request for %s, result == %s", address, key)

      if users.check_allow_email_send(get_db(), address):
        try:
          analysis_util.send_email(current_app.config, [address],
                                   "Access Link for DocWorker", email)
          flask.flash("Email sent to: %s" % address)
          users.note_email_send(get_db(), address)
          sent = True

        except Exception as e:
          logging.info("Failed to send email %s", str(e))
          flask.flash("Email send failed")

      else:
        flask.flash("Email already recently sent to %s" % address)
      
    return redirect(url_for('analysis.login', sent=sent))      

if __name__ == "__main__":
  app = create_app()
  app.run(debug=True)
  
