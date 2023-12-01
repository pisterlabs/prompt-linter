import cohere
from cohere.responses.classify import Example
from cohere.custom_model_dataset import CsvDataset
from cohere.custom_model_dataset import JsonlDataset
import time
import numpy as np
import json
import flask
import sqlite3 as sql
from cohere.responses.classify import Example
import openai
import math

openai.api_key = "YOUR_API_KEY"


def chatgptcall(prompt):
  response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[{
          "role":
          "user",
          "content":
          f"give me the 4 main ingredients in {prompt}, following the format ingredient, ingridient, etc - limited to only plainly formatted text without a preamble sentence or numbering. And no period nor and at the finally. furthermore, no generalized ingridients such as (toppings) rather, specify an example. Finally, all characters must be in lower case. Make sure to leave a space after each comma. "
      }],
      temperature=0,
      max_tokens=256,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0)

  return response["choices"][0]["message"]["content"]


cursor = sql.connect("Train.db")

#If you want to reset the database and retrain from scratch, uncomment the below code and then comment it again once you have created the table.
#cursor.execute(
#"CREATE TABLE Train(Sample string, Class string, PointID integer PRIMARY KEY)")

cursor.commit()
cursor.close()
#
#)

Cohe = cohere.Client("YOUR-KEY")

#ModelReturns = Cohe.create_custom_model("food_guide", "CLASSIFY", Train_Set)


def recieve_prompt(prompt, review, reg=False):
  response = chatgptcall(prompt).split(", ")
  return response, review


def ingridients_to_vectors(ingredients_vectors, review):
  #1 is bad, 0 is good
  ingredients = []

  for ingredient in ingredients_vectors:
    temp_ingredient = (ingredient, review)
    ingredients.append(temp_ingredient)

  return ingredients


def update_database(ingredients):
  cursor = sql.connect("Train.db")
  for item in ingredients:

    query = ''
    for value in cursor.execute("SELECT class from Train WHERE sample = (?)",
                                (item[0], )):
      query = value[0]
    if query != '':
      NewVal = math.floor(((query + item[1]) / 2))
      cursor.execute("UPDATE Train set class = (?) WHERE sample = (?)",
                     (NewVal, item[0]))
    else:
      cursor.execute("INSERT INTO Train (Sample, Class) VALUES (?, ?)", item)
  lines = []

  for rows in cursor.execute("SELECT DISTINCT sample, class FROM Train "):
    lines.append(str(rows[0]) + "," + str(rows[1]) + "\n")
  with open("Train.csv", "w", encoding="UTF-8") as file:
    file.writelines(lines)

  cursor.commit()
  cursor.close()


def to_classify(userid):
  Model = Cohe.create_custom_model(str(userid), "CLASSIFY",
                                   CsvDataset("Train.csv", ","))


# import modules needed
from flask import *
from flask_login import login_required, logout_user, current_user, login_user, UserMixin, current_user
from datetime import datetime, timedelta
from flask_sqlalchemy import *
from werkzeug.security import *
from flask_login import LoginManager
from flask_admin import *
from flask_admin.contrib.sqla import ModelView
import random
from flask_migrate import Migrate
import requests

# create the flask app and choose the static folder
app = Flask(__name__, static_folder='./static')
# secret key
app.secret_key = 'YOUR_SECRET_KEY'
# the database uri
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
# login manager for flask_login
login_manager = LoginManager()
login_manager.init_app(app)
# initializing and migrating the db
db = SQLAlchemy(app)
migrate = Migrate(app, db)


# one single model, just the user
class User(db.Model, UserMixin):
  __tablename__ = "Login"
  id = db.Column(db.Integer, primary_key=True)
  password = db.Column(db.String)
  email = db.Column(db.String, unique=True)
  code = db.Column(db.Integer)

  def __repr__(self):
    return "Registered User " + str(self.id)

  def set_password(self, password):
    self.password = generate_password_hash(password)

  def check_password(self, password):
    return check_password_hash(self.password, password)


@login_manager.user_loader
def load_user(user_id):
  if user_id is not None:
    return User.query.get(user_id)
  return None


# main route for home page
@app.route("/")
def index():
  if current_user.is_authenticated:
    return redirect("/dashboard")
  return render_template("index.html")


# login page
@app.route("/login")
def login_page():
  return render_template("login.html")


# signup page
@app.route("/signup")
def signup_page():
  return render_template("signup.html")


# login submission route
@app.route("/login_submit", methods=["GET",
                                     "POST"])  # access through post in html
def login_submit():
  email = request.form["em"]
  pwd = request.form["pwd"]
  # check if there's an existing user with that email
  user = User.query.filter_by(email=email).first()
  if user is None:
    return "This username is not registered in our system."
  if user.check_password(pwd):
    login_user(user)
    return redirect("/")
  return "Your password is incorrect."


# signup submission route
@app.route("/signup_submit", methods=["GET", "POST"])
def signup_submit():
  email = request.form["em"]
  pwd = request.form["pwd"]
  # check if there's an existing user with that email
  user = User.query.filter_by(email=email).first()
  if user is not None:
    return "This username is already registered in our system."
  user1 = User(email=email)
  user1.set_password(pwd)
  db.session.add(user1)
  db.session.commit()
  login_user(user1)  # log them in
  return redirect("/")


# dashboard route
@app.route("/dashboard")
@login_required
def dashboard():
  return render_template("dashboard.html")


# the about page
@app.route("/about")
def about_page():
  return render_template("about.html")


# the logout route
@app.route("/logout")
@login_required
def logout():
  logout_user()
  return redirect("/")


# rate meal code
@app.route("/rate-meal", methods=["GET", "POST"])
@login_required
def rate_meal():
  prompt = request.form["meal"]
  review = int(request.form["rate"])
  ingredients = ingridients_to_vectors(
      recieve_prompt(prompt, review)[0], review)
  update_database(ingredients)
  return redirect("/success")


# check meal code
@app.route("/check-meal", methods=["GET", "POST"])
@login_required
def check_meal():
  Examples = []
  cursor = sql.connect("Train.db")
  for row in cursor.execute("SELECT sample, class FROM Train"):
    Examples.append(Example(row[0], str(row[1])))
  Inquiry = request.form["meal"]
  response = Cohe.classify(model='embed-english-v2.0',
                           inputs=[Inquiry],
                           examples=Examples)

  cursor.close()

  print(response.classifications[0].confidence)

  percentage1 = response.classifications[0].labels["1"].confidence
  percentage1 = percentage1 * 100

  percentage2 = response.classifications[0].labels["2"].confidence
  percentage2 = percentage2 * 100

  percentage3 = response.classifications[0].labels["3"].confidence
  percentage3 = percentage3 * 100
  return render_template("progress.html",
                         percentage1=percentage1,
                         percentage2=percentage2,
                         percentage3=percentage3)


@app.route("/success")
def success_page():
  return render_template("success.html")


if __name__ == '__main__':
  with app.app_context():
    db.create_all()
  app.run(host='0.0.0.0', port=8080)
