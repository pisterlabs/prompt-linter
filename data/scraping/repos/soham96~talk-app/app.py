from flask import Flask, redirect, url_for, session, request, render_template, flash
from flask_sqlalchemy import SQLAlchemy
import os
import sqlite3
import json
import openai
import requests
import socket
from flask_login import LoginManager, current_user, login_required, login_user, logout_user
from oauthlib.oauth2 import WebApplicationClient

from db import init_db_command
from user import User

GOOGLE_CLIENT_ID = ""
GOOGLE_CLIENT_SECRET = ""
GOOGLE_DISCOVERY_URL = (
            "https://accounts.google.com/.well-known/openid-configuration"
            )

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Set OpenAI API key
openai.api_key = ""

# User session management setup
# https://flask-login.readthedocs.io/en/latest
login_manager = LoginManager()
login_manager.init_app(app)

# Naive database setup
try:
        init_db_command()
except sqlite3.OperationalError:
        # Assume it's already been created
            pass

# OAuth 2 client setup
client = WebApplicationClient(GOOGLE_CLIENT_ID)

# Flask-Login helper to retrieve a user from our db
@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

def get_google_provider_cfg():
    return requests.get(GOOGLE_DISCOVERY_URL).json()

@app.route("/login")
def login():
    # Find out what URL to hit for Google login
    google_provider_cfg = get_google_provider_cfg()
    authorization_endpoint = google_provider_cfg["authorization_endpoint"]

    # Use library to construct the request for Google login and provide
    # scopes that let you retrieve user's profile from Google
    request_uri = client.prepare_request_uri(
        authorization_endpoint,
        redirect_uri=request.base_url + "/callback",
        scope=["openid", "email", "profile"],
                        )
    return redirect(request_uri)

@app.route("/login/callback")
def callback():
    # Get authorization code Google sent back to you
    code = request.args.get("code")
    # Find out what URL to hit to get tokens that allow you to ask for
    # things on behalf of a user
    google_provider_cfg = get_google_provider_cfg()
    token_endpoint = google_provider_cfg["token_endpoint"]

    # Prepare and send a request to get tokens! Yay tokens!
    token_url, headers, body = client.prepare_token_request(
                token_endpoint,
                    authorization_response=request.url,
                        redirect_url=request.base_url,
                            code=code
                            )
    token_response = requests.post(
                token_url,
                    headers=headers,
                        data=body,
                            auth=(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET),
                            )

    # Parse the tokens!
    client.parse_request_body_response(json.dumps(token_response.json()))

    # Now that you have tokens (yay) let's find and hit the URL
    # from Google that gives you the user's profile information,
    # including their Google profile image and email
    userinfo_endpoint = google_provider_cfg["userinfo_endpoint"]
    uri, headers, body = client.add_token(userinfo_endpoint)
    userinfo_response = requests.get(uri, headers=headers, data=body)

    # You want to make sure their email is verified.
    # The user authenticated with Google, authorized your
    # app, and now you've verified their email through Google!
    if userinfo_response.json().get("email_verified"):
        unique_id = userinfo_response.json()["sub"]
        users_email = userinfo_response.json()["email"]
        picture = userinfo_response.json()["picture"]
        users_name = userinfo_response.json()["given_name"]
    else:
        return "User email not available or not verified by Google.", 400

    # Create a user in your db with the information provided
    # by Google
    user = User(
                id_=unique_id, name=users_name, email=users_email, profile_pic=picture
                )

    # Doesn't exist? Add it to the database.
    if not User.get(unique_id):
        User.create(unique_id, users_name, users_email, picture)

    # Begin user session by logging the user in
    login_user(user)

    # Send user back to homepage
    return redirect(url_for("create"))

@app.route("/feedback", methods=["POST"])
def feedback():
    rating = request.form.get("rating")
    feedback = request.form.get("feedback")

    # Here you might want to store this feedback in a database or process it in some way

    flash("Thank you for your feedback!", "success")
    return redirect(url_for("index"))


@app.route("/feedback", methods=["GET"])
def feedback_get():
    return redirect(url_for("index"))



@app.route("/create", methods=["GET", "POST"])
def create():
    if request.method == "POST":
        talk_type = request.form.get("talk_type")
        idea = request.form.get("idea")
        conference = request.form.get("conference")
        year = int(request.form.get("year"))
        level = request.form.get("level")  # Added this line

        # Check that all fields are filled in
        if not all(
            field is not None and field != ""
            for field in [talk_type, idea, conference, year]
        ):
            flash("Please fill in all fields", "error")
            return render_template("index.html")

        # Scrape guidelines and previous talks
        guidelines = scrape_conference_guidelines(conference)
        previous_talks = scrape_previous_talks(conference, year)

        # Prepare the prompt for OpenAI API
        prompt = f"""
First, generate a list of relevant talks from the {conference} conference in {year - 1} that relate to the following talk concept:
{idea}

Next, the speaker plans to give a {talk_type} at {conference} conference. The talk's level is {level}. The concept of the talk revolves around the following ideas:
{idea}

Based on this information, and it should have the following sections:

- Title of the talk: (Come up with something relevant based on previous talks at the conference.)
- Abstract: (Provide a brief summary of the talk here. This should be a concise representation of what will be discussed in the talk.)
- Outline: (Discuss the main points and progression of the talk here.)
- Prerequisites: (Specify what the audience should already know or do before the talk.)
- Audience: (Indicate who would benefit from this talk and why.)
"""

        # Generate proposal using OpenAI API
        response = openai.Completion.create(
            engine="text-davinci-003", prompt=prompt, max_tokens=500
        )

        response_text = response.choices[0].text.strip()
        relevant_talks, proposal = response_text.split("\n\n", 1)
        relevant_talks = relevant_talks.split("\n")[
            1:
        ]  # Skip the first line which is an instruction

        return render_template("index.html", proposal=proposal)

    return render_template("index.html")

@app.route("/", methods=["GET", "POST"])
def homepage():
    print('here')
    if current_user.is_authenticated:
        return create()
    else:
        return render_template("login.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", ssl_context=("server.crt", "server.key"))
