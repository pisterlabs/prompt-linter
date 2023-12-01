# Python Imports
from flask import Flask
from flask_cors import CORS

# Imports
from routes.treatments import treatments
from routes.cohere_api import cohere_blueprint
from logging import FileHandler,WARNING
from env_secrets import load_secrets


load_secrets()
app = Flask(__name__)

# Services
app.register_blueprint(treatments, url_prefix="/treatments")
app.register_blueprint(cohere_blueprint, url_prefix="/cohere")
CORS(app)


@app.route("/")
def home() -> str:
    return 'Health Harbor APP BACKEND API :: UNAUTHORIZED ACCESS'


if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5050, debug=True)
