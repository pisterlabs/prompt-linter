from flask import Flask
from pymongo import MongoClient
import cohere
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.config["SECRET_KEY"] = "ffa5538b7e176c8ddb4fe1fb39a2775dbe9e1643"

cluster = "mongodb+srv://user:JUHnPl4Ub4IVQn2w@methackscluster.qfvwt6z.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(cluster)
user_db = client.db
prompt_db = client.db

api_key = "p9xVIxwnJdbY7AkNBZ121U1ImJeMT80HKM5Jm1DZ"
co = cohere.Client(api_key)

from application import routes
