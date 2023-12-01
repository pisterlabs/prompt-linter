import os
from dotenv import load_dotenv
from flask import Flask
from openai_util.chat import chat_route
from view.blog.blog import blog_route
from view.chat.my_chat import my_chat_route
from view.excel.excel import excel_route

# Create a flask application
app = Flask(__name__)
# load configuration file
load_dotenv()
# Import and register routes
app.register_blueprint(excel_route)
app.register_blueprint(blog_route)
app.register_blueprint(chat_route)
app.register_blueprint(my_chat_route)
# Set a key to save the session
app.secret_key = os.getenv('APP_SECRET_KEY')
