import re
from click import prompt
from flask import Flask, render_template, request
import requests
from wsgiref.handlers import format_date_time
import cohere
from cgi import print_arguments
co = cohere.Client('')
from youtubesearchpython import VideosSearch


app = Flask(__name__)
@app.route('/', methods=["GET", "POST"])
def index():

    if request.method == 'POST':
            apikey = ""
            webquery= ("https://api.spoonacular.com/recipes/random?apiKey="+apikey+"&number=1")  
            response1 = requests.get(webquery)
            respjson=  response1.json() 
            name = str(respjson['recipes'][0]['title'])
            recipeurl= str(respjson['recipes'][0]['sourceUrl'])
            imgurl= str(respjson['recipes'][0]['image'])
            summary = str(respjson['recipes'][0]['summary'])

    return render_template('index.html', **locals())

