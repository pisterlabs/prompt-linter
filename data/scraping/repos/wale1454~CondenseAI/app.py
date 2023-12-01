# Import the necessary modules
import os
from flask import Flask, redirect, render_template, request, url_for, jsonify
from goose3 import Goose
import cohere
import psycopg2

app = Flask(__name__)

# Postgres DB Setup
PostgresHost = os.getenv('PostgresHost', 'default_value')
PostgresDB = os.getenv('PostgresDB', 'default_value')
PostgresUser = os.getenv('PostgresUser', 'default_value')
PostgresPassword = os.getenv('PostgresPassword', 'default_value')

app.config['POSTGRES_HOST'] = PostgresHost
app.config['POSTGRES_PORT'] = '5432'
app.config['POSTGRES_DB'] = PostgresDB
app.config['POSTGRES_USER'] = PostgresUser
app.config['POSTGRES_PASSWORD'] = PostgresPassword

# Establish a connection to the Postgres DB 
def connect_to_db():
    conn = psycopg2.connect(
        host=app.config['POSTGRES_HOST'],
        port=app.config['POSTGRES_PORT'],
        database=app.config['POSTGRES_DB'],
        user=app.config['POSTGRES_USER'],
        password=app.config['POSTGRES_PASSWORD']
    )
    return conn

# 

@app.after_request
def add_cors_header(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, X-Requested-With, Content-Type, Accept'
    return response

# Define the endpoint for the root URL path ("/")
@app.route("/", methods=("GET", "POST"))
def index():
    
    # Initialize variables for the article name and summary
    articleName, summary, articleBody = None, None, None 

    if request.method == "POST":
        articleUrl = request.form["article_Url"]
        
        # Use Goose to extract article information from the URL
        g = Goose()
        article = g.extract(url=articleUrl)

        # Extract the article name and body
        articleName = article.title
        articleBody = article.cleaned_text
        
        # Use the Cohere API to summarize the article
        apiKey = os.getenv('COHERE_APIKEY', 'default_value')
        co = cohere.Client(apiKey)

        response = co.summarize( 
            text = articleBody,
            format='paragraph',
            temperature=0.4,
            model='summarize-xlarge', 
            length='long',
        )

        # Store the summary in the summary variable
        summary = response.summary
        
        # Store URL and title in the DB
        conn = connect_to_db()
        cursor = conn.cursor()
        cursor.execute(f"INSERT INTO condense_live (article_title, article_url) VALUES('{articleName}', '{articleUrl}') ; ")
        
        conn.commit() # Used after Insert to persist the insert query.
        conn.close()
               
    # Renders the HTML template with the article name and summary included.
    return render_template("index.html", result=articleName, result5 =summary, fullArticle= articleBody )     


@app.route("/pro", methods=("GET", "POST"))
def process_text():
    articleURL = request.args.get('articleURL')

    g = Goose()
    article = g.extract(url=articleURL)

    articleName = article.title
    articleBody = article.cleaned_text

    # Use the Cohere API to summarize the article
    apiKey = os.getenv('COHERE_APIKEY', 'default_value')
    co = cohere.Client(apiKey)

    response = co.summarize( 
        text = articleBody,
        format='paragraph',
        temperature=0.4,
        model='summarize-xlarge', 
        length='long',
    )

    # Store the summary in the summary variable
    summary = response.summary
    json_result = { "Summary": summary,
                    "Title":articleName
        }

    # Store URL and title in the DB
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute(f"INSERT INTO condense_chromex (article_title, article_url) VALUES('{articleName}', '{articleURL}') ; ")
    
    conn.commit() # Used after Insert to persist the insert query.
    conn.close()

    return jsonify(json_result)

