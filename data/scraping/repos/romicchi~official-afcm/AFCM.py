from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import pymysql
import requests
import io
import pdfplumber
from pdfminer.high_level import extract_text
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import os
import joblib
from datetime import datetime
from glob import glob
import subprocess

app = Flask(__name__)
CORS(app)

# Set your OpenAI API key
openai.api_key = "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"

# Define your database connection parameters
db_config = {
    'host': 'localhost',
    'user': 'u203878552_gener',
    'password': 'Generlnu2023!',
    'db': 'u203878552_laravel_auth',
    'port': 3306
}

# Function to load all PDF titles from the database
def load_all_pdf_titles():
    conn = pymysql.connect(**db_config)
    query = "SELECT title FROM Resources"
    with conn.cursor() as cursor:
        cursor.execute(query)
        rows = cursor.fetchall()
        titles = [row[0] for row in rows]
    conn.close()
    return titles

# Function to download a PDF document
def download_pdf(pdf_url):
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Failed to download PDF from {pdf_url}: {str(e)}")
        return None

# Function to remove stop words
def remove_stop_words(text):
    words = text.split()
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word.lower() not in stop_words]
    processed_text = ' '.join(words)
    return processed_text

# Function to summarize a PDF given its title
def summarize_pdf_by_title(title):
    titles = load_all_pdf_titles()
    if title not in titles:
        return None

    conn = pymysql.connect(**db_config)
    query = "SELECT title, url FROM Resources WHERE title = %s"
    with conn.cursor() as cursor:
        cursor.execute(query, (title,))
        row = cursor.fetchone()
        if row:
            title, pdf_url = row
            pdf_content = download_pdf(pdf_url)
            if pdf_content is not None:
                with io.BytesIO(pdf_content) as pdf_stream:
                    pdf_text = extract_text(pdf_stream)

                # Limit text to first 4096 tokens
                text_to_summarize = pdf_text[:4096]

                # Remove stop words
                text_to_summarize = remove_stop_words(text_to_summarize)

                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": f"Summarize the document titled '{title}'."}
                    ]
                )

                summary = response['choices'][0]['message']['content']
                summary = summary.replace("In conclusion,", "").strip()

                # Extract the relevant content and limit to 2-4 sentences
                sentences = summary.split('.')
                if len(sentences) > 4:
                    summary = '.'.join(sentences[:4])  # Limit to 4 sentences
                summary = summary.replace("The document titled", "").strip()

                conn.close()
                return summary

# Define a function to fetch the URL by title from your database
def fetch_url_by_title(title):
    connection = pymysql.connect(host=db_config['host'], user=db_config['user'], password=db_config['password'],
                                db=db_config['db'], port=db_config['port'])

    try:
        with connection.cursor() as cursor:
            # Execute a SQL query to fetch the URL based on the title
            sql = "SELECT url FROM resources WHERE title = %s"
            cursor.execute(sql, (title,))
            result = cursor.fetchone()
            if result:
                return result[0]  # Return the URL

    finally:
        connection.close()

    return None  # Return None if no URL is found

# Load the latest machine learning model (pipeline)
existing_models = glob(os.path.join(os.path.dirname(__file__), 'model', 'AFCM_pipeline*.joblib'))
latest_counts = [int(re.search(r'\d+', model).group()) for model in existing_models if re.search(r'\d+', model) is not None]
latest_count = max(latest_counts, default=0)
latest_model_path = os.path.join(os.path.dirname(__file__), 'model', f'AFCM_pipeline{latest_count}.joblib')
pipeline = joblib.load(latest_model_path, mmap_mode='r')

# Mapping of disciplines to colleges
discipline_to_college = {
    'Computer Science': 'CAS',
    'Mathematics': 'CAS',
    'Natural Sciences': 'CAS',
    'The Arts': 'CAS',
    'Applied Sciences': 'CAS',
    'Social Sciences': 'CAS',
    'Language': 'CME',
    'Linguistics': 'CME',
    'Literature': 'CME',
    'Geography': 'CME',
    'Management': 'CME',
    'Philosophy': 'COE',
    'Psychology': 'COE',
    'History': 'COE',
}

@app.route('/autofill', methods=['POST'])
def autofill():
    if request.method == 'POST':
        if 'title' in request.json:
            title = request.json['title']

            # Fetch the PDF URL based on the title
            pdf_url = fetch_url_by_title(title)

            if pdf_url:
                # Download the PDF file using the URL
                pdf_content = download_pdf(pdf_url)

                if pdf_content is not None:
                    pdf_content_text = ""
                    with io.BytesIO(pdf_content) as pdf_buffer:
                        for page in pdfplumber.open(pdf_buffer).pages:
                            pdf_content_text += page.extract_text()

                    cleaned_text = preprocess_text(remove_stop_words(pdf_content_text))
                    keywords = extract_keywords(cleaned_text)

                    predicted_discipline = pipeline.predict([cleaned_text])
                    college = discipline_to_college.get(predicted_discipline[0], 'Unknown')  # Default to 'Unknown' if not found
                    
                    summary = summarize_pdf_by_title(title)

                    return jsonify({'discipline': predicted_discipline[0], 'college': college, 'keywords': keywords, 'summary': summary})

        return jsonify({'error': 'Invalid title or PDF URL'})

def preprocess_text(text):
    stop_words_custom = set(["would", "could", "should", "might", "must", "shall"])
    stop_words = stop_words_custom.union(set(stopwords.words('english')))
    
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalnum() and token.lower() not in stop_words]

    return ' '.join(tokens)

def extract_keywords(text, max_keywords=5, include_bigrams=True):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Extract the keywords from the following text:\n{text}",
        max_tokens=50,
        n=1,
        stop=None
    )

    keywords = response['choices'][0]['text'].strip().split('\n')

    # Remove dashes and empty strings
    cleaned_keywords = [keyword.replace('-', '').strip() for keyword in keywords if keyword.strip() != '']

    # Tokenize each cleaned keyword
    tokenized_keywords = [word_tokenize(keyword) for keyword in cleaned_keywords]

    # Filter out unwanted elements and limit to max_keywords
    final_keywords = []
    for tokens in tokenized_keywords:
        valid_tokens = [token for token in tokens if token.isalnum()]
        if len(valid_tokens) > 0:
            if len(valid_tokens) == 1:
                final_keywords.append(valid_tokens[0])

            if include_bigrams and len(valid_tokens) == 2:
                final_keywords.append(' '.join(valid_tokens))

        if len(final_keywords) == max_keywords:
            break

    return final_keywords

def remove_stops(text):
    text = re.sub(r'M\d+_GADD\d+_\d+_SE_C01\.QXD \d+/\d+/\d+ \d+:\d+ [APMapm]{2} Page \d+', '', text)
    text = text.replace("\n", "")

    return text.strip()

def train_svm_model():
    try:
        current_date = datetime.now()
        if current_date.month == 12 and current_date.day == 20:
            svm_script_path = 'C:/Users/LENOVO/Desktop/afcmflask/AFCM/venv/svmtraining.py'
            subprocess.run(['python', svm_script_path])
            print("SVM model re-training completed successfully.")
        else:
            print("SVM model training skipped. Today is not the 20th of December.")
    except Exception as e:
        print(f"Error during SVM model training: {e}")

# Train the SVM model only on the 20th of December
train_svm_model()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)