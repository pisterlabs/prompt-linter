import os
import sys
import joblib
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
import cohere

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

load_dotenv()
app = Flask(__name__)
CORS(app)
API_KEY = os.getenv("API_KEY")
co = cohere.Client(API_KEY)

this_dir = os.path.dirname(__file__) # Path to loader.py
sys.path.append(os.path.join(this_dir, './model/resume-classifier'))

model = joblib.load(f"{this_dir}/model/resume-classifier")

def resume_data(resume):
    vect = CountVectorizer(max_features=200, stop_words='english')
    X = vect.fit_transform([resume]) #vectorize by frequency of words

    tfidf = TfidfTransformer()
    X = tfidf.fit_transform(X) #remove filler words like "the"
    return X

#find job method
def get_job(resume):
    #run ml model on resume to get job
    category_index = model.predict(resume_data(resume))
    category_index = category_index[0]
    categories = ["Advocate", "Arts", "Automation Testing", "Blockchain", "Business Analyst", "Civil Engineer", "Data Science", "Database", "DevOps Engineer", "DotNet Developer", "ETL Developer", 
                  "Electrical Engineering", "HR", "Hadoop", "Health and fitness", "Java Developer", "Mechanical Engineer", "Network Security Engineer",
                  "Operations Manager", "PMO", "Python Developer", "SAP Developer", "Sales", "Testing", "Web Designing"];
    job = categories[category_index]
    print(job)
    return job


#get feedback method
@app.route('/api/feedback', methods=['POST'])
def get_recommendations():
    content_type = request.headers.get('Content-Type')
    if (content_type != 'application/json'):
        return 'Content-Type not supported!'
    
    resume = request.json['resume']
    job = get_job(resume)
    prompt = "I'm a " + job + ". Critique my resume, does my experience focus on impact?: \n --k" + resume
    res = co.generate( 
        model='command-xlarge-nightly', 
        prompt = prompt,
        max_tokens=300,
        temperature=0.7,
        stop_sequences=["--"]
        )

    feedback = res.generations[0].text
    print(feedback)
    feedback = feedback.split('\n')
    feedback = feedback[-1]
    
    print(feedback)
    return jsonify({"feedback": str(feedback)})

app.run(host="localhost", port=8000)