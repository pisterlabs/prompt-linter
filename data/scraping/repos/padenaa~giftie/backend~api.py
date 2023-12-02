import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
import cohere

load_dotenv()
app = Flask(__name__)
CORS(app)
API_KEY = os.getenv("API_KEY")
co = cohere.Client(API_KEY)

#get recommendations method
@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    content_type = request.headers.get('Content-Type')
    if (content_type != 'application/json'):
        return 'Content-Type not supported!'
    
    info = request.json['info']

    prompt = "Recommend a gift for me to buy my someone who's my " + info['relationship'] + ". I know this person likes " + info['ideas'] + ". What other gifts do you think they might like? \n --k" 
    res = co.generate( 
        model='command-xlarge-nightly', 
        prompt = prompt,
        max_tokens=300,
        temperature=0.7,
        stop_sequences=["--"]
        )

    recommendation = res.generations[0].text
    print(recommendation)
    
    return jsonify({"recommendation": str(recommendation)})

app.run(host="localhost", port=8000)