from flask import Flask, request, jsonify
import cohere
from cohere import Client

app = Flask(__name__)

@app.route('/cohere', methods=['POST'])
def cohere():
    #Cohere stuff

    co = Client('sqsbtYg22j12TP210KOOEqtRgTHNP9hO4Cd2ZHXW')
    message = request.json.get('message')
    
    # Process the message and generate a response
    response = co.generate(  
    model='command-nightly',  
    prompt = message,  
    max_tokens=200,  
    temperature=0.750)
    
    
    output = response.generations[0].text

    return jsonify({'response': output})
    

@app.route('/test', methods=['GET'])
def test():
    
    return jsonify({'response': 'Hello World!'})

if __name__ == '__main__':
    app.run()
