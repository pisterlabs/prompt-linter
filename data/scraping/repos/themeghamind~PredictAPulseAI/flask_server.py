from flask import Flask, request, jsonify
from tesseract import transcribe
import requests
import json
import os
import openai

app = Flask(__name__)

@app.route('/checkrisk', methods=['POST'])
def check_risk():
    age = int(request.args.get('age'))
    sex = int(request.args.get('sex'))
    cholesterol = int(request.args.get('cholesterol'))
    smoking = int(request.args.get('smoking'))
    bp = int(request.args.get('bp'))
    diabetes = int(request.args.get('diabetes'))
    obesity = int(request.args.get('obesity'))

    url = 'http://127.0.0.1:47334/api/sql/query'
    resp = requests.post(url, json={'query':
                                        'CREATE DATABASE cockroachdb WITH engine = \'cockroachdb\', parameters = {"host": "plumed-piranha-3682.g95.cockroachlabs.cloud", "database": "defaultdb", "user": "adrian", "password": "WsBlLnu1G6v4Wd_VAKa6dw", "port": "26257"};'})
    resp = requests.post(url, json={'query':
                                        'CREATE MODEL mindsdb.heart_attack_risk_predictor FROM cockroachdb (SELECT * FROM heart_attack_predict) PREDICT risk;'})
    risk_score = requests.post(url, json={'query':
                                        f'SELECT risk FROM mindsdb.heart_attack_risk_predictor WHERE age= {age} AND sex= {sex} AND cholesterol= {cholesterol} AND smoking= {smoking} AND bp= {bp} AND diabetes= {diabetes} AND obesity= {obesity};'})

    if risk_score.status_code == 200:
        response_data = json.loads(risk_score.text)
        risk_value = response_data["data"][0][0] if "data" in response_data else None

        with open("risk.txt", "a") as f:
            if risk_value == 1:
                f.write("This patient has a high probability of suffering a heart attack in the near future.")
            elif risk_value == 0:
                f.write("This patient has a low probability of suffering a heart attack in the near future. They have very little to worry about.")

        return jsonify({'risk_score': risk_value})
    else:
        return jsonify({'error': 'Failed to retrieve risk score'})

@app.route('/tesseract', methods=['POST'])
def tesseract_reader():
    url1 = request.args.get('url1')
    url2 = request.args.get('url2')

    transcribe(url1)
    transcribe(url2)

    # Open and read the contents of the output.txt file
    with open("output.txt", "r") as file:
        content = file.read()

    return jsonify({"output": content})

@app.route('/retrieverecs', methods=['GET'])
def retrieve_recommendations():
    with open("risk.txt", "r") as file:
        risk = file.read()

    with open("output.txt", "a") as f:
        f.write(risk)
        output_all = f.read()

    prompt = f"This is some information about a patient: {output_all}. This information includes information about two prospective health insurance plans they have access to and whether or not they have a high likelihood of suffering a heart attack. Considering the terms in both health insurance plans, which one should the user choose to save money? Respond ONLY with the name of the insurance plan and clear reasons why to. If both plans are equally economical, explain the pros and cons of both plans."
    openai.api_key = os.getenv("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")
    openai.Completion.create(
    model="gpt-3.5-turbo-instruct",
    prompt=prompt,
    max_tokens=100,
    temperature=0
)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6666)
    