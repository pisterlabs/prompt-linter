from fitbit_user import FitbitUser
import openai
from flask import Flask, request
import os
import requests
from openai import OpenAI, GPT
import json
from langchain import LLMChain, PromptTemplate, FewShotPromptTemplate
from langchain.prompts.chat import (ChatPromptTemplate, SystemMessagePromptTemplate,
                                   AIMessagePromptTemplate, HumanMessagePromptTemplate)
from langchain.chat_models import ChatOpenAI

openai.api_key = "your_openai_api_key"
FITBIT_CLIENT_ID = os.getenv('FITBIT_CLIENT_ID', 'your_fitbit_id')
FITBIT_CLIENT_SECRET = os.getenv('FITBIT_CLIENT_SECRET', 'your_fitbit_secret')
FITBIT_REDIRECT_URL = os.getenv('FITBIT_REDIRECT_URI', 'your_fitbit_redirect_uri')

# Set up Azure HealthBot credentials
AZURE_HEALTHBOT_SECRET = 'your_healthbot_secret'

app = Flask(__name__)

@app.route('/get_data')
def get_data():
    user = FitbitUser(app)
    user_data = user.get_all_data()
    return user_data

@app.route('/get_recommendation', methods=['POST'])
def get_recommendation():
    query = request.json.get('query')
    user_data = get_data()
    # Generate recommendation using OpenAI model
    recommendation = generate_recommendation(query, user_data)

    # Further process the recommendation using Azure Healthbot API
    azure_response = connect_to_azure(recommendation)
    
    return azure_response



def generate_recommendation(query, data):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    gpt = GPT(version="4")

    prompt = f"Based on the following health and Fitbit data, what health recommendation would you provide?\n\n{data}\n\nQuery: {query}\n\nRecommendation:"

    response = gpt.complete(prompt)

    return response.choices[0].text.strip()



def connect_to_azure(recommendation):
    direct_line_secret = os.getenv("DIRECT_LINE_SECRET")
    base_url = "https://directline.botframework.com/"

    # start conversation
    response = requests.post(
        f"{base_url}v3/directline/conversations",
        headers={"Authorization": f"Bearer {direct_line_secret}"},
    )
    conversation = json.loads(response.text)

    # send message
    response = requests.post(
        f"{base_url}v3/directline/conversations/{conversation['conversationId']}/activities",
        headers={"Authorization": f"Bearer {direct_line_secret}"},
        json={
            "type": "message",
            "from": {"id": "user1"},
            "text": recommendation,
        },
    )
    return response.json()


if __name__ == '__main__':
    app.run(debug=True)
