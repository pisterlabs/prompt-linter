from flask import Flask, request
import openai
import os 
from twilio.rest import Client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('.env')


app = Flask(__name__)

openai.api_key =  os.getenv('OPENAI_API_KEY')

account_sid = os.getenv('TWILIO_ACCOUNT_SID')
auth_token = os.getenv('TWILIO_AUTH_TOKEN')
twilio_client = Client(account_sid, auth_token)
twilio_number = os.getenv('TWILIO_NUMBER')
recepient_number= os.getenv('RECEPIENT_NUMBER')

@app.route('/message', methods=['POST'])
def initiate_conversation():

    message_body = request.form.get('Body', '')

    store_conversation( message_body)

    return ''

def store_conversation(message_body):

    messages = [{"role": "user", "content": message_body}]
    messages.append({"role": "system", "content": "You're a lawyer who knows nothing but the rights of women according to Pakistani Law. You reply in Roman Urdu"})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0.5
    )

    chatgpt_response = response.choices[0].message.content

    send_message( chatgpt_response)

def send_message( message):
    twilio_client.messages.create(
        body=message,
         from_= twilio_number,
        to= recepient_number
    )

if __name__ == '__main__':
    app.run()
