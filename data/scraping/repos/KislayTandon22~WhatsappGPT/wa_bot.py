from flask import Flask,request
from dotenv import load_dotenv
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
import os
import openai
openai.api_key=os.getenv("Openapi_key")
def ai_reposnse(prompt: str) -> dict:
    # directly I copied it from the open api site
    response = openai.Completion.create(
            model='text-davinci-003',
            prompt=f'Human: {prompt}\nAI: ',
            temperature=0.9,
            max_tokens=150,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0.6,
            stop=['Human:', 'AI:']
        )
    return {
        # we are returning the custom data structure to know the succesful execution if status is correct then it will be 1
        'status': 1,
        'response': response['choices'][0]['text']
    }
    
load_dotenv()# to take varibales from env file
sid=str(os.getenv('Twillio_sid'))
token=str(os.getenv('Twillio_token'))
'''we are using flask to creat web app and we will host the web app using ngrok we need to do so we can recieve 
the webhook request what twilio will give once it recieve the message'''
app=Flask(__name__)
client=Client(sid,token)
'''this function will basically get all the required variable like message body recipent that is the number you use to send query
and the whatsapp number assigned to you by twilio when message comes in box with setting as post and add "/message" to end of line'''
def send_message(to: str, message: str) -> None:
    _ = client.messages.create(
        from_=os.getenv('FROM'),
        body=message,
        to=to
    )

@app.route('/')
def home():
    return 'Home'
'''this the web app we will connect. to connect we will start ngrok with same port number as the one runn file uses to run the app
now we will get a receiving link copy it to snadbox '''
@app.route('/message',methods=['POST'])
def reply():       # we will get details from the web
    message = request.form['Body']
    sender_id = request.form['From']
    print(sender_id)
        # Get response from Openai
    result = ai_reposnse(message)
    if result['status'] == 1:
        return send_message(sender_id, result['response'])
    else:
        return "Error"
    

