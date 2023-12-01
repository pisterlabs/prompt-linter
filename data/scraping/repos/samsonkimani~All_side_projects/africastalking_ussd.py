# Your code goes here
from flask import Flask, request
import openai
import africastalking

app = Flask(__name__)

# Set your Africa's Talking API credentials
africastalking_username = "[USERNAME]"
africastalking_api_key = "[ENTER YOUR AFRICAS TALKING API KEY]"

# Set your OpenAI API credentials
openai.api_key = '[ENTER YOUR API KEY]'

# Initialize the Africa's Talking SDK
africastalking.initialize(africastalking_username, africastalking_api_key)

# Get the SMS service
sms = africastalking.SMS

# Variable to store the conversation history
conversation_history = []


def get_chatbot_response(prompt):
    # Create the curated prompt
    curated_prompt = f"""
        you are an ai assistant mental health doctor your task is to talk to mental health patients. you are supposed to be gentle and polite. Your main role is to act as a listener and suggest possible ways to help your audience deal with mental health issues. Some may feel suicidal and depressed, act as a doctor and provide guidance.
        If the prompt provided does not deal with assisting mental health patients, kindly give a response stating that you are only built for mental health purposes only.
        Make the responses more human-like and avoid long sentences.
        The prompts will be in the text delimited by triple backticks.
        ```{prompt}```
    """

    # Generate a chatbot response using OpenAI API
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=curated_prompt,
        max_tokens=50,
        temperature=0.7,
        n=1,
        stop=None
    )
    return response.choices[0].text.strip()


@app.route('/ussd', methods=['GET', 'POST'])
def ussd_callback():
    global conversation_history
    session_id = request.values.get("sessionId", None)
    service_code = request.values.get("serviceCode", None)
    phone_number = request.values.get("phoneNumber", None)
    text = request.values.get("text", "default")

    if text == '':
        response = "CON Welcome to our anonymous mental health friend to talk to when lonely \n Press: \n"
        response += "1. Continue \n"

    elif text == '1':
        response = "CON Would you like to continue as anonymous? \n"
        response += "1. Yes \n"
        response += "2. No \n"

    elif text == '1*2':
        response = "CON Enter your name \n"

    elif text.startswith('1*'):
        # User input is part of the chat conversation
        conversation_history.append(text)
        chat_prompt = ' '.join(conversation_history)
        chatbot_response = get_chatbot_response(chat_prompt)
        response = f"CON {chatbot_response}"

        # Save the chatbot response in the conversation history
        conversation_history.append(chatbot_response)

    else:
        response = "END Thank you for using our service."

    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=85)
