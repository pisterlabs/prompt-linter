# import os
import json
import openai
import requests
import urllib.parse
from flask import Flask, request, make_response
from twilio.twiml.voice_response import VoiceResponse

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/answer", methods=['GET', 'POST'])
def answer_call():
    """Respond to incoming phone calls with a brief message."""
    # Start our TwiML response
    resp = VoiceResponse()

    # Read a message aloud to the caller
    resp.say("Thank you for calling! Have a great day.", voice='Polly.Amy')

    return str(resp)

@app.route("/transcribe", methods=['GET', 'POST'])
def transcribe():
    # Create a TwiML Voice Response object to build the response
    twiml = VoiceResponse()

    # If no previous conversation is present, or if the conversation is empty, start the conversation
    if not request.cookies.get('convo'):
        # Greet the user with a message using AWS Polly neural voice
        twiml.say("Hey! I'm Joanna, a chatbot created using Twilio and ChatGPT. What would you like to talk about today?",
                 voice='Polly.Joanna-Neural')

    # Listen to the user's speech and pass the input to the /respond Function
    twiml.gather(
        timeout='auto', # Automatically determine the end of user speech
        model='experimental_conversations', # Use the conversation-based speech recognition model
        input='speech', # Specify the speech as the input type
        action='/respond', # Send the collected input to /respond
    )

    # Create a Twilio Reponse object
    response = make_response(str(twiml))

    # Set the response content type to XML (TwiML)
    response.headers['Content-Type'] = 'application/xml'

    # Set the response body to the generated TwiML


    # If no conversation cookie is present, set an empty conversation cookie
    if not request.cookies.get('convo'):
        response.set_cookie('convo', '', path='/')

    # Return the response to Twilio
    return response

@app.route("/respond", methods=['POST'])
def respond():
    # Set up the Twilio VoiceResponse object to generate the TwiML
    twiml = VoiceResponse()

    # Parse the cookie value if it exists
    cookie_value = request.cookies.get('convo')
    cookie_data = json.loads(cookie_value)

    # Get the user's voice input from the event
    voice_input = request.form.get('SpeechResult')
    
    # Create a conversation variable to store the dialog and the user's input to the conversation history
    conversation = cookie_data['conversation']
    conversation.append(f"user: {voice_input}")

    # Get the AI's response based on the conversation history
    ai_response = generate_ai_response(conversation.join(';'))

    # For some reason the OpenAI API loves to prepend the name or role in its responses, so let's remove 'assistant:' 'Joanna:', or 'user:' from the AI response if it's the first word
    cleaned_ai_response = ai_response.replace("/^\w+:\s*/i", "").strip()

    # Add the AI's response to the conversation history
    conversation.append(f"assistant: {cleaned_ai_response}")

    # Limit the conversation history to the last 10 messages; you can increase this if you want but keeping things short for this demonstration improves performance
    conversation = conversation[-10:]

    # Generate some <Say> TwiML using the cleaned up AI response
    twiml.say(cleaned_ai_response, voice='Polly.Joanna-Neural')

    # Redirect to the Function where the <Gather> is capturing the caller's speech
    twiml.redirect(url='/transcribe', method='POST')

    # Since we're using the response object to handle cookies we can't just pass the TwiML straight back to the callback, we need to set the appropriate header and return the TwiML in the body of the response
    response = make_response(str(twiml))
    response.headers['Content-Type'] = 'application/xml'

    # Update the conversation history cookie with the response from the OpenAI API
    new_cookie_value = urllib.parse.quote(json.dumps({'conversation': conversation}))
    response.set_cookie('convo', new_cookie_value, path='/')

    # Return the response to the handler
    return response

if __name__ == "__main__":
    app.run(port=8000, debug=True)
