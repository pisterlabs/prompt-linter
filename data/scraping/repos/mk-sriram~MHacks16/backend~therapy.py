from openai import OpenAI
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import base64
import os

matplotlib.use('agg')
client = OpenAI()
 # This is the API key for the OpenAI API REMOVE IT BEFORE COMMITTING

user_msg = ''

messages=[
    {"role": "system", "content": "You are an experienced therapist. You will inquire about the user's problems and provide guidance to the user. dont always talk within 100 words, 100 words should be the max, but it should be variable from 5 to 100 words ."},
    {"role": "system", "content": "Is there anything you would like to talk about today?"},
]
'''
messages = [
    {"role": "system", "content": "Imagine you are a therapist engaging in a session with a client who is needs to process life events and has difficulty expressing their emotions. The client begins to share their experiences and feelings during the session. Demonstrate the qualities of a therapist, with a particular emphasis on active listening. Respond in a way that reflects your understanding of the client's emotions and encourages them to explore their thoughts and feelings further."},
    {"role": "system", "content": "Is there anything you would like to talk about today?"},
    ]
    '''
emotions =[]
model = "gpt-4-1106-preview"
def add_emotion(emotion):
    global emotions
    emotions.append(emotion)
def get_therapist_message():
    completion = client.chat.completions.create(
        model=model,
        messages=messages
    )
    messages.append({"role": "system", "content": completion.choices[0].message.content})
    return completion.choices[0].message.content
def emotion_from_text(text):
    '''
    Determines the emotion from the text
    '''
    this_message = [{"role": "system", "content": 'Determine the emotion of the following text. Your options are (anger, fear, joy, love, sadness, surprise). Reply with only the name of the emotion: '}, {"role": "user", "content": text}]
    completion = client.chat.completions.create(
        model=model,
        messages=this_message
    )
    return completion.choices[0].message.content

def post_user_message(msg, use_emotion=False):
    if use_emotion:
        msg += 'My current emotion is ' + emotions[-1] + '.'
    messages.append({"role": "user", "content": msg})
def post_system_message(msg):
    messages.append({"role": "system", "content": msg})
def plot_sentiment_graph():
    global emotions
    print(emotions)
    if not emotions:
        return None
    # graph of emotions with frequency
    fig, ax = plt.subplots()
    frequencies = {"anger": 0, "fear": 0, "joy": 0, "love": 0, "sadness": 0, "surprise": 0}
    for emotion in emotions:
        if emotion in frequencies:
            frequencies[emotion] += 1
    emotions_list = list(frequencies.keys())  # Convert to list
    

    # Plot the graph
    ax.bar(emotions_list, frequencies.values())
    
    # convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read()).decode('utf-8')  # Decode bytes to a UTF-8 string
    return string
    


