from flask import Flask, request, render_template, Response, make_response
import os

import logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

import datetime
import shutil
import sys

import time
import openai
import dropbox
import requests

# remove all files under static folder if they exist and create static folder if it doesn't exist
try:
    shutil.rmtree('static')
except:
    pass
os.mkdir('static')


print("Loading libraries...")
from elevenlabslib import *

# *********** CONFIG ***********
from configuration import *

openai.api_key = OPENAI_API_KEY
user = ElevenLabsUser(ELEVENLABS_API_KEY)
dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)

# *********** END API KEYS ***********

# check if subscription still has enough credits for ElevenLabs voice synthesis
print("Checking ElevenLabs Speech Synthesis credits...")
chars_used = user.get_current_character_count()
char_limit = user.get_character_limit()
can_extend = user.get_can_extend_character_limit()
monthly_payment = user.get_next_invoice()['amount_due_cents'] / 100
print('ElevenLabs: chars used: ' + str(chars_used))
print('ElevenLabs: char limit: ' + str(char_limit))
print('ElevenLabs: can extend: ' + str(can_extend))
print('ElevenLabs: monthly payment: $' + str(monthly_payment))

# if there are less than 100 characters left, stop the program
if char_limit - chars_used < 100:
    print('Not enough ElevenLabs Speech Synthesis credits left to continue')
    exit()

voice = None
use_whisper = True
# Keep track of OpenAI API usage and display cost of the session
reply_count = 0
tokens_used = 0

# ChatGPT API call:
# This code defines a function named 'get_reply' that takes a list of messages as input.
# The function calls the OpenAI API using the 'openai.ChatCompletion.create' function to
# generate a response based on the input message.
# The reply from the OpenAI API is stored in the 'reply' variable,
# and the total number of tokens used in the API call is stored in the 'tokens' variable.
# The function then increments the 'tokens_used' global variable by the number
# of tokens used in the API call and calculates the cost of the API call based on the number of tokens used.
# Finally, the function appends the response to the 'messages' list,
# prints the number of tokens used and the cost, and returns the generated reply.
def get_reply(messages):
    global tokens_used
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=messages
    )
    reply = response['choices'][0]['message']['content']
    tokens = response['usage']['total_tokens']
    # The ChatGPT API costs $0.002 per 1,000 tokens. So we need to keep track of how many tokens we use.
    tokens_used += tokens
    cost = tokens_used / 1000 * 0.002
    print('tokens: ' + str(tokens_used) + ' Total used this session: ' + str(tokens_used) + ' Cost: $' + str(cost))
    messages.append({'role': 'assistant', 'content': reply})
    return reply


def elevenlabs_generate(text):
    mp3_bytes = None
    while True:
        try:
            mp3_bytes = voice.generate_audio_bytes(text,stability=0.35,similarity_boost=0.95)
            break
        except:
            print('Error connecting to ElevenLabs API, retrying...')
            time.sleep(5)

    return mp3_bytes

# Upload a file to dropbox and get a shared link.
def share_dropbox(filename):
    with open(filename, 'rb') as f:
        dbx.files_upload(f.read(), '/' + filename, mute=True, mode=dropbox.files.WriteMode.overwrite)

    # share the file via link that can be downloaded via wget,
    shared_link_metadata = dbx.sharing_create_shared_link_with_settings('/' + filename)
    shared_link = shared_link_metadata.url
    # This creates a www.dropbox.com link that can be downloaded
    # via wget or curl so we need to convert it to a dl.dropbox.com link
    shared_link = shared_link.replace('www.dropbox.com', 'dl.dropbox.com')
    # also remove the ?dl=0 at the end
    shared_link = shared_link.replace('?dl=0', '')
    return shared_link

# Create an animated photo using d-id api.
# This should only be done once per character
# and the file saved as animation.mp4 and placed in the character's directory.
# This animation.mp4 can also be created for free with the https://www.myheritage.com/ website.
def get_animation(photo_url):
    # Set the URL and headers for the API request
    url = 'https://api.d-id.com/animations'
    headers = {
        'Authorization': 'Basic ' + DID_API_KEY,
        'Content-Type': 'application/json'
    }

    # Set the parameters for the API request
    source_url = photo_url
    driver_url = 'bank://fun'
    config = {'mute': True}

    # Send the API request to create the animation
    # Expected Response
    # JSON
    #
    # {
    #    'id':'<id>',
    #    'created_at':'<time>',
    #    'status':'created',
    #    'object':'animation'
    # }

    response = requests.post(url, headers=headers, json={
        'source_url': source_url,
        'driver_url': driver_url,
        'config': config
    })
    print(response)
    # Check if the request was successful
    if response.status_code == 201 or response.status_code == 200:
        id = response.json()['id']
        print('Animation created, id:', id)
        # Send the API request to check the status of the animation and get the result URL
        # Expected Response
        # JSON
        #
        # {
        #    'result_url':'<result_url>',
        #    'metadata':{...},
        #    'source_url':'https://.../david.jpg',
        #    'status':'done',
        #    'driver_url':'bank://...',
        #    'modified_at':'<time>',
        #    'user_id':'<user_id>',
        #    'id':'<id>',
        #    'started_at':'<time>'
        # }
        while True:
            response = requests.get(url + '/' + id, headers=headers)
            if response.status_code != 200:
                print('error getting animation: ', response.json())
                quit()
            # Print the response
            print('animation created, checking for result url in: ', response.json())
            if 'result_url' in response.json():
                break
            time.sleep(1)
        # Get the result URL
        result_url = response.json()['result_url']
        print('Animation result URL:', result_url)
        # download the mp4 file
        r = requests.get(result_url, allow_redirects=True)
        open(os.path.join(CHARACTER, 'animation.mp4'), 'wb').write(r.content)
        # play the mp4 file
        # PlayVideo('animation.mp4')
    else:
        print(f'Error: {response.status_code} - {response.text}')


# transcribe an MP3 file using OpenAI Whisper API
def transcribe_audio(mp3_file):
    audio_file = open(mp3_file, 'rb')
    # make sure the audio file is not empty and is long than Minimum audio length is 0.1 seconds.
    audio_file.seek(0, 2)
    if audio_file.tell() == 0:
        audio_file.close()
        return '..... silence'
    audio_file.seek(0)

    while True:
        try:
            transcript = openai.Audio.transcribe(model='whisper-1', file=audio_file,
                                                 prompt="Umm, let me think like, hmm... Okay, here's what I'm, like, thinking.")[
                'text']
            break
        except openai.error.InvalidRequestError as e:
            print('Audio file is too short, retrying, returning silence: ', e)
            return '..... silence'
        except Exception as e:
            print('error transcribing audio, retrying: ', e)
            time.sleep(5)
    audio_file.close()
    return transcript

voice = user.get_voices_by_name(VOICE_NAME)[0]  # This is a list because multiple voices can have the same name
print('voice: ', voice)

# speak_avatar('This is a test', photo_url) # test speaking video avatar using d-id api.

# delete old audio/video input/output files in current directory in a cross platform way (mp3, wav, mp4)
try:
    for f in os.listdir('.'):
        if f.endswith('.mp3') or f.endswith('.wav') or f.endswith('.mp4'):
            os.remove(f)
except:
    pass

# Get the photo of the CHARACTER from photo.jpg (only used for the video avatar, which is disabled for now)
# if the CHARACTER doesn't have an animation.mp4 file, then create one
if not os.path.exists(os.path.join(CHARACTER, 'animation.mp4')):
    photo_file = CHARACTER + '/photo.jpg'
    photo_url = share_dropbox(photo_file)
    get_animation(photo_url) # get animation.mp4 for a new avatar image

# copy animation.mp4 to to the output folder
animation_src_path = os.path.join(CHARACTER, 'animation.mp4')
animation_dst_path = os.path.join('static', 'animation.mp4')
shutil.copy(animation_src_path, animation_dst_path)


messages = []  # This will hold the conversation history
# Load character's personality from personality.txt
with open(os.path.join(CHARACTER, 'personality.txt'), 'r') as f:
    personality = f.read()

# Get the Name of the CHARACTER from first line of personality.txt format is eg: Name: Jason Pottinger
full_name = personality.splitlines()[0].split(':')[1].strip()
print('Full Name: ', full_name)

system_message = personality  # System message defines some background information about the character
# Append current date and time to the system message
system_message += 'Current date and time: ' + datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")

messages.append({'role': 'system', 'content': system_message})

# Add some initial back/forth dialogue
# to establish the identity of the CHARACTER and give ChatGPT some context
message = "Hello! Who are you?"
messages.append({'role': 'user', 'content': message})
assistant_message = "Well, um, I am: " + system_message
messages.append({'role': 'assistant', 'content': assistant_message})
message = "That's cool! So now I understand that you are " + full_name + ". Ok, let's start chatting and you play the role of " + full_name + " and stay in that role, speaking naturally, just as you would in real life."
messages.append({'role': 'user', 'content': message})
assistant_message = "Ok, so uh, let's start chatting and I will play the role of " + full_name + " and stay in that role. I will emulate their speech realistically and naturally (including a lot of uhs and umms to be realistic) , personality and their interests/likes/dislikes, etc."
messages.append({'role': 'assistant', 'content': assistant_message})



reply_count = 0



# Takes user input and returns a reply from ChatGPT API
def handle_message(message, save_audio=True):
    global reply_count
    messages.append({'role': 'user', 'content': message})  # add user message to messages list
    reply = get_reply(messages)  # Get the reply from ChatGPT API
    messages.append({'role': 'assistant', 'content': reply})  # add reply to messages list
    print(reply)
    # remove the text before the : from the reply (eg: Jason: Hello, how are you? -> Hello, how are you?) but keep the rest of the reply including : and spaces
    reply = reply[reply.find(':') + 1:]

    # elevenlabs_speak(reply)  # ElevenLabs API for streaming text to speech (server side playback)
    mp3_bytes = elevenlabs_generate(reply)  # ElevenLabs API for generating mp3 file (client side playback)
    with open(os.path.join('static', 'chatbot_output.mp3'), 'wb') as f:
        f.write(mp3_bytes)

    # Note: Talking Avatar API is not used in this example but code is included for reference
    # The reason is that it is very expensive. Only useful for demo purposes but not
    # for longer term use. Instead, it's more economical to just create an animation.mp4
    # This can be done via the d-id API
    # or going to the https://www.myheritage.com/ website and creating for free.
    # speak_avatar(reply, photo_url) # d-id API for streaming text to speech with video

    reply_count += 1
    if save_audio:
        # copy the mic audio and the reply audio/video files to the output folder, and append transcript to conversation.txt
        shutil.copy(os.path.join('static', 'recorded_audio.mp3'),
                    os.path.join('static', 'recorded_audio_' + str(reply_count) + '.mp3'))
        shutil.copy(os.path.join('static', 'chatbot_output.mp3'),
                    os.path.join('static', 'chatbot_output_' + str(reply_count) + '.mp3'))

        # Uncomment the following line to copy the avatar video file to the output folder (disabled by default, too expensive)
        # shutil.copy('avatar.mp4', os.path.join('static','avatar_' + str(reply_count) + '.mp4'))

    # Conversation is formatted as 'User: message\nAI: reply\n\n'
    conversation_history = 'User: ' + message + '\nAI: ' + reply + '\n\n'

    # append to conversation.txt
    with open(os.path.join('static', 'conversation.txt'), 'a') as f:
        f.write(conversation_history)

    return reply


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/save-audio", methods=["POST"])
def save_audio():
    global reply_count

    # delete chatbot_output.mp3 and recorded_audio.mp3 files
    try:
        os.remove(os.path.join('static', 'chatbot_output.mp3'))
        os.remove(os.path.join('static', 'recorded_audio.mp3'))
    except:
        pass

    # Get the recorded audio data from the request
    audio_data = request.data
    print('audio_data bytes: ', len(audio_data))

    save_path = os.path.join('static', 'recorded_audio')
    ogg_file = os.path.join(save_path + '.ogg')
    mp3_file = os.path.join(save_path + '.mp3')

    # Save the audio data to a file
    try:
        os.remove(ogg_file)
        os.remove(mp3_file)
    except:
        pass

    with open(ogg_file, 'wb') as f:
        f.write(audio_data)
    # Convert the audio file to mp3
    os.system('ffmpeg -i ' + ogg_file + ' ' + mp3_file)

    # Get the transcript of the audio
    transcript = transcribe_audio(mp3_file)
    print('transcript: ', transcript)
    reply = handle_message(transcript)

    response_text = transcript + '\n' + reply + '\n\n'
    return Response(response_text, mimetype="text/plain")



@app.route('/chatbot-tts-audio', methods=['GET'])
def serve_audio():
    # wait until the chatbot_output.mp3 file is created
    while not os.path.exists(os.path.join('static', 'chatbot_output.mp3')):
        time.sleep(0.1)
    # wait until the chatbot_output.mp3 file is finished writing
    while os.stat(os.path.join('static', 'chatbot_output.mp3')).st_size == 0:
        time.sleep(0.1)
    # wait until it is safe to read the file
    time.sleep(0.1)

    audio_file = open(os.path.join('static', 'chatbot_output.mp3'), 'rb').read()
    response = make_response(audio_file)
    response.headers['Content-Type'] = 'audio/mpeg'
    response.headers['Cache-Control'] = 'no-cache'
    return response

@app.route('/send-message', methods=['POST'])
def send_message():

    # delete chatbot_output.mp3 and recorded_audio.mp3 files
    try:
        os.remove(os.path.join('static', 'chatbot_output.mp3'))
        os.remove(os.path.join('static', 'recorded_audio.mp3'))
    except:
        pass

    print('send_message')
    # get the message from the request
    # read json data from request
    data = request.get_json()
    print('data: ', data)
    message = data['message']
    print('message: ', message)
    reply = handle_message(message, save_audio=False)

    # send the message to the client
    response = make_response(message + '\n' + reply + '\n\n')
    response.headers['Content-Type'] = 'text/plain'
    response.headers['Cache-Control'] = 'no-cache'
    return response

if __name__ == "__main__":
    host = '0.0.0.0'
    #context = ('local.crt', 'local.key')  # certificate and key files
    #context = 'adhoc'
    #app.run(debug=True, host=host, port=8080, ssl_context=context)
    app.run(debug=True, host=host, port=8080)
