import os
import time
from datetime import datetime
import configparser
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, jsonify, url_for, session
import openai
from gtts import gTTS
import secrets
import csv
import uuid

# Loading OpenAI API key from configuration file
config = configparser.ConfigParser()
config.read('config.ini')
openai.api_key = config.get('OPENAI_API', 'key')

# Development mode, unable requests to OpenAI
DEV_MODE = False
# Used when testing the app with the flask server locally, needs a signed certificate
DEV_MODE_APP = False

# Initializing Flask app
app = Flask(__name__)
# Setting up paths for upload and audio directories
app.config['UPLOAD_FOLDER'] = "static/audio/"  
app.config['AUDIO_FOLDER'] = "static/audio/"
# Generating a secret key for the session
app.config['SECRET_KEY'] = secrets.token_hex(16)

@app.route('/')
def index():
    # Serving the initial index page
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    # Transcribing uploaded audio file
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    # Securing the filename and saving it in the defined upload directory
    audio_file = request.files['audio']
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = f"recording_{timestamp}_{uuid.uuid4()}.webm"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio_file.save(file_path)

    # Waiting for the file to be completely written to the disk
    wait_for_file(file_path)

    # Transcribing the audio file using OpenAI's whisper model
    input_language = request.form['input_language']
    
    if(DEV_MODE):
        transcript = "This is DEV mode."
    else:
        transcript = transcribe(file_path, input_language)

    # Save the transcript to the CSV file along with the IP address and User-Agent
    user_agent = request.headers.get('User-Agent', 'Unknown')  # Default to 'Unknown' if User-Agent header is missing
    save_to_csv(transcript, request.remote_addr, user_agent)
        
    return jsonify({'transcript': transcript})

@app.route('/translate', methods=['POST'])
def translate_audio():
    # Translating provided text and converting it into speech
    req_data = request.get_json()

    # If the selected output language is 'auto'
    if req_data['output_language'] == 'auto':
        return jsonify({
            'audio_url': '', 
            'translation': "The output language cannot be set to 'auto'"
        })
        
    if(DEV_MODE):
        translation = "This is DEV mode."
        
        return jsonify({
            'audio_url': url_for('static', filename=f'audio/dev_audio.m4a'), 
            'translation': translation
        })
    else:
        # Translating the text
        translation = translate(
            req_data['text'], 
            input_language=req_data['input_language'], 
            output_language=req_data['output_language']
        )
        
        # Converting the translated text into speech
        tts = gTTS(translation, lang=req_data['output_language'])

        # Remove the previous audio file
        if not session.get('last_audio_file', None) == None:
            if os.path.exists(session.get('last_audio_file', '')):
                os.remove(session.get('last_audio_file', ''))
            
        # Saving the speech file to the audio directory
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = f"text2speech_{timestamp}.mp3"
        file_path = os.path.join(app.config['AUDIO_FOLDER'], filename)
        tts.save(file_path)

        wait_for_file(file_path)

        # Storing the path of the last audio file in the session
        session['last_audio_file'] = file_path
        
        return jsonify({
            'audio_url': url_for('static', filename=f'audio/{filename}'), 
            'translation': translation
        })

@app.route('/audio', methods=['GET'])
def get_last_audio():         
    if(DEV_MODE):
        return jsonify({
            'audio_url': url_for('static', filename=f'audio/dev_audio.m4a')
        })
    else:
        # Returning the path of the last audio file from the session
        return jsonify({'audio_url': session.get('last_audio_file', '') })

def transcribe(file_path, input_language):
    # Transcribing audio using OpenAI's whisper model
    with open(file_path, "rb") as audio_file:
        
        if(input_language == "auto"):
            transcript = openai.Audio.transcribe(
                "whisper-1", audio_file
            )
        else:
            transcript = openai.Audio.transcribe(
                "whisper-1", audio_file, language=input_language
            )
    return transcript['text']

def translate(text, input_language, output_language):
    # Translating text using OpenAI's gpt-3.5-turbo model
    if input_language == "auto":
        messages = [
            {
                "role": "system", 
                "content": (
                    f"You are a helpful AI translator. You will receive a transcribe in "
                    f"'and you have to translate in '{output_language}'."
                    "The translation should be in spoken language. Only reply with the direct translation."
                )
            },
            {"role": "user", "content": f"Transcribe: {text}\nTranslation:"}
        ]
    else:
        messages = [
            {
                "role": "system", 
                "content": (
                    f"You are a helpful AI translator. You will receive a transcribe in "
                    f"'{input_language}', and you have to translate in '{output_language}'. "
                    "The translation should be in spoken language. Only reply with the direct translation."
                )
            },
            {"role": "user", "content": f"Transcribe: {text}\nTranslation:"}
        ]
    
    translation = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    
    return translation['choices'][0]['message']['content']

def wait_for_file(file_path):
    # Wait for file to exist and to be non-empty before proceeding
    while not os.path.exists(file_path) or not os.path.getsize(file_path) > 0:
        time.sleep(0.1)

def save_to_csv(transcript, ip_address, user_agent, filename="history/transcripts.csv"):
    # Check if the directory exists, if not, create it
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        # The current time, transcript, IP address, and User-Agent are saved
        writer.writerow([datetime.now(), transcript, ip_address, user_agent])



# Run the Flask app
if __name__ == '__main__':
    if DEV_MODE_APP:
        app.run(ssl_context=('cert.pem', 'key.pem'), debug=True, host="0.0.0.0", port=5009)