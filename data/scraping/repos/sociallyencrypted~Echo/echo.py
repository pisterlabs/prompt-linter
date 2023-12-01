import os
import uuid
from flask import Flask, flash, request, redirect, render_template, jsonify
from speech_recognition import Recognizer, AudioFile
from pydub import AudioSegment
from dotenv import load_dotenv
import openai

UPLOAD_FOLDER = 'files'
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
openai.api_key = os.getenv('CHATGPT_API_KEY')

# save the user chat into a list of tuples.

chat = []


@app.route('/')
def root():
    return render_template('index.html', text="Transcription will be displayed here")

@app.route('/history')
def history():
    return render_template('historyindex.html')

@app.route('/chats')
def chats():
    return chat



@app.route('/save-record', methods=['POST'])
def save_record():

    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    file_name = str(uuid.uuid4()) + '.wav'
    full_file_name = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    file.save(full_file_name)
    
    pcm_file_name = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()) + '_pcm.wav')
    audio_segment = AudioSegment.from_file(full_file_name)
    audio_segment = audio_segment.set_channels(1)  # Convert to mono
    audio_segment.export(pcm_file_name, format="wav")

    os.remove(full_file_name)


    recognizer = Recognizer()
    with AudioFile(pcm_file_name) as source:
        audio_data = recognizer.record(source)
        print('Recognizing...')
        text = recognizer.recognize_google(audio_data)
        print(text)
        chat.append(('other', text))
        response  = openai.Completion.create(
            model="text-davinci-003",
            prompt=text,
            max_tokens=20, 
            temperature=0.5,  
            n=3 
        )
        os.remove(pcm_file_name)
        response = [r['text'] for r in response['choices']]
        return jsonify(text=text, predict=response)
    
@app.route('/chat', methods=['POST'])
def chatadd():
    if 'text' not in request.form:
        flash('No text')
        return redirect(request.url)
    text = request.form['text']
    chat.append(('user', text))
    print(chat)
    return jsonify(text=text)
        
    
if __name__ == '__main__':
    app.run(port=31337, debug=True)