from flask import Flask, render_template
from flask import Flask, flash, redirect, render_template, \
     request, url_for
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from config import key
import openai
import os


app = Flask(__name__)
ls = []

UPLOAD_FOLDER = 'audio_files'

#Add XFLOW Key here
openai.api_key = key


@app.route('/')
def audio_file():
     return render_template('audio.html')


@app.route('/upload_audiofile', methods=['GET', 'POST'])
def upload_audiofile():
    if request.method == 'POST':
        try:
            file = request.files['file']
            if file:
                filename = secure_filename(file.filename)
                print('filename: ', filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                # print('hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
                audio_file1 = open('audio_files/' + filename, "rb")
                # prompt = '''
                #     Split the below paragraph based on full stop and then write sentiment and intent classification for every sentence.
                #     First write the sentence then sentiment and intent classification at the end of every sentence. 
                #      Write in this format: Sentence: (Write the sentence), Sentiment: (Write the sentiment of the sentence) 
                #      and Intent : (Write the intent of the sentence)
                #     '''
                prompt = '''
                    Based on the below paragraph, Write sentence and after that write the sentiment and intent classification
                    at the end of each sentence. Write in this format Sentence: Sentiment: and Intent: 
                    '''

                transcript1 = openai.Audio.transcribe("whisper-1", audio_file1)
                prompt = prompt + '\n' + transcript1['text']

                gptresponse = get_gpt3_response(prompt)
                print('gpt response: ', gptresponse)
                finaltext =  'Transcription: ' + '\n\n' + transcript1.text + '\n\n' + gptresponse

                print(finaltext)

                return render_template('audio.html', text=finaltext)
        except Exception as e:
            # Handle the exception
            print('An error occurred:', str(e))
            return "Error occurred during file upload."

    return render_template('audio.html')




def get_gpt3_response(prompt, response=""):
    content = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
          {"role": "assistant", "content": response},
          {"role": "user", "content": prompt}

        ]
    )
    return content['choices'][0]['message']['content']



if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.run(debug=False, host='0.0.0.0')
