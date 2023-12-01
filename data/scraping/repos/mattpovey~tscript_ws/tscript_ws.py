import os
import openai
import tempfile
import json
import requests
import logging
import logging.handlers
import tempfile
from support_funcs import gpt_proc, split_text, tok_count, rename_m4a, split_audio, convert_to_wav, process_transcription
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import BadRequest
from datetime import timedelta

# Configure the Flask app
ts_flask_app = Flask(__name__)
CORS(ts_flask_app)
ts_flask_app.config['upload_dir'] = 'uploads'
ts_flask_app.config['allowed_exts'] = {'wav', 'mp3', 'ogg', 'm4a', 'mp4'}
ts_flask_app.config['model'] = 'gpt-3.5-turbo'
openai.api_key = os.getenv("OPENAI_API_KEY")

# Create a logger
logger = logging.getLogger('tscript_logger')
logger.setLevel(logging.DEBUG)

# Create a SysLogHandler
syslog_handler = logging.handlers.SysLogHandler(address='/dev/log')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ts_flask_app.config['allowed_exts']

@ts_flask_app.route('/')
def index():
    return render_template('index.html')

@ts_flask_app.route('/task_sel', methods=['GET', 'POST'])
def task_selection():
    # check that we have 'task'. If not, return to index.html
    if request.form.get('task_sel'):
        task = request.form.get('task_sel')
    else:
        return render_template('index.html')
    
    if task == "ts_proc":
        logger.info("TS_PROC - Task selected: " + task)
        return render_template('ts_proc.html')       
    elif task == "pod_sum":
        logger.info("TS_PROC - Task selected: " + task)
        return render_template('pod_sum.html')
    else:
        logger.info("TS_PROC - Task selected: " + task)
        return render_template('index.html')

@ts_flask_app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if request.method == 'POST': # Seems to hang sometimes after this. Doesnt' get into the next statement...  print("/transcribe received POST request") # Setup the file if 'file' not in request.files: return jsonify(error='No file part'), 400 print("No file part") file = request.files['file'] print(file.filename) # Setup options resp_format = request.form.get('output-format') print(resp_format) # Get engine to determine which API to use # openai_api or local_whisper engine = request.form.get('engine') model = request.form.get('model') lang = request.form.get('language') translate = request.form.get('translate') processing = request.form.get('processing') processing_role = request.form.get('processing-role') # Create a dictionary to store the output
        # Get engine to determine which API to use
        # openai_api or local_whisper
        file_url = request.form.get('file_url')
        engine = request.form.get('engine')
        model = request.form.get('model')
        lang = request.form.get('language')
        translate = request.form.get('translate')
        resp_format = request.form.get('output-format')
        # Check whether the file is a URL or a local file
        if file_url:
            print("File is a URL")
            # Try to download the file from the URL and return an error if it fails
            try:
                file = requests.get(file_url)
                # Set file.filename to the last part of the URL but strip anything after a ? or #
                file.filename = file_url.split('/')[-1].split('?')[0].split('#')[0]
                print("Audio file, " + file_url + " fetched: " + str(file.filename))
                logger.info("Audio file, " + file_url + " fetched: " + file.filename)
            except Exception as e:
                #print(e)
                logger.error("Error downloading file from URL: " + file_url)
                return jsonify(error='Error downloading file from URL'), 500
        else:
            print("File is passed from client")
            file = request.files['file']

            if 'file' not in request.files:
                logger.error("No file part")
                return jsonify(error='No file part'), 400
            
            logger.info("Audio file, " + file.filename + " received from client")

        # Create a dictionary to store the output
        output = {}

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(ts_flask_app.config['upload_dir'], filename)
            print("Saving file locally: " + file_path)
            if file_url:
                try:
                    print("Saving file from URL")
                    with open (file_path, 'wb') as f:
                        f.write(file.content)
                except Exception as e:
                    print(e)
                    return jsonify(error='Error saving file from URL'), 500
            else:
                print("Saving file from client")
                try:
                    file.save(file_path)
                except Exception as e:
                    print(e)
                    return jsonify(error='Error saving file'), 500

            # Create a dictionary to store the output for OpenAI transcriptions
            transcriptions = []

            if engine == 'openai_api':
                # Option is misspelled in index.html
                if resp_format == 'txt':
                    resp_format = 'text'
                print("Using OpenAI API")
                f_ext = file_path.rsplit('.', 1)[1].lower()

                # the ffmpeg library on MacOS has problems with m4a files
                # they can be renamed to mp4 without any issues
                if f_ext == 'm4a':
                    file_path = rename_m4a(file_path, 'mp4')
                    # update file_extension to mp4
                    print(file_path)
                    f_ext = file_path.rsplit('.', 1)[1].lower()

                # Split audio into chunks - split_audio returns a list of AudioSegment objects
                # file_ext=file_ext is gross. Shouldn't use same variable name.
                # OpenAI max size should be stored in one place...
                audio_chunks = split_audio(file_path, max_size_bytes=25 * 1024 * 1024, file_ext=f_ext)
                logger.info(f"{len(audio_chunks)} chunks created from {file_path} for processing using OpenAI API")
                print(f"Audio chunks: {len(audio_chunks)}")

                for index, chunk in enumerate(audio_chunks):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{f_ext}') as chunk_file:
                        chunk.export(chunk_file.name, format=f_ext)
                        print(f"Sending chunk {index + 1} to API: {chunk_file.name}")
                        with open(chunk_file.name, 'rb') as audio_file:
                            # Use openai.Audio.transcribe if 'translate' is false
                            # Use openai.Audio.translate if 'translate' is true
                            if translate == "true":     
                                print("Translating")
                                logger.info(f"Translating chunk {index + 1} of {len(audio_chunks)}")
                                # Translation is always to English and language detection is automatic and cannot be specified.
                                try:
                                    transcript = openai.Audio.translate(model="whisper-1", file=audio_file, response_format=resp_format)
                                except openai.error.APIError as e:
                                    print(f"An API error occurred: {e}")
                                    return jsonify({"error": "Error using OpenAI API", "details": str(e)}), 500  
                                except openai.error.APIConnectionError as e:
                                    print(f"Failed to connect to OpenAI API: {e}")
                                    return jsonify({"error": "Error connecting to the OpenAI API", "details": str(e)}), 500
                                except openai.error.RateLimitError as e:
                                    print(f"Rate limit exceeded: {e}")
                                    return jsonify({"error": "OpenAI API Rate limit exceeded", "details": str(e)}), 500
                                except openai.error.AuthenticationError as e:
                                    print(f"Authentication error: {e}")
                                    return jsonify({"error": "OpenAI API authentication error", "details": str(e)}), 500

                            else:
                                logger.info(f"Transcribing chunk {index + 1} of {len(audio_chunks)}")
                                # The API only takes the language parameter to when NOT translating.
                                try:
                                    transcript = openai.Audio.transcribe(model="whisper-1", file=audio_file, response_format=resp_format, language=lang)
                                except openai.error.APIError as e:
                                    print(f"An API error occurred: {e}")
                                    return jsonify({"error": "Error using OpenAI API", "details": str(e)}), 500  
                                except openai.error.APIConnectionError as e:
                                    print(f"Failed to connect to OpenAI API: {e}")
                                    return jsonify({"error": "Error connecting to the OpenAI API", "details": str(e)}), 500
                                except openai.error.RateLimitError as e:
                                    print(f"Rate limit exceeded: {e}")
                                    return jsonify({"error": "OpenAI API Rate limit exceeded", "details": str(e)}), 500
                                except openai.error.AuthenticationError as e:
                                    print(f"Authentication error: {e}")
                                    return jsonify({"error": "OpenAI API authentication error", "details": str(e)}), 500
                                                
                        os.remove(chunk_file.name)

                        # add the current chunk to transcriptions
                        transcriptions.append(transcript)
                        print(transcriptions)
                        print(transcriptions[0])
                        output = jsonify({'transcript': transcriptions[0]})

            else:
                # Run using the local API
                print("Using local API")
                # BASE_URL = "http://localhost:5007" # change this to the local server address
                options = {
                    'file_id': file_path, # this assumes that the local file path can be used as the file_id
                    'model': model,
                    'language': lang,
                    'outfmt': resp_format
                }

                # Convert options dictionary to json
                options_json = json.dumps(options)

                # Convert the file to a wav
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav_file:
                    output_wav_file_path = temp_wav_file.name
                try:
                    print("Attempting to convert to wav")
                    convert_to_wav(file_path, output_wav_file_path)
                    print(f"Converted, {output_wav_file_path}")
                except Exception as e:
                    return jsonify({"error": str(e)}), 500

                # Process the transcription
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{options['outfmt']}") as temp_output_file:
                    output_file_path = temp_output_file.name
                try:
                    print("Attempting to process transcription")
                    process_transcription(output_wav_file_path, options, output_file_path)
                    print(f"Processed, {output_file_path}")
                except Exception as e:
                    return jsonify({"error": str(e)}), 500

                # Return the transcription from output_file_path or the error returned by process_transcription
                try:
                    with open(output_file_path, 'r') as output_file:
                        transcriptions = output_file.read()
                        # Create a dictionary with the content of transcriptions as the value of a key called 'transcript'
                        output = jsonify({'transcript': transcriptions})
                except Exception as e:
                    return jsonify({"error": str(e)}), 500
                
        print("Returning transcriptions")
        print(output)
        return output, 200
 
@ts_flask_app.route('/process', methods=['POST'])
def process():
    output = {}
    processing = request.form.get('processing')
    processing_role = request.form.get('processing-role')
    tscript = request.form.get('transcript')
    # If the gpt_model has been configured in the form, update the value in the app.config
    if request.form.get('gpt_model'):
        ts_flask_app.config['model'] = request.form.get('gpt_model')

    print("processing_role: " + processing_role)
    # TODO: Next job is to get search the roles dict for the name rather than pass it directly
    # TODO: Also allow for a custom role description

    # Remember is an attempt to give GPT a memory of the previous summary to enable summaries of longer items
    logger.info(f"Requesting GPT processing of {tok_count(tscript)} tokens using role: {processing_role}")
    try:
        summary = gpt_proc(text=tscript, sys_role=processing_role)
    except Exception as e:
        print("Error summarizing:", e)
        logger.info(f"GPT processing error: {e}")
        return jsonify({"error": "Error summarizing", "details": str(e)}), 500
    
    print("-----------\nSummary:\n-----------")
    print(summary)
    tscript = summary[0]
    output['summary'] = summary
    return output
    
if __name__ == '__main__':
    ts_flask_app.run(debug=True, port=5000)


