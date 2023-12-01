#kidus berhanu
import os
import io
import cv2
import wave
import base64
import requests
import numpy as np
import sounddevice as sd
import traceback
from flask import Flask, request, jsonify, send_from_directory, make_response
from google.cloud import vision, language_v1
import openai
from google.cloud import speech_v1p1beta1 as speech
import boto3
import json
import tensorflow as tf
import torch
from sklearn.ensemble import RandomForestClassifier

s3 = boto3.client('s3')
lambda_client = boto3.client('lambda')

# Set environment variables
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/ubuntu/ARC-ANGEL/key.json'
openai.organization = "kidus is the king "
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize clients
vision_client = vision.ImageAnnotatorClient()
language_client = language_v1.LanguageServiceClient()

app = Flask(__name__)
app.debug = True

def handle_request(user_speech):
    text = transcribe_speech(user_speech)
    intent = detect_intent(text)

    if intent == 'objectDetection':
        image = get_image_from_user()
      while True:
    ret, frame = video_cap.read()
    if not ret:
        break

        result = vision_client.object_localization(image=image)
        return process_object_detection_result(result)
   elif intent == 'sentimentAnalysis':
        # Invoke Lambda function for sentiment analysis
        try:
            lambda_response = lambda_client.invoke(
                FunctionName='your-lambda-function-name',
                InvocationType='RequestResponse',
                Payload=json.dumps({'text': text})
            )
            sentiment_result = json.load(lambda_response['Payload'])
            return sentiment_result
        except Exception as e:
            return f"An error occurred while invoking Lambda: {e}"
    elif intent == 'generateText':
        gpt_response = openai.Completion.create(engine='text-davinci-003', prompt=text, max_tokens=100)
        return gpt_response['choices'][0]['text']
    else:
        return 'Sorry, I did not understand your request.'

def invoke_lambda_function(function_name, payload):
    try:
        response = lambda_client.invoke(
            FunctionName=function_name,
            InvocationType='RequestResponse',
            Payload=json.dumps(payload)
        )
        response_payload = json.load(response['Payload'])
        return response_payload
    except Exception as e:
        return f"An error occurred: {e}"


def transcribe_speech(audio_source):
    if audio_source == 'microphone':
        CHANNELS = 1
        RATE = 16000
        RECORD_SECONDS = 5
        WAVE_OUTPUT_FILENAME = "audio.wav"

        print("Recording audio...")
        audio_data = sd.rec(int(RATE * RECORD_SECONDS), samplerate=RATE, channels=CHANNELS)
        sd.wait()
        print("Finished recording.")

        with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wavefile:
            wavefile.setnchannels(CHANNELS)
            wavefile.setsampwidth(2)
            wavefile.setframerate(RATE)
            wavefile.writeframes(audio_data.tobytes())

        audio_file_path = WAVE_OUTPUT_FILENAME
    else:
        audio_file_path = audio_source

    with open(audio_file_path, 'rb') as audio_file:
      
        content = audio_file.read()

    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)

    transcript = ""
    for result in response.results:
        transcript += result.alternatives[0].transcript

    if not transcript:
        return 'Transcription failed.'

    return transcript
      




def detect_intent(text):
    lower_text = text.lower()
    object_detection_keywords = ['object', 'detect', 'find']
    sentiment_analysis_keywords = ['sentiment', 'feel', 'emotion']
    generate_text_keywords = ['generate', 'text', 'write', 'create']

    for keyword in object_detection_keywords:
        if keyword in lower_text:
            return 'objectDetection'
    for keyword in sentiment_analysis_keywords:
        if keyword in lower_text:
            return 'sentimentAnalysis'
    for keyword in generate_text_keywords:
        if keyword in lower_text:
            return 'generateText'
    return 'unknown'

def get_image_from_user():
    video_cap = cv2.VideoCapture(0)
    frames = []
    while True:
        ret, frame = video_cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (300, 300))
        frames.append(frame)
    video_cap.release()

    output = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output.avi', output, 20.0, (300, 300))
    for frame in frames:
        out.write(frame)
    out.release()

    with open('output.avi', 'rb') as f:
        content = f.read()
    return vision.Image(content=content)

import time

request_count = 0
last_reset_time = time.time()

def rewrite_self_with_openai():
    global request_count, last_reset_time
    
    try:
        # Check if 3 hours have passed since the last reset
        if time.time() - last_reset_time >= 3 * 60 * 60:
            request_count = 0
            last_reset_time = time.time()

        # Check if the request limit has been reached
        if request_count >= 50:
            return "Request limit reached. Try again later."

        with open(__file__, 'r') as f:
            original_code = f.read()

        while True:
            # Increment the request count
            request_count += 1

            gpt_response = openai.Completion.create(
                engine='text-davinci-003',
                prompt=f"Rewrite the following Python code:\n{original_code}",
                max_tokens=2000
            )
            rewritten_code = gpt_response['choices'][0]['text'].strip()

            # Test the rewritten code by executing it
            try:
                exec(rewritten_code)
                # If the code executes successfully, break the loop
                break
            except Exception as e:
                # If the code fails, send the error message to OpenAI for debugging
                error_message = str(e)
                gpt_response = openai.Completion.create(
                    engine='text-davinci-003',
                    prompt=f"The following Python code produced an error:\n{rewritten_code}\nError: {error_message}\nHow can this be fixed?",
                    max_tokens=2000
                )
                rewritten_code = gpt_response['choices'][0]['text'].strip()

        # If the code is successfully rewritten and tested, save it
        with open(__file__, 'w') as f:
            f.write(rewritten_code)

        return "Code rewritten and tested successfully."
    except Exception as e:
        return f"An error occurred: {e}"


from contextlib import redirect_stdout
import sys

def execute_code(code):
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    try:
        exec(code)
    except Exception as e:
        return str(e)

    output = new_stdout.getvalue()
    sys.stdout = old_stdout

    return output.strip()

from sklearn.ensemble import IsolationForest

@app.route('/anomaly-detection', methods=['POST'])
def anomaly_detection():
    data = request.get_json()['data']
    model = IsolationForest().fit(data)
    anomalies = model.predict(data)
    return jsonify({'anomalies': anomalies.tolist()})





def process_object_detection_result(response):
    detected_objects = response.localized_object_annotations
    message = "Detected objects are:\n"
    for i, obj in enumerate(detected_objects):
        message += f"{i+1}. {obj.name} with confidence: {obj.score}\n"
    return message

def process_sentiment_analysis_result(response):
    emotions = [
        {'range': [0.7, 1], 'label': 'very positive'},
        {'range': [0.25, 0.7], 'label': 'positive'},
        {'range': [-0.25, 0.25], 'label': 'neutral'},
        {'range': [-0.7, -0.25], 'label': 'negative'},
        {'range': [-1, -0.7], 'label': 'very negative'},
    ]
    sentiment = response.document_sentiment.score
    for emotion in emotions:
        if emotion['range'][0] <= sentiment <= emotion['range'][1]:
            return f"The sentiment of the text is {emotion['label']}."

@app.route('/api')
def api():
    return 'API endpoint'

@app.route('/object-detection', methods=['POST'])
def object_detection():
    data = request.get_json()
    frame_data = data.get('frameData')
    img_str = frame_data.split(',')[1]
    img_bytes = base64.b64decode(img_str)
    img_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = vision.Image(content=img_bytes)
    response = vision_client.object_localization(image=image)
    objects = response.localized_object_annotations
    return jsonify({'result': [obj.name for obj in objects]})

@app.route('/sentiment-analysis', methods=['POST'])
def perform_sentiment_analysis():
    try:
        text = request.form['text']
        document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
        result = language_client.analyze_sentiment(request={'document': document})
        response = process_sentiment_analysis_result(result)
        return response
    except Exception as e:
        print(traceback.format_exc())
        return 'An error occurred.', 500

@app.route('/rewrite-self', methods=['GET'])
def rewrite_self_route():
    result = rewrite_self_with_openai()
    return result, 200 if "successfully" in result else 500


@app.route('/execute-code', methods=['POST'])
def execute_code_route():
    try:
        code = request.form['code']
        output = execute_code(code)
        return jsonify({'output': output}), 200
    except Exception as e:
        print(traceback.format_exc())
        return 'An error occurred.', 500

@app.route('/self-heal', methods=['GET'])
def self_heal():
    try:
        # Your recovery logic here
        return "System healed", 200
    except Exception as e:
        return f"An error occurred: {e}", 500








@app.route('/text-generation', methods=['POST'])
def generate_text():
    try:
        text = request.form['prompt']
        gpt_response = openai.Completion.create(engine='text-davinci-003', prompt=text, max_tokens=100)
        generated_text = gpt_response['choices'][0]['text']
        return generated_text
    except Exception as e:
        print(traceback.format_exc())
        return 'An error occurred.', 500

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/object-detection')
def object_detection_page():
    return send_from_directory('static', 'object_detection.html')

@app.route('/sentiment-analysis')
def sentiment_analysis_page():
    return send_from_directory('static', 'sentiment_analysis.html')

from sklearn.ensemble import IsolationForest
import json

@app.route('/realtime-analytics', methods=['POST'])
def realtime_analytics():
    try:
        # Assume the incoming data is in JSON format
        data = json.loads(request.data)
        feature_matrix = data.get('feature_matrix', [])

        if not feature_matrix:
            return jsonify({'error': 'Invalid data'}), 400

        # Fit the Isolation Forest model
        model = IsolationForest(n_estimators=100, contamination=0.1)
        model.fit(feature_matrix)

        # Predict anomalies in the data
        anomalies = model.predict(feature_matrix)

        # Count the number of anomalies and normal points
        anomaly_count = sum(anomalies == -1)
        normal_count = sum(anomalies == 1)

        # Generate a summary
        summary = {
            'total_data_points': len(feature_matrix),
            'anomaly_count': anomaly_count,
            'normal_count': normal_count,
            'anomaly_indices': [i for i, x in enumerate(anomalies) if x == -1]
        }

        return jsonify({'summary': summary}), 200

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': 'An error occurred.'}), 500

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import json

# Sample data: for the purpose t=of this repositiry
sample_data = [
    {"id": 1, "description": "apple orange banana"},
    {"id": 2, "description": "apple grape"},
    {"id": 3, "description": "mango banana"},
    {"id": 4, "description": "orange grape"}
]

vectorizer = CountVectorizer().fit_transform([item['description'] for item in sample_data])
vectors = vectorizer.toarray()

@app.route('/realtime-recommendation', methods=['POST'])
def realtime_recommendation():
    try:
        # Assume the incoming data is in JSON format
        data = json.loads(request.data)
        user_input = data.get('user_input', "")

        if not user_input:
            return jsonify({'error': 'Invalid input'}), 400

        # Vectorize the user input
        user_vector = vectorizer.transform([user_input]).toarray()

        # Compute cosine similarity
        cosine_sim = cosine_similarity(user_vector, vectors)

        # Get the index of the most similar item
        recommended_index = cosine_sim.argsort()[0][-2]

        # Fetch the recommended item
        recommended_item = sample_data[recommended_index]

        return jsonify({'recommended_item': recommended_item}), 200

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': 'An error occurred.'}), 500




if __name__ == '__main__':
    app.run(host='0.0.0.0')
    



