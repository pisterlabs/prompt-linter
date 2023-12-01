import axios
from google.cloud import vision
from google.cloud import language_v1
import assemblyai
import openai
import cv2
import numpy as np
import io
import os
from flask import Flask
import requests

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/ubuntu/ARC-ANGEL/key.json'


vision_client = vision.ImageAnnotatorClient()
language_client = language_v1.LanguageServiceClient()
openai.organization = "if you are reading this hire me  lol"
openai.api_key = os.getenv("OPENAI_API_KEY")


def handle_request(user_speech):
   
text = transcribe_speech(user_speech)

    intent = detect_intent(text)

    if intent == 'objectDetection':
   
        image = get_image_from_user()
        result = vision_client.object_localization(image=image)
        return process_object_detection_result(result)
    elif intent == 'sentimentAnalysis':
      
        document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
        result = language_client.analyze_sentiment(request={'document': document})
        return process_sentiment_analysis_result(result)
    elif intent == 'generateText':
     
        gpt_response = openai.Completion.create(engine='text-davinci-003', prompt=text, max_tokens=200)
        return gpt_response['choices'][0]['text']
    else:
       
        return 'Sorry, I did not understand your request.'
def transcribe_speech(user_speech):
    response = requests.post('https://api.assemblyai.com/v2/transcript', json={'audio_url': user_speech},
                             headers={'authorization': os.environ['ASSEMBLY_AI_API_KEY']})
    response_json = response.json()
    text = response_json.get('text', 'Transcription not available')  # Use a fallback value if 'text' key is not found
    return text



def detect_intent(text):
   
    lower_text = text.lower()
  
    object_detection_keywords = ['detect', 'object', 'identify', 'recognition']
    sentiment_analysis_keywords = ['sentiment', 'feeling', 'emotion', 'mood']
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
    video_cap = cv2.VideoCapture(0)  # 0 for default camera
    frames = []
    while True:
      
        ret, frame = video_cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (300, 300))

        # Add frame to list of frames
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

app = Flask(__name__)
app.debug = True


import traceback

@app.route('/api')
def api():
    dataset_path = "/home/ubuntu/ARC-ANGEL/LibriSpeech"  # Path to the LibriSpeech folder
    try:
        transcriptions = []
        for root, dirs, files in os.walk(dataset_path):
            for file_name in files:
                if file_name.endswith(".flac"):
                    file_path = os.path.join(root, file_name)
                    transcription = transcribe_speech(file_path)  # Process the speech file
                    transcriptions.append(transcription)

      

        return "Transcriptions processed successfully."
    except Exception as e:
        print(traceback.format_exc())  # Print the exception traceback
     return 'An error occurred.', 500
if __name__ == '__main__':
    app.run(host='0.0.0.0')
