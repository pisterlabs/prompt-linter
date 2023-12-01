from fastapi.middleware.cors import CORSMiddleware
#import io
#from google.cloud import storage
from fastapi import FastAPI, UploadFile, File
from google.auth import default
from google.auth.transport import requests as grequests
import requests
import json
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import openai
#import the List class from typing module
from typing import List
from lime import lime_text
import numpy as np
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import llm_explainability_prompt

#import torch
#import torch.nn as nn
#from PIL import Image
#from torch.nn import functional as F
#from torchvision import models, transforms

app = FastAPI()
load_dotenv()  # take environment variables from .env.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

class Text(BaseModel):
    text: str

class Image(BaseModel):
    image: str #base 64 image string

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://blank-to-bard-frontend-dlkyfi4jza-uc.a.run.app"],  # Change this to the actual URL of your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to the auth middleman blank-to-bard API!"}

@app.get("/health")
def health_check():
    return {"status": "Healthy"}

@app.post("/transcribe/{language}")
async def transcribe_audio(audio: UploadFile = File(...), language: str = "en"):
    # Save temporary audio file
    with open("temp_audio.mp3", "wb") as buffer:
        buffer.write(await audio.read())

     # Transcribe the audio
    with open("temp_audio.mp3", "rb") as audio_file:
        result = openai.Audio.transcribe("whisper-1", audio_file, language=language)

    return {"transcription": result["text"]}

@app.post("/classifier/predict")
def predict(text: Text):
    print('incoming request: ', text)
    # Generate access token
    credentials, project = default()
    auth_request = grequests.Request()
    credentials.refresh(auth_request)
    access_token = credentials.token
    # Prepare the data in the required format
    data = {
        'instances': [{'text': str(text.text)}],
    }
    print('data: ', data)
    # Send request to Vertex AI
    project_id = '260219834114'
    endpoint_id = '5000042321450893312'
    response = requests.post(
        f'https://europe-west4-aiplatform.googleapis.com/v1/projects/{project_id}/locations/europe-west4/endpoints/{endpoint_id}:predict',
        headers={
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json',
        },
        data=json.dumps(data),
    )

    return response.json()

@app.post("/classifier/explain")
def explain_lime(text: Text):
    print(text.text)

    class_names = ['blank', 'bard']

    def prediction_probs(texts: List[str]):
        probabilities = []
        for text in texts:
            try:
                # Run the predict function and wait for it to complete
                result = predict(Text(text=text))
                # Extract the probability scores from the result
                scores = [prediction['softmax'] for prediction in result['predictions']]
                scores = [item for sublist in scores for item in sublist]
                probabilities.append(scores)
                print(f"Predicted probabilities: {probabilities}")
            except Exception as e:
                print(f"An error occurred while predicting: {str(e)}")
                # Handle the error appropriately, 
                # for example by appending a default value to the probabilities list
                probabilities.append([0, 0])

        return np.array(probabilities)

    explainer = lime_text.LimeTextExplainer(class_names=class_names)
    explanation = explainer.explain_instance(text.text, prediction_probs, num_samples=10)

    weightage = explanation.as_list()

    # Split the text into a list of words
    words = text.text.split()

    # sort the weightage list based on the order of words in the sentence
    weightage.sort(key=lambda x: words.index(x[0]) if x[0] in words else len(words))

    return {'weightage': weightage}



@app.post("/classifier/explain/llm")
def explain_llm(text: Text):
	llm = OpenAI(temperature=0.7, model_name="gpt-4")

	question = text.text

	prompt_template = llm_explainability_prompt.PROMPT_PREFIX + question + llm_explainability_prompt.PROMPT_SUFFIX
	prompt = PromptTemplate(template=prompt_template, input_variables=[])

	chain = LLMChain(prompt=prompt,llm=llm)

	pred = chain.predict()

	try:
		return json.loads(pred)
	except ValueError:
		return dict()


@app.post("/face_classifier/predict")
async def face_predict(base64image: Image):
    print('incoming request: ', base64image)
    # Generate access token
    credentials, project = default()
    auth_request = grequests.Request()
    credentials.refresh(auth_request)
    access_token = credentials.token
    # Prepare the data in the required format
    data = {
        'instances': [{'image': str(base64image.image)}],
    }
    # print('data: ', data)
    # Send request to Vertex AI
    project_id = '260219834114'
    endpoint_id = '1464927720197586944'
    response = requests.post(
        f'https://europe-west4-aiplatform.googleapis.com/v1/projects/{project_id}/locations/europe-west4/endpoints/{endpoint_id}:predict',
        headers={
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json',
        },
        data=json.dumps(data),
    )

    return response.json()

# async def face_predict(file: UploadFile = File(...)):

#     def load_weights_from_gcs(bucket_name, folder_name, file_name):
#         storage_client = storage.Client("blank-to-bard")
#         bucket = storage_client.bucket(bucket_name)
#         blob = bucket.blob(f"{folder_name}/{file_name}")
#         weights_bytes = blob.download_as_bytes()
#         buffer = io.BytesIO(weights_bytes)
#         model_state_dict = torch.load(buffer)
#         return model_state_dict
    
#     bucket_name = "blank-to-bard"
#     folder_name = "face_reco"
#     file_name = "weights.pt"

#     model_state_dict = load_weights_from_gcs(bucket_name, folder_name, file_name)

#     data_transforms = {
#         "api": transforms.Compose(
#             [
#                 transforms.Resize((224, 224)),
#                 transforms.ToTensor(),
#                 transforms.Normalize(
#                     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#                 ),
#             ]
#         ),
#     }
#     model = models.resnet50()
#     model.fc = nn.Sequential(
#         nn.Linear(2048, 128), nn.ReLU(inplace=True), nn.Linear(128, 2)
#     )
#     model.load_state_dict(model_state_dict)
#     request = Image.open(file.file).convert("RGB")
#     request_transformed = data_transforms["api"](request)
#     pred = model(request_transformed.unsqueeze(0))
#     pred_probs = F.softmax(pred, dim=1)
#     prob_negative, prob_positive = pred_probs.detach().numpy()[0]
#     return {"prediction": 1} if prob_positive > prob_negative else {"prediction": 0}
