# Libraries
import json
import numpy as np
import requests
import openai
import os

from dotenv import load_dotenv
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


# Variables
# Keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# Summarization
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Input paths
parent_dir = Path(__file__).resolve().parents[0] / "context"
random_img_for_clip = parent_dir / "img.jpg"

chatbot_dir = "chatbot"
class_dir = "class"
training_chatbot_question_file = parent_dir / chatbot_dir / "question.json"
training_context_file = parent_dir / chatbot_dir / "context.json"
training_class_question_file = parent_dir / class_dir / "question.json"

# Output paths
training_chatbot_question_features = parent_dir / chatbot_dir / "question_feature.npy"
training_context_features = parent_dir / chatbot_dir / "context_feature.npy"
training_class_question_features = parent_dir / class_dir / "question_feature.npy"

# CLIP config
encoder = "openai/clip-vit-base-patch16"
clip_model = CLIPModel.from_pretrained(encoder)
clip_processor = CLIPProcessor.from_pretrained(encoder)


# Main function
def main():
    chatbot_questions = json.load(open(training_chatbot_question_file, "r"))
    if isinstance(chatbot_questions, dict):
        temp = chatbot_questions["questions"]
        chatbot_questions = [temp[idx]['question'] for idx in range(len(temp))]
        
    contexts = json.load(open(training_context_file, "r"))
    if isinstance(contexts, dict):
        temp = contexts["context_examples"]
        contexts = [temp[idx]['context'] for idx in range(len(temp))]

    class_questions = json.load(open(training_chatbot_question_file, "r"))
    if isinstance(class_questions, dict):
        temp = class_questions["questions"]
        class_questions = [temp[idx]['question'] for idx in range(len(temp))]

    summarized_context = [query({"inputs": context})[0]['summary_text'] for context in contexts]
    
    chatbot_question_clip_input = clip_processor(text=chatbot_questions, images=[Image.open(random_img_for_clip)]*len(chatbot_questions), return_tensors="pt", padding=True)
    chatbot_question_clip_output = clip_model(**chatbot_question_clip_input)

    context_clip_input = clip_processor(text=summarized_context, images=[Image.open(random_img_for_clip)]*len(contexts), return_tensors="pt", padding=True)
    context_clip_output = clip_model(**context_clip_input)

    class_question_clip_input = clip_processor(text=class_questions, images=[Image.open(random_img_for_clip)]*len(class_questions), return_tensors="pt", padding=True)
    class_question_clip_output = clip_model(**class_question_clip_input)

    np.save(training_chatbot_question_features, chatbot_question_clip_output.text_embeds.detach().numpy())
    np.save(training_context_features, context_clip_output.text_embeds.detach().numpy())
    np.save(training_class_question_features, class_question_clip_output.text_embeds.detach().numpy())
    
    
if __name__ == "__main__":
  main()