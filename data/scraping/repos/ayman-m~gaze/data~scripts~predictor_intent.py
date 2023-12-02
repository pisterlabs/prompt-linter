import torch
import pickle
import ast
import os
import pandas as pd
import numpy as np
import openai

from pathlib import Path
from dotenv import load_dotenv
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity

# Load environment variables from .env file if it exists
env_path = Path('.') / '.env'
if env_path.exists():
    load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Get user inputs
model_choice = input("Choose a model (BERT or ADA): ")
prompt = input("Enter a prompt to test: ")

# Check if the chosen model is valid
if model_choice not in ["BERT", "ADA"]:
    print("Invalid model choice. Please choose a valid model.")
    exit()

if model_choice == "BERT":
    # Load the trained model
    model = DistilBertForSequenceClassification.from_pretrained("models/bert")
    model.eval()
    # Load the tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    # Load the label encoder
    with open('models/bert/label_encoder.pkl', 'rb') as file:
        label_encoder = pickle.load(file)

    # Prepare the text data into the correct format for the model
    inputs = tokenizer(prompt, truncation=True, padding=True, max_length=512, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        logits = model(**inputs).logits

    # Get the predicted class indices and their corresponding probabilities
    probabilities = torch.softmax(logits, dim=-1)
    predicted_class_indices = torch.argsort(probabilities, descending=True)[0]
    predicted_class_probabilities = probabilities[0, predicted_class_indices]

    # Convert the predicted class indices back into their corresponding labels
    predicted_labels = label_encoder.inverse_transform(predicted_class_indices)

    # Print the predicted labels and their probabilities
    for label, probability in zip(predicted_labels, predicted_class_probabilities):
        print("Intent Name:", label)
        print("Probability:", probability.item())
        print()

if model_choice == "ADA":
    df = pd.read_csv("data/processed/embedding/intents/ada-basic-intent-embedding.csv", usecols=['embedding', 'name'])
    question_vector = get_embedding(prompt, engine="text-embedding-ada-002")
    print (question_vector)
    df["similarities"] = df['embedding'].apply(lambda x: cosine_similarity(np.array(ast.literal_eval(x)),
                                                                           question_vector))
    similar_rows = df.sort_values(by='similarities', ascending=False).head(3)

    # Print the similar intent names and their similarities
    for index, row in similar_rows.iterrows():
        print("Intent Name:", row['name'])
        print("Similarity:", row['similarities'])
        print()

