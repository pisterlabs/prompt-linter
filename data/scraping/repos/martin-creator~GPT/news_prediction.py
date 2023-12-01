import os
import openai
import pandas as pd
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity
from sklearn.metrics import precision_score


# Predicting News Category Using Embedding


def init_api():
    ''' Load API key from .env file'''
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value

    openai.api_key = os.environ["API_KEY"]
    openai.organization = os.environ["ORG_ID"]


init_api()

categories = ['POLITICS', 'WELLNESS', 'ENTERTAINMENT', 'TRAVEL', 'STYLE & BEAUTY', 'PARENTING', 'HEALTHY LIVING',
              'QUEER VOICES', 'FOOD & DRINK', 'BUSINESS', 'COMEDY', 'SPORTS', 'BLACK VOICES', 'HOME & LIVING', 'PARENTS']


# Define a function to classify sentence
def classify_sentence(sentence):
    # Get the embedding for the sentence
    sentence_embedding = get_embedding(
        sentence, engine="text-embedding-ada-002")

    # Calculate the similarity score between the sentence and each category
    similarity_scores = {}
    for category in categories:
        category_embedding = get_embedding(
            category, engine="text-embedding-ada-002")
        similarity_scores[category] = cosine_similarity(
            sentence_embedding, category_embedding)

    # Return the category with the highest similarity score
    return max(similarity_scores, key=similarity_scores.get)


# Classify a sentence
sentences = [
    "1 dead and 3 injured in El Paso, Texas, mall shooting",
    "Director Owen Kline Calls Funny Pages His ‘Self-Critical’ Debut",
    "15 spring break ideas for families that want to get away",
    "The US is preparing to send more troops to the Middle East",
    "Bruce Willis' 'condition has progressed' to frontotemporal dementia, his family\
says",
    "Get an inside look at Universal’s new Super Nintendo World",
    "Barcelona 2-2 Manchester United: Marcus Rashford shines but Raphinha salvages d\
raw for hosts",
    "Chicago bulls win the NBA championship", "The new iPhone 12 is now available",
    "Scientists discover a new dinosaur species",
    "The new coronavirus vaccine is now available",
    "The new Star Wars movie is now available",
    "Amazon stock hits a new record high",
]


for sentence in sentences:
    print("{:50} category is {}".format(sentence, classify_sentence(sentence)))
    print()

def evaluate_precision(categories):
    # Load the dataset
    df = pd.read_json("data/News_Category_Dataset_v3.json", lines=True).head(20)
    y_true = []
    y_pred = []

    # classify each sentence
    for _, row in df.iterrows():
        true_catergory = row["category"]
        predicted_category = classify_sentence(row["headline"])


        # Append the true and predicted categories to the lists
        y_true.append(true_catergory)
        y_pred.append(predicted_category)

    
    # Calculate the precision score
    return precision_score(y_true, y_pred, average='micro', labels=categories)


precision_evaluated = evaluate_precision(categories)
print("------------------------------ Evaluation ------------------------------")
print("Precision evaluated: {:.2f}".format(precision_evaluated))
