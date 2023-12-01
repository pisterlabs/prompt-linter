# Import libraries
import numpy as np
import pandas as pd
import os
import pickle
import re
import emoji
import string
import openai

# import spacy
# from spacy.lang.en.stop_words import STOP_WORDS

from google_play_scraper import Sort, reviews, reviews_all, app
from datetime import datetime, timedelta
from pymongo import MongoClient

# Create a connection to MongoDB
client = MongoClient(
    os.environ["MONGODB_URL"],
    serverSelectionTimeoutMS=300000
)
db = client["vidio"]
collection = db["google_play_store_reviews"]
collection2 = db["current_timestamp"]

# Load the data from MongoDB
df = pd.DataFrame(list(collection.find()))
df = df.drop("_id", axis=1)
df = df.sort_values("at", ascending=False)

# Collect 5000 new reviews
result = reviews(
    "com.vidio.android",
    lang="id",
    country="id",
    sort=Sort.NEWEST,
    count=5000
)
new_reviews = pd.DataFrame(result[0])
new_reviews = new_reviews.fillna("empty")
new_reviews = new_reviews.rename(columns={"content": "content_original"})

# Filter the scraped reviews to exclude any that were previously collected
common = new_reviews.merge(df, on=["reviewId", "userName"])
new_reviews_sliced = new_reviews[(~new_reviews.reviewId.isin(common.reviewId)) & (~new_reviews.userName.isin(common.userName))]

# Translate all reviews from Indonesian to English
openai.api_key = os.environ["OPENAI_API_KEY"]
neg_new_reviews_sliced = new_reviews_sliced[new_reviews_sliced["score"] <= 3]

def translate_to_english(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f'Please translate this Indonesian text "{text}" to english in the format [EN: translation], but if there is no English translation, return [EN: Cannot be translated]. Please make sure write in the format that I requested only.'
            }
        ]
    )
    return response["choices"][0]["message"]["content"]

english = []
for i in neg_new_reviews_sliced["content_original"]:
    translated_text = "[EN: Cannot be translated]"
    for j in range(5):
        try:
            translated_text = translate_to_english(i)
            break
        except:
            pass
    english.append(translated_text)

def find_invalid_indices(english):
    invalid_indices = []
    for i, text in enumerate(english):
        if not re.match(r'^\[EN: [^\[\]]+\]$', text):
            invalid_indices.append(i)
    return invalid_indices

invalid_indices = find_invalid_indices(english)

if len(invalid_indices) > 0:
    english_revision = []
    for i in [list(neg_new_reviews_sliced["content_original"])[i] for i in invalid_indices]:
        translated_text = "[EN: Cannot be translated]"
        for j in range(5):
            try:
                while True:
                    translated_text = translate_to_english(i)
                    if re.match(r'^\[EN: [^\[\]]+\]$', translated_text):
                        break
                break
            except:
                pass
        english_revision.append(translated_text)

    for i, j in zip(invalid_indices, english_revision):
        english[i] = j

neg_new_reviews_sliced["content_english"] = english

# Apply topic modeling
def assign_topic(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f'Please assign one of the topics (Advertisement, Watching Experience, Package, Technical, Network, Others) to this text "{text}" in the format [Topic: assigned topic]. Please make sure write in the format that I requested only.'
            }
        ]
    )
    return response["choices"][0]["message"]["content"]

topics = []
for i in neg_new_reviews_sliced["content_original"]:
    labeled_topic = "[Topic: Others]"
    for j in range(5):
        try:
            labeled_topic = assign_topic(i)
            break
        except:
            pass
    topics.append(labeled_topic)

cleaned_topics = [i for i in topics]

for idx, val in enumerate(cleaned_topics):
    if "Advertisement" in val:
        cleaned_topics[idx] = "Advertisement"
    elif "Watching Experience" in val:
        cleaned_topics[idx] = "Watching Experience"
    elif "Package" in val:
        cleaned_topics[idx] = "Package"
    elif "Technical" in val:
        cleaned_topics[idx] = "Technical"
    elif "Network" in val:
        cleaned_topics[idx] = "Network"
    elif "Others" in val:
        cleaned_topics[idx] = "Others"

neg_new_reviews_sliced["topic"] = cleaned_topics

# Merge neg_new_reviews_sliced to new_reviews_sliced
new_reviews_sliced_merged = pd.merge(new_reviews_sliced, neg_new_reviews_sliced[["topic"]], left_index=True, right_index=True, how="outer")
new_reviews_sliced_merged = pd.merge(new_reviews_sliced_merged, neg_new_reviews_sliced[["content_english"]], left_index=True, right_index=True, how="outer")
new_reviews_sliced_merged = new_reviews_sliced_merged[["reviewId", "userName", "userImage", "content_original", "content_english", "score", "thumbsUpCount", "reviewCreatedVersion", "at", "replyContent", "repliedAt", "topic"]]
new_reviews_sliced_merged = new_reviews_sliced_merged.fillna("empty")

# Update MongoDB with any new reviews that were not previously scraped
if len(new_reviews_sliced_merged) > 0:
    new_reviews_sliced_merged_dict = new_reviews_sliced_merged.to_dict("records")

    batch_size = 1_000
    num_records = len(new_reviews_sliced_merged_dict)
    num_batches = num_records // batch_size

    if num_records % batch_size != 0:
        num_batches += 1

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, num_records)
        batch = new_reviews_sliced_merged_dict[start_idx:end_idx]

        if batch:
            collection.insert_many(batch)

# Insert the current timestamp to MongoDB
current_datetime = datetime.now()
updated_datetime = current_datetime + timedelta(hours=7)
current_timestamp = updated_datetime.strftime("%A, %B %d %Y at %H:%M:%S")
collection2.replace_one({}, {"timestamp": current_timestamp}, upsert=True)