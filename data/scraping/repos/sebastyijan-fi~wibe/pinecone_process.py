import os
import json
import openai
import pinecone
import numpy as np
from dotenv import load_dotenv
import uuid

def process_text_and_store(text):
    load_dotenv()

    openai.api_key = os.getenv("OKEY")
    pinecone_api_key = os.getenv("PKEY")

    if pinecone_api_key is None:
        raise ValueError("Pinecone API key not found in environment variables.")

    pinecone.init(api_key=pinecone_api_key, environment='us-east1-gcp')

    index_name = "wibe-moods"

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, metric='cosine', dimension=28, shards=1)

    index = pinecone.Index(index_name)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that outputs mood scores for text in JSON format. The moods you should consider are: Happiness, Contentment, Pleasure, Excitement, Hope, Optimism, Comfort, Reliability, Astonishment, Amazement, Wonder, Sorrow, Disappointment, Unhappiness, Anxiety, Worry, Unease, Reflective Sadness, Contemplative Longing, Calm, Peace, Tranquility, Curiosity, Fascination, Engagement, Affection, Warmth, and Fondness.",
            },
            {
                "role": "user",
                "content": f"reply only in JSON format, with the moods as key values. Use all moods and values range from 0-1!!! Exctract from here: '{text}'?",
            },
        ],
    )

    mood_scores_str = response.choices[0].message["content"]  # type: ignore
    mood_scores_dict = json.loads(mood_scores_str)
    mood_scores_list = [float(score) for score in mood_scores_dict.values()]

    vector_id = str(uuid.uuid4())

    # add mood_scores_dict as metadata
    metadata = {vector_id: mood_scores_dict}

    print("Formatted command for Pinecone:")
    print(f"index.upsert([(vector_id, mood_scores_list)], metadata={metadata})")

    index.upsert([(vector_id, mood_scores_list)], metadata=metadata)
    fetched_vector = index.fetch(ids=[vector_id])
    print(fetched_vector)


    return vector_id, mood_scores_list

