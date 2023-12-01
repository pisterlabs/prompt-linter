import psycopg2
from psycopg2.extras import DictCursor
from psycopg2.extras import execute_batch
import pandas as pd
from datetime import datetime
import cohere
import time
from tqdm.auto import tqdm
from secret import db_connection, get_cohere_api_key
import json
from openai import OpenAI

co = cohere.Client(get_cohere_api_key())

db_params = db_connection()

reliable_users = ["amitsegal", "kanarab", "bnetanyahu", "bengvir", "FakeReporter_official",
                  "PikudHaOref_all", "justiceil", "hadshothadover", "idf_telegram", "ynetalerts", "MOHreport"]


def get_top_n(embedding, n=5):

    emb_type = "cohere"
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor(cursor_factory=DictCursor)

    cur.execute(f"SELECT id, channel_title, channel_username, created_on, message, 1 - ({emb_type} <=> %s::vector) AS cosine_similarity FROM telegram_messages ORDER BY {emb_type} <=> %s::vector LIMIT {n};", (embedding, embedding))
    out = cur.fetchall()
    conn.commit()
    cur.close()
    conn.close()

    return pd.DataFrame([dict(x) for x in out])


def get_related_openai(text, candidates):
    
    # Your OpenAI API key
    client = OpenAI(api_key='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')

    candidate_text = "\n\n".join([f"{i}: {x}" for i, x in enumerate(candidates)])
    prompt = f"The following is a news item or factual claim in Hebrew: {text}\n\n The following numbered texts were retrieved via semantic search as related to the text: {candidate_text}.\n\nYou are to classify for each retrieved text whether it factually supports or negates the first text. Classify it as factually supporting if the retrieved text reports the same facts or significantly supports the first text. Return a JSON file with fields 'support', and 'negate', where each should contain a list of text id numbers. Retrieved texts that neither support or negate the text, or are likely unrelated, should be omitted. Do not include anything else in your response."
    
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ]

    response = client.chat.completions.create(
        messages=messages,
        model="gpt-4-1106-preview",
        temperature=0
    )
    out = response.choices[0].message.content
    out = out.replace("```json", "").replace('```', "")
    
    try:
        out = json.loads(out)
        return out
    except Exception as e:
        print(e)
        print(out)
        print("\n\n\n")


def embed_text(text):
    
        vec = co.embed(
        texts=[text],
        model='embed-multilingual-v3.0',
        input_type='classification'
        ).embeddings
    
        return vec[0]


def get_related_openai(text, candidates):

    # Your OpenAI API key
    client = OpenAI() # key should be passed as env variable in lambda

    candidate_text = "\n\n".join([f"{i}: {x}" for i, x in enumerate(candidates)])
    prompt = f"The following is a news item or factual claim in Hebrew: {text}\n\n The following numbered texts were retrieved via semantic search as related to the text: {candidate_text}.\n\nYou are to classify for each retrieved text whether it factually supports or negates the first text. Classify it as factually supporting if the retrieved text reports the same facts or significantly supports the first text. Return a JSON file with fields 'support', and 'negate', where each should contain a list of text id numbers. Retrieved texts that neither support or negate the text, or are likely unrelated, should be omitted. Do not include anything else in your response."

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    response = client.chat.completions.create(
        messages=messages,
        model="gpt-4-1106-preview",
        temperature=0
    )
    out = response.choices[0].message.content
    out = out.replace("```json", "").replace('```', "")
    # yaml_out = "\n".join([x.strip() for x in yaml_out.split("\n")])

    try:
        # out = yaml.safe_load(yaml_out) 
        out = json.loads(out)
        return out
    except Exception as e:
        print(e)
        print(out)
        print("\n\n\n")


def similarity_search(text, threshold=0.9, n=5):

    vec = embed_text(text)
    df = get_top_n(vec, n)
    df = df[df["cosine_similarity"] > threshold]
    df['reliable'] = df['channel_username'].isin(reliable_users)
    df['created_on'] = df['created_on'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))


    return df


def proc_text(text, n=30):
    vec = embed_text(text)
    df = get_top_n(vec, n)
    map_dict = get_related_openai(text, df["message"])
    df['category'] = 'other'

    for category, indices in map_dict.items():
        df.loc[df.index.isin(indices), 'category'] = category

    # check if channel_username is in reliable_users
    df['reliable'] = df['channel_username'].isin(reliable_users)
    df['created_on'] = df['created_on'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))

    # format response to user given whether reliable channels support or negate the text
    support = df[df['category'] == 'support']
    negate = df[df['category'] == 'negate']

    support_reliable = support[support['reliable'] == True]
    support_unreliable = support[support['reliable'] == False]
    negate_reliable = negate[negate['reliable'] == True]
    negate_unreliable = negate[negate['reliable'] == False]

    support_reliable = support_reliable.to_dict('records')
    support_unreliable = support_unreliable.to_dict('records')
    negate_reliable = negate_reliable.to_dict('records')
    negate_unreliable = negate_unreliable.to_dict('records')

    return {
        'support_reliable': support_reliable,
        'support_unreliable': support_unreliable,
        'negate_reliable': negate_reliable,
        'negate_unreliable': negate_unreliable
    }

def handler(event, context):
    print(event)

    if "text" in event:
        text = event["text"]

        if "task" not in event or event["task"] not in ["similarity_search", "proc_text"]:
            return {"statusCode": 400, "body": "Invalid task"}

        if "n" in event:
            n = event["n"]
        else:
            n = 30

        if "threshold" in event:
            threshold = event["threshold"]
        else:
            threshold = 0.9

        if event["task"] == "similarity_search":
            print("similarity_search")
            df = similarity_search(text, n=n, threshold=threshold)
            return {"statusCode": 200, "body": df.to_json()}

        elif event["task"] == "proc_text":
            print("proc_text")
            out = proc_text(text, n=n)
            return {"statusCode": 200, "body": out}
    else:
        return {"statusCode": 400, "body": "No text provided"}

