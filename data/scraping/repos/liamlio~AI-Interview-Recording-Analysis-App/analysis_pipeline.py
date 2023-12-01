import os
import time
import cohere
import numpy as np
import pandas as pd
from ast import literal_eval
from vindent_utils.transcribe import transcribe
from transformers import pipeline
from nltk.tokenize import sent_tokenize
from scipy.spatial.distance import cosine

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
COHERE_API_KEY = ""
CO = cohere.Client(COHERE_API_KEY)

CLASSIFIER = pipeline("zero-shot-classification",
                      model="valhalla/distilbart-mnli-12-1")

BUILT_IN_MODELS = ["Entrepreneurship", "Confidence", "Growth Mindset"]

BUILT_IN_CUSTOM_MODELS = {
    "Craftpersonship": "Find meaning in what we do through crafting excellence.",
    "Playfulness": "Great ideas come from health and happiness.",
    "Grit": "Perseverance driven by determination and passion.",
    "Empathy": "Innovation starts with understanding.",
    "Zest": "What sets you apart makes us unique.",
    "Courage": "Dare often and greatly."
}

def create_embeds(texts, user_id="test"):
    texts = sent_tokenize(texts)
    embeds = CO.embed(texts, model="large", truncate="right")
    return texts, embeds

def score_BART_text(text, model_name):
    pred = CLASSIFIER(text, [model_name, "not " + model_name ])
    if pred["scores"][0] > pred["scores"][1] and pred["labels"][0] == model_name:
        return 1
    elif pred["scores"][1] > pred["scores"][0] and pred["labels"][1] == model_name:
        return 1
    else:
        return 0

def score_CUSTOM_text(text_embed, custom_model_embeds):
    mean_score = np.mean([cosine(e, text_embed) for e in custom_model_embeds.embeddings])
    if len(custom_model_embeds.embeddings) < 5:
        threshold = 1.0 - len(custom_model_embeds.embeddings) / 10
    else:
        threshold = 0.5
    if mean_score > threshold:
        return 1
    else:
        return 0

def score_custom_models(texts_df, custom_models=BUILT_IN_CUSTOM_MODELS):
    for model_name in custom_models:
        _, embeds_custom = create_embeds(custom_models[model_name], user_id="test")
        texts_df[model_name] = texts_df["embeddings"].apply(lambda x: score_CUSTOM_text(x, embeds_custom))
        time.sleep(10) #Throttle since we're using the free tier for cohere
    return texts_df

def score_new_custom_model(texts_df, custom_model):
    texts_df["embeddings"] = texts_df["embeddings"].apply(literal_eval)
    for model_name in custom_model:
        _, embeds_custom = create_embeds(custom_model[model_name], user_id="test")
        texts_df[model_name] = texts_df["embeddings"].apply(lambda x: score_CUSTOM_text(x, embeds_custom))
        time.sleep(10) #Throttle since we're using the free tier for cohere
    return texts_df

def audio_pipeline(audio_file, custom_models=BUILT_IN_CUSTOM_MODELS):
    paragraphs = transcribe(audio_file)
    texts = " ".join([p["text"] for p in paragraphs])
    texts, embeds = create_embeds(texts)
    texts_df = pd.DataFrame({"text":texts, "embeddings": embeds.embeddings})
    texts_df = score_custom_models(texts_df, custom_models=custom_models)
    for model in BUILT_IN_MODELS:
         texts_df[model] = score_BART_text(texts_df["text"].values, model)
    return texts_df