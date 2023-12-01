import torch
import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm
import time

from sentence_transformers import SentenceTransformer
import cohere

from elasticsearch_index_episodes import elasticsearch_index_chapters

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

COHERE = True

## Load sentence transformer
model_name = "all-mpnet-base-v2"
model = SentenceTransformer(model_name)
model = model.to(device)

## Load cohere client
cohere_api_key = 'c8ES1KWN9nd8uObqxiBvBWEQ450asuAkoF61EYCg'
co = cohere.Client(cohere_api_key)

## Load chapter data
selected_episodes = os.listdir(os.path.join("data", "transcriptions"))
selected_episodes = [episode for episode in selected_episodes if not episode.startswith(".")]

## Load episod data
df = pd.read_csv('data/news_episodes.csv')

## Initialize data structures
embedded_summaries = {}
episode_ids = []
chapter_ids = []
chapter_gists = []
chapter_summaries = []
audio_urls = []
starts = []
ends = []
episode_titles = []
episode_pub_dates = []
podcast_names = []

counter = 0
for episode in tqdm(selected_episodes):
    episode_id = episode[:-5]
    
    ## Load Json file
    with open(os.path.join("data", "transcriptions", episode), "r") as f:
        data = json.load(f)
    
    if COHERE:
        if counter > 85:
            time.sleep(50)
            counter = 0
    
    ## Get Chapters 
    chapters = data["chapters"]
    audio_url = data["audio_url"]

    ## Get episode data
    episode_data = df[df['episode_audio_link'] == audio_url]
    episode_title = episode_data['episode_title'].values[0]
    episode_pub_date = episode_data['episode_pub_date'].values[0]
    podcast_name = episode_data['podcast_name'].values[0]
    
    print("Episode:", episode)
    print("# chapters:", len(chapters))
    print("Chapters:")

    counter += len(chapters)

    for i, chapter in enumerate(chapters):
        chapter_id = episode_id + "_" + str(i)
        gist = chapter["gist"]
        summary = chapter["summary"]
        start = chapter["start"]
        end = chapter["end"]

        ## Store the information retrieved
        chapter_ids.append(chapter_id)
        chapter_gists.append(gist)
        chapter_summaries.append(summary)
        episode_ids.append(episode_id)
        audio_urls.append(audio_url)
        starts.append(start)
        ends.append(end)
        episode_titles.append(episode_title)
        episode_pub_dates.append(episode_pub_date)
        podcast_names.append(podcast_name)

        
        ## Embed the summary
        if COHERE:
            response = co.embed(texts=[summary], model="small")
            embedded_summary = response.embeddings[0]
            embedded_summary = np.array(embedded_summary)
        else:
            embedded_summary = model.encode(summary)

        ## Store the embedded summary
        embedded_summaries[chapter_id] = embedded_summary
        

        print("\t", i+1, "-", gist)
        print("\t  ", "-", summary)


df = pd.DataFrame()
df["episode_id"] = episode_ids
df["chapter_id"] = chapter_ids
df["chapter_gist"] = chapter_gists
df["chapter_summary"] = chapter_summaries
df["audio_url"] = audio_urls
df["start"] = starts
df["end"] = ends
df["episode_title"] = episode_titles
df["episode_pub_date"] = episode_pub_dates
df["podcast_name"] = podcast_names

## Save the dataframe
df.to_csv("data/df_chapters.csv", index=False)

dense_dim = len(embedded_summary)
elasticsearch_index_chapters(df, embedded_summaries, dense_dim)
