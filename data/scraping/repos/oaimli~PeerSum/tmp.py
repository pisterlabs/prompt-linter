"""
Use OpenAI embeddings to compute the relevance of summaries
"""

import os
from datasets import load_dataset
import random
import json
import spacy
import openai
from tqdm import tqdm
import tiktoken
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

openai.api_key = "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"
summary = " – Queen's \"Bohemian Rhapsody\" came out in 1975, but it just earned a new honor. The song passed 1.6 billion global streams, according to label Universal Music Group, making it the most streamed song out of the 20th century. The count includes streams on Spotify, Apple Music, and similar services, as well as views on YouTube, reports Billboard. \"Very happy that our music is still flowing to the max!\" says guitarist Brian May. \"Bohemian Rhapsody,\" included on the album A Night at the Opera, made its third appearance on the Billboard Hot 100 chart last month thanks to the release of a Freddie Mercury biopic. As the Guardian notes, you're sure to recognize other tracks rounding out the top five:"
perturbated_summary = " – Queen's \"Bohemian Rhapsody\" came out in 1975, but it just earned a new honor. The song passed 1.6 billion global streams, according to label Universal Music Group, making it the most streamed song out of the 20th century. The count includes streams on Spotify, Apple Music, and similar services, as well as views on YouTube, reports Billboard. \"Very happy that our music is still flowing to the max!\" says guitarist Brian May. \"Bohemian Rhapsody,\" included on the album A Night at the Opera, made its third appearance on the Billboard Hot 100 chart last month thanks to the release of a Freddie Mercury biopic. As the Guardian notes, you're sure to recognize other tracks rounding out the top five:"
source = "Queen's 'Bohemian Rhapsody' is the 20th Century's Most-Streamed Song, Says Label \n  \n The classic rock hit is aging exceptionally well. \n  \n Boosted by the recent success of the Freddie Mercury biopic Bohemian Rhapsody, Queen's iconic song of the same name is aging gracefully -- to say the least. On Monday (Dec. 10), Universal Music Group (UMG) announced the song is officially the most-streamed track from the 20th century, achieving more than 1.6 billion global streams. \n  \n That qualification also makes it the most-streamed classic rock song of all time, edging out Nirvana's \"Smells Like Teen Spirit,\" Guns N' Roses' \"Sweet Child O'Mine\" and \"November Rain\" and a-ha's \"Take on Me\" in both categories. The ranking considers all registered streams on global on-demand streaming services including Spotify, Apple Music, Deezer and others, as well as streams from official song or video streams on YouTube. \n  \n Queen guitarist and founding member Brian May said in a statement, \"So the River of Rock Music has metamorphosed into streams! Very happy that our music is still flowing to the max!\" \n  \n Added Lucian Grainge, chairman and CEO of UMG: \"'Bohemian Rhapsody' is one the greatest songs by one of the greatest bands in history. We are so proud to represent Queen and are thrilled to see the song still inspiring new fans around the world more than four decades after its release. My congratulations to Queen and [manager] Jim Beach on an incredible achievement that is a testament to the enduring brilliance of Queen.\" \n  \n Since its release in 1975 with the album A Night at the Opera, \"Bohemian Rhapsody\" has proven a resilient hit. On May 9, 1992, it peaked at No. 2 on the Hot 100 chart, more than 16 years after its original release, thanks to the success of the movie Wayne’s World and its head-banging sequence to the song. In 2004, \"Bohemian Rhapsody\" was inducted into the Grammy Hall of Fame. \n  \n Last month, following the release of Bohemian Rhapsody, the song re-entered the Hot 100 at No. 33, marking its third appearance on the chart. It also landed at No. 41 on the Streaming Songs chart with a 77 percent surge to 13.3 million U.S. streams on the chart dated Nov. 17. \n  \n According to a UMG spokesperson, the label and its teams around the world actively promoted discovery across streaming platforms to introduce the song to new fans around the world in the streaming era, more than 40 years after its original release. Bohemian Rhapsody has been named the most streamed song from the 20th century, overtaking Smells Like Teen Spirit by Nirvana. \n  \n The Queen hit, which reached No 1 in the UK in 1975 and then again in 1991 following Freddie Mercury’s death, has now been streamed 1.6bn times across services including YouTube and Spotify. \n  \n The song is currently as popular as it has been in a generation, following the release of the Queen biopic also entitled Bohemian Rhapsody, which has made almost $600m at the global box office since its release in November. The song re-entered the UK charts the same month and reached No 45, and for the week of 22 November, was the 11th most streamed song in the world on Spotify. \n  \n Queen's 50 UK singles – ranked! Read more \n  \n In a statement accompanying the news, Queen guitarist Brian May said: “So the river of rock music has metamorphosed into streams! Very happy that our music is still flowing to the max!” Lucian Grainge, CEO of Queen’s label Universal Music Group, said it was an “incredible achievement that is a testament to the enduring brilliance of Queen”. \n  \n The top five most streamed tracks of the 20th century are surprising, with major stars such as Michael Jackson and the Beatles absent. Nirvana’s 1991 grunge anthem Smells Like Teen Spirit is at No 2 with more than 1.5bn streams, followed by two songs by Guns N’ Roses: Sweet Child O’ Mine and November Rain. a-ha’s synthpop hit Take on Me rounds out the top five. Bohemian Rhapsody was also named the most-streamed classic rock song of all time. \n  \n Ed Sheeran’s 2017 single Shape of You has meanwhile become the first song ever to pass 2bn streams on Spotify. The British singer-songwriter said “thank you world” on Instagram, along with an announcement of the milestone. "

summary_embedding = []
perturbated_summary_embedding = []
source_documents_embedding = []
while True:
    try:
        summary_embedding = \
            openai.Embedding.create(input=summary, model="text-embedding-ada-002")["data"][0][
                "embedding"]
        perturbated_summary_embedding = \
        openai.Embedding.create(input=perturbated_summary, model="text-embedding-ada-002")["data"][0][
            "embedding"]
        source_documents_embedding = \
            openai.Embedding.create(input=source, model="text-embedding-ada-002")["data"][0][
                "embedding"]
    except:
        source_documents_embedding = []
        perturbated_summary_embedding = []
        summary_embedding = []
    if len(source_documents_embedding) == 1536 and len(perturbated_summary_embedding)==1536 and len(summary_embedding)==1536:
        break

print("relevance summary", cosine_similarity([summary_embedding], [source_documents_embedding])[0][0])
print("relevance perturbated summary", cosine_similarity([perturbated_summary_embedding], [source_documents_embedding])[0][0])