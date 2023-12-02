"""
This script takes all the songs we have and use the lyric to create a list of 8 emotions we then use to replace the lyric itself.
This is needed to properly match user's emotions to the songs.
"""

from dotenv import load_dotenv

load_dotenv()

import json
from collections import defaultdict
from pathlib import Path

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["song"],
    template=Path("prompts/summary_with_emotions.prompt").read_text(),
)

llm = ChatOpenAI(temperature=0.7)

chain = LLMChain(llm=llm, prompt=prompt)

with open("data/lyrics_with_spotify_url.json", "r") as f:
    data = json.load(f)

new_data = defaultdict(list)

for movie, songs in data.items():
    for song in songs:
        print(f"{song['name']}")
        emotions = chain.run(song=song["text"])
        new_data[movie].append(
            {"name": song["name"], "text": emotions, "embed_url": song["embed_url"]}
        )


with open("data/emotions_with_spotify_url.json", "w") as f:
    json.dump(new_data, f)
