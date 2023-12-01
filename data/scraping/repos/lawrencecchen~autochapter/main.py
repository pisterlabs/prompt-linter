import glob
import json
import os
from typing import List

import annoy
import cohere
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# COHERE_API_KEY = "8rLIrm60Gf9mSmeIF2RHxv1TNcjXpHFDZ9XnqjD2"
# COHERE_API_KEY = "2eXv0jtdPHV3tpe6ak9w5NmHDd1cV1n4mwLu8pLL" # chris' key
COHERE_API_KEY = "TvRrbHjuUPfaw44fYipL3zt5bYiKNLLJoajZRDnQ" # chris' key
co = cohere.Client(COHERE_API_KEY)

@app.get("/")
async def root():
    return {"message": "Hello World"}

def video_url_to_name(video_url: str):
  return video_url.split("/")[-1].split(".")[0]

def get_all_summaries()->List[str]:
  summaries = []
  for filename in glob.glob(os.path.join(os.getcwd(),"tsummaries/*.json")):
    with open(os.path.join(os.getcwd(), "tsummaries", f"{filename}"), "r") as f:
      json_data = json.loads(f.read())
      video_name = video_url_to_name(filename)
      for item in json_data:
        summaries.append(item["summary"] + f"  __INTERNAL__:__timestamp__={item['start_time']}__video_name__={filename}")
  return summaries
  
summaries = get_all_summaries()
embeds = co.embed(texts=summaries,
                  model='large',
                  truncate='LEFT').embeddings
# print(len(embeds[0]))
# print(summaries)

search_index = annoy.AnnoyIndex(len(embeds[0]), 'angular')
# search_index = annoy.AnnoyIndex(embeds.shape[1], 'angular')

for i in range(len(embeds)):
    search_index.add_item(i, embeds[i])
search_index.build(10) # 10 trees
# similar_item_ids = search_index.get_nns_by_item(example_id,10,
#                                                 include_distances=True)


@app.get("/search")
async def search(q: str):
    # query_embed = co..batch_embed([q])
    query_embed = co.embed(texts=[q], model='large', truncate='LEFT').embeddings
    result = search_index.get_nns_by_vector(query_embed[0],10, include_distances=True)
    result_indices = result[0]
    nearest = [summaries[i] for i in result_indices]
    # nearest = summaries[result]
    # result = search_index.get_nns_by_vector(query_embed[0],1, include_distances=True)
    return {"result": nearest}

# @app.get("")

@app.get("/autochapter")
async def autochapter(youtube_url: str = ""):
    return {"message": "Hello World"}
