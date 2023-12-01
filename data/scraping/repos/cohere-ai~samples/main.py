import os
import uuid
import json
import urllib
import datetime
import requests
from datetime import datetime
from typing import List, Tuple, Dict

import cohere
from fastapi import FastAPI
from pydantic import BaseModel
from timeit import default_timer
from fastapi.staticfiles import StaticFiles

LOG_LOCATION = "re-rank-request-log.jsonl"
FEEDBACK_LOCATION = "feedback-log.jsonl"
api_app = FastAPI(title="api app")

# Replace the 'x' with your Cohere API key
os.environ["COHERE_KEY"] = "x"
# Initialize a cohere client for making calls to rerank
co = cohere.Client(os.environ["COHERE_KEY"])

class SearchInput(BaseModel):
    session_id: str
    query: str
    source: str


class ReRankInput(BaseModel):
    session_id: str
    query: str
    passages: List[str]


class FeedbackInput(BaseModel):
    session_id: str
    feedback: str
    query_id: str


def search_wikipedia(query):
    """
    The wikipedia api only returns proper search results alone (using the list operator).
    DO NOT use the generator operator, because the results would not be ranked correctly.
    A second call gets all the page infos we need (leading passage, url, and image).
    """
    wiki_search_string = "https://en.wikipedia.org/w/api.php?action=query&format=json&list=search&srlimit=20&srsearch="
    wiki_search_string += urllib.parse.quote_plus(query)

    initial_req = requests.get(wiki_search_string).json()

    if "query" not in initial_req:
        return [],[]
    
    # this is the correct wikipedia ranking (same as the wikipedia.org website)
    page_ids = []
    for page in initial_req["query"]["search"]:
        page_ids.append(str(page["pageid"]))

    if len(page_ids) == 0:
        return [],[]
    
    # Get extra infos for the returned page ids
    wiki_data_string = "https://en.wikipedia.org/w/api.php?action=query&format=json&prop=info%7Cextracts%7Cpageimages&formatversion=2&inprop=url&exchars=1200&exlimit=20&exintro=1&explaintext=1&exsectionformat=plain&piprop=thumbnail%7Cname&pithumbsize=100&pilimit=50&pilicense=any"
    wiki_data_string += "&pageids="+"|".join(page_ids) 

    res = requests.get(wiki_data_string).json()

    initial_passages = [None] * len(page_ids)
    initial_structured = [None] * len(page_ids)

    # Get infos in the correct format
    for page in res["query"]["pages"]:
        # Use the initial list operator ranking, because the second request gives a different result ordering, 
        # which is not relevance based (!)
        actual_ranking_idx = page_ids.index(str(page['pageid']))
        initial_structured[actual_ranking_idx] = {
            "title": page["title"],
            "url": page["fullurl"],
            "text": page["extract"],
            "img": "" if "thumbnail" not in page else page["thumbnail"]["source"],
        }
        initial_passages[actual_ranking_idx] = page["title"] + " " + page["extract"]

    return initial_structured, initial_passages

@api_app.post("/re-rank")
async def re_rank(data: ReRankInput):
    if len(data.passages) == 0:
        return {"results": []}
    rerank_time = default_timer()
    re_ranked_result = co.rerank(
        model="rerank-english-v2.0",
        query=data.query, 
        documents=data.passages)
    serializable = []
    for res in re_ranked_result:
        serializable.append(
            {
                "index": int(res.index),
                "relevance_score": round(float(res.relevance_score), 3)
                }
            )
    re_ranked_result = {"results": serializable}
    
    rerank_time = round((default_timer() - rerank_time) * 1000, 1)
    
    _id = str(uuid.uuid4())
    re_ranked_result["id"] = _id
    return re_ranked_result

@api_app.post("/search-and-rerank-wiki")
async def rerank_wiki(data: SearchInput):
    source_time = default_timer()
    initial_structured, initial_passages = search_wikipedia(data.query)

    if len(initial_structured) == 0:
        return {"results":[],"struct_data_initial_rank":[]}
    source_time = round((default_timer() - source_time) * 1000, 1)

    rerank_time = default_timer()

    re_ranked_result = co.rerank(
        model="rerank-english-v2.0",
        query=data.query, 
        documents=initial_passages)
    serializable = []
    for res in re_ranked_result:
        serializable.append({"index": int(res.index),
                     "relevance_score": round(float(res.relevance_score), 3)})
    re_ranked_result = {"results": serializable}

    rerank_time = round((default_timer() - rerank_time) * 1000, 1)

    _id = str(uuid.uuid4())

    with open(LOG_LOCATION, "a", encoding="utf8") as log:
        log.write(json.dumps({
            "id":_id,
            "session_id": data.session_id,
            "timestamp": str(datetime.now()),
            "input":{
                "query": data.query,
                "char_length": len(data.query)
            },
            "timing-ms":{
                "re-rank": rerank_time
            },
            "source": data.source,
            "result": re_ranked_result
        },ensure_ascii=False) + "\n")


    re_ranked_result["id"]=_id
    re_ranked_result["timing"] = {"source": source_time,
                                  "rerank": rerank_time}
    re_ranked_result["struct_data_initial_rank"] = initial_structured
    return re_ranked_result

@api_app.post("/feedback")
async def feedback(data: FeedbackInput):

    with open(FEEDBACK_LOCATION, "a", encoding="utf8") as log:
        log.write(json.dumps({
            "query_id": data.query_id,
            "session_id": data.session_id,
            "timestamp": str(datetime.now()),
            "feedback": data.feedback
        })+"\n")

    return {}


app = FastAPI(title="main app")

app.mount("/api", api_app)
app.mount("/", StaticFiles(directory="ui", html=True), name="ui")
